"""
Causal Inference Memory Engine
==============================

Automatically constructs and evolves a dynamic causal DAG from every
experience.  Supports counterfactual queries ("What if I had done X
differently?") and auto-detects contradictions or outdated causes when
new evidence arrives.

The graph is built incrementally — each new memory is scanned for
causal language, entity transitions, and temporal ordering.  Stale
links are weakened over time; contradictions trigger alerts.

References:
- Pearl (2009): Causality — Models, Reasoning, and Inference
- Halpern & Pearl (2005): Causes and Explanations
- Spirtes et al. (2000): Causation, Prediction, and Search
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ── Causal patterns ──────────────────────────────────────────────

_CAUSAL_PATTERNS = [
    # "X because Y"
    (re.compile(r"(.+?)\s+because\s+(.+?)(?:\.|,|$)", re.I), "effect", "cause"),
    # "due to X, Y"
    (re.compile(r"due\s+to\s+(.+?),\s*(.+?)(?:\.|,|$)", re.I), "cause", "effect"),
    # "X so that Y"
    (re.compile(r"(.+?)\s+so\s+that\s+(.+?)(?:\.|,|$)", re.I), "cause", "effect"),
    # "X in order to Y"
    (re.compile(r"(.+?)\s+in\s+order\s+to\s+(.+?)(?:\.|,|$)", re.I), "cause", "effect"),
    # "X led to Y"
    (re.compile(r"(.+?)\s+led\s+to\s+(.+?)(?:\.|,|$)", re.I), "cause", "effect"),
    # "X resulted in Y"
    (re.compile(r"(.+?)\s+resulted\s+in\s+(.+?)(?:\.|,|$)", re.I), "cause", "effect"),
    # "X caused Y"
    (re.compile(r"(.+?)\s+caused\s+(.+?)(?:\.|,|$)", re.I), "cause", "effect"),
    # "after X, Y"
    (re.compile(r"after\s+(.+?),\s*(.+?)(?:\.|$)", re.I), "cause", "effect"),
    # "X, therefore Y"
    (re.compile(r"(.+?),?\s+therefore\s+(.+?)(?:\.|$)", re.I), "cause", "effect"),
    # "since X, Y"
    (re.compile(r"since\s+(.+?),\s*(.+?)(?:\.|$)", re.I), "cause", "effect"),
]


@dataclass
class CausalLink:
    """Single causal relationship."""
    cause: str
    effect: str
    strength: float = 1.0
    evidence_count: int = 1
    source_episodes: List[str] = field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class CausalEngineConfig:
    """Configuration for the causal inference engine."""
    decay_per_day: float = 0.02           # Strength decay per day without reinforcement
    contradiction_threshold: float = 0.5  # Min strength for contradiction detection
    min_link_strength: float = 0.1        # Links below this are pruned
    max_graph_nodes: int = 5000           # Safety cap


class CausalInferenceEngine:
    """
    Dynamic causal graph that grows from episodic memory.

    Nodes are concepts/events (short text), edges are causal links
    with strength, evidence count, and provenance.
    """

    def __init__(self, config: Optional[CausalEngineConfig] = None):
        self.config = config or CausalEngineConfig()
        self.graph = nx.DiGraph()
        self._contradictions: List[Dict[str, Any]] = []

    # ── Ingestion ────────────────────────────────────────────────

    def ingest(self, text: str, episode_id: str = "") -> List[CausalLink]:
        """Extract causal links from text and add to graph."""
        links = self._extract_links(text, episode_id)
        for link in links:
            self._add_link(link)
        # Check for new contradictions
        self._detect_contradictions_incremental(links)
        return links

    def _extract_links(self, text: str, episode_id: str) -> List[CausalLink]:
        """Extract causal relationships from natural language."""
        links: List[CausalLink] = []
        now = datetime.now(timezone.utc).isoformat()

        for pattern, first_role, second_role in _CAUSAL_PATTERNS:
            for m in pattern.finditer(text):
                parts = {first_role: m.group(1).strip(), second_role: m.group(2).strip()}
                cause = parts["cause"]
                effect = parts["effect"]
                # Skip very short or very long extractions
                if len(cause) < 3 or len(effect) < 3:
                    continue
                if len(cause) > 200 or len(effect) > 200:
                    continue
                links.append(CausalLink(
                    cause=cause,
                    effect=effect,
                    source_episodes=[episode_id] if episode_id else [],
                    first_seen=now,
                    last_seen=now,
                ))

        return links

    def _add_link(self, link: CausalLink):
        """Add or strengthen a causal link in the graph."""
        if self.graph.number_of_nodes() >= self.config.max_graph_nodes:
            self._prune_weak_links()

        cause_key = self._normalize(link.cause)
        effect_key = self._normalize(link.effect)

        # Add/update nodes
        for key, label in [(cause_key, link.cause), (effect_key, link.effect)]:
            if not self.graph.has_node(key):
                self.graph.add_node(key, label=label, mentions=1)
            else:
                self.graph.nodes[key]["mentions"] = self.graph.nodes[key].get("mentions", 0) + 1

        # Add/update edge
        if self.graph.has_edge(cause_key, effect_key):
            edge = self.graph.edges[cause_key, effect_key]
            edge["strength"] = min(10.0, edge["strength"] + 1.0)
            edge["evidence_count"] += 1
            edge["last_seen"] = link.last_seen
            edge["source_episodes"] = list(
                set(edge.get("source_episodes", []) + link.source_episodes)
            )
        else:
            self.graph.add_edge(
                cause_key,
                effect_key,
                strength=link.strength,
                evidence_count=link.evidence_count,
                source_episodes=link.source_episodes,
                first_seen=link.first_seen,
                last_seen=link.last_seen,
            )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for node keys."""
        return re.sub(r"\s+", " ", text.lower().strip())[:120]

    # ── Queries ──────────────────────────────────────────────────

    def query_causes(self, effect: str, depth: int = 3) -> List[Dict[str, Any]]:
        """Find what caused a given effect (trace back through the graph)."""
        key = self._normalize(effect)
        if key not in self.graph:
            return []

        results = []
        visited = set()
        queue = [(key, 0)]

        while queue:
            node, d = queue.pop(0)
            if d >= depth or node in visited:
                continue
            visited.add(node)

            for pred in self.graph.predecessors(node):
                edge = self.graph.edges[pred, node]
                results.append({
                    "cause": self.graph.nodes[pred].get("label", pred),
                    "effect": self.graph.nodes[node].get("label", node),
                    "strength": edge["strength"],
                    "evidence_count": edge["evidence_count"],
                    "depth": d + 1,
                })
                queue.append((pred, d + 1))

        results.sort(key=lambda x: x["strength"], reverse=True)
        return results

    def query_effects(self, cause: str, depth: int = 3) -> List[Dict[str, Any]]:
        """Find what a given cause led to (trace forward through the graph)."""
        key = self._normalize(cause)
        if key not in self.graph:
            return []

        results = []
        visited = set()
        queue = [(key, 0)]

        while queue:
            node, d = queue.pop(0)
            if d >= depth or node in visited:
                continue
            visited.add(node)

            for succ in self.graph.successors(node):
                edge = self.graph.edges[node, succ]
                results.append({
                    "cause": self.graph.nodes[node].get("label", node),
                    "effect": self.graph.nodes[succ].get("label", succ),
                    "strength": edge["strength"],
                    "evidence_count": edge["evidence_count"],
                    "depth": d + 1,
                })
                queue.append((succ, d + 1))

        results.sort(key=lambda x: x["strength"], reverse=True)
        return results

    def counterfactual(self, removed_cause: str) -> Dict[str, Any]:
        """
        Answer: "What if *removed_cause* had not happened?"

        Removes the node and returns all downstream effects that would
        be lost (nodes only reachable through the removed cause).
        """
        key = self._normalize(removed_cause)
        if key not in self.graph:
            return {"removed": removed_cause, "lost_effects": [], "preserved_effects": []}

        # Find all descendants of this node
        descendants = nx.descendants(self.graph, key)

        # Check which descendants are ONLY reachable through this node
        lost = []
        preserved = []
        temp_graph = self.graph.copy()
        temp_graph.remove_node(key)

        # Find all root nodes (nodes with no predecessors)
        roots = [n for n in temp_graph.nodes() if temp_graph.in_degree(n) == 0]

        # BFS from all roots to see what's still reachable
        still_reachable: Set[str] = set()
        for root in roots:
            still_reachable.update(nx.descendants(temp_graph, root))
            still_reachable.add(root)

        for desc in descendants:
            label = self.graph.nodes[desc].get("label", desc)
            if desc in still_reachable:
                preserved.append(label)
            else:
                lost.append(label)

        return {
            "removed": removed_cause,
            "lost_effects": lost,
            "preserved_effects": preserved,
            "total_downstream": len(descendants),
        }

    # ── Contradiction Detection ──────────────────────────────────

    def _detect_contradictions_incremental(self, new_links: List[CausalLink]):
        """Check if new links contradict existing ones."""
        for link in new_links:
            cause_key = self._normalize(link.cause)
            effect_key = self._normalize(link.effect)

            # Check for reverse edge (A→B and B→A = cycle/contradiction)
            if self.graph.has_edge(effect_key, cause_key):
                existing = self.graph.edges[effect_key, cause_key]
                if existing["strength"] >= self.config.contradiction_threshold:
                    contradiction = {
                        "type": "reverse_causation",
                        "existing": {
                            "cause": self.graph.nodes[effect_key].get("label", effect_key),
                            "effect": self.graph.nodes[cause_key].get("label", cause_key),
                            "strength": existing["strength"],
                        },
                        "new": {"cause": link.cause, "effect": link.effect},
                        "detected_at": datetime.now(timezone.utc).isoformat(),
                    }
                    self._contradictions.append(contradiction)
                    logger.warning("Causal contradiction detected: %s", contradiction)

    def detect_all_contradictions(self) -> List[Dict[str, Any]]:
        """Find all contradictions in the causal graph (cycles, reversals)."""
        contradictions = list(self._contradictions)

        # Find cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            for cycle in cycles[:20]:  # Cap at 20
                labels = [self.graph.nodes[n].get("label", n) for n in cycle]
                contradictions.append({
                    "type": "cycle",
                    "nodes": labels,
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                })
        except nx.NetworkXError:
            pass

        return contradictions

    def get_contradictions(self) -> List[Dict[str, Any]]:
        """Return detected contradictions."""
        return list(self._contradictions)

    # ── Maintenance ──────────────────────────────────────────────

    def decay_strengths(self, days_elapsed: float = 1.0):
        """Weaken all links by time-based decay."""
        decay = self.config.decay_per_day * days_elapsed
        to_remove = []
        for u, v, data in self.graph.edges(data=True):
            data["strength"] = max(0, data["strength"] - decay)
            if data["strength"] < self.config.min_link_strength:
                to_remove.append((u, v))
        for u, v in to_remove:
            self.graph.remove_edge(u, v)
        # Remove orphaned nodes
        orphans = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        self.graph.remove_nodes_from(orphans)

    def _prune_weak_links(self):
        """Remove weakest links when graph is too large."""
        edges = sorted(
            self.graph.edges(data=True), key=lambda e: e[2]["strength"]
        )
        n_remove = max(1, len(edges) // 10)
        for u, v, _ in edges[:n_remove]:
            self.graph.remove_edge(u, v)
        orphans = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        self.graph.remove_nodes_from(orphans)

    # ── Stats ────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "contradictions": len(self._contradictions),
            "strongest_links": [
                {
                    "cause": self.graph.nodes[u].get("label", u),
                    "effect": self.graph.nodes[v].get("label", v),
                    "strength": d["strength"],
                }
                for u, v, d in sorted(
                    self.graph.edges(data=True),
                    key=lambda e: e[2]["strength"],
                    reverse=True,
                )[:10]
            ],
        }

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: Path):
        data = {
            "nodes": [
                {"id": n, **self.graph.nodes[n]}
                for n in self.graph.nodes()
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ],
            "contradictions": self._contradictions,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path):
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.graph.clear()
            for node in data.get("nodes", []):
                nid = node.pop("id")
                self.graph.add_node(nid, **node)
            for edge in data.get("edges", []):
                src = edge.pop("source")
                tgt = edge.pop("target")
                self.graph.add_edge(src, tgt, **edge)
            self._contradictions = data.get("contradictions", [])
        except Exception as e:
            logger.error("Failed to load causal graph: %s", e)
