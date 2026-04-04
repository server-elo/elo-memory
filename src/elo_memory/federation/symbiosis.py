"""
Cross-Agent Memory Symbiosis Network
=====================================

Secure federated system where agents contribute anonymized, differentially-
private "memory modules" to a shared evolutionary pool.  Agents can export
memories, import from others, and revoke previously shared data.

Uses evolutionary selection: memories that are useful to many agents get
promoted; unused ones decay.

References:
- McMahan et al. (2017): Communication-Efficient Learning (FedAvg)
- Kairouz et al. (2021): Advances in Distributed/Federated ML
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .privacy import DifferentialPrivacy, PrivacyAccountant, PrivacyConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryModule:
    """
    A shareable memory unit with provenance tracking.

    Contains a noised embedding (never the raw content), category tags,
    and provenance metadata.
    """
    module_id: str
    embedding: np.ndarray           # Differentially private
    category: str = ""              # Topic/domain category
    utility_score: float = 0.0      # How useful this has been to others
    adoption_count: int = 0         # How many agents adopted this
    source_agent_hash: str = ""     # Anonymous source identifier
    created_at: str = ""
    revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "embedding": self.embedding.tolist(),
            "category": self.category,
            "utility_score": self.utility_score,
            "adoption_count": self.adoption_count,
            "source_agent_hash": self.source_agent_hash,
            "created_at": self.created_at,
            "revoked": self.revoked,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> MemoryModule:
        return cls(
            module_id=data["module_id"],
            embedding=np.array(data["embedding"]),
            category=data.get("category", ""),
            utility_score=data.get("utility_score", 0.0),
            adoption_count=data.get("adoption_count", 0),
            source_agent_hash=data.get("source_agent_hash", ""),
            created_at=data.get("created_at", ""),
            revoked=data.get("revoked", False),
            metadata=data.get("metadata", {}),
        )


class MemoryPool:
    """
    Shared evolutionary pool of memory modules.

    Modules are scored by utility (how often they're adopted and rated
    positively).  Low-utility modules decay and are eventually pruned.
    """

    def __init__(self, pool_path: Optional[str] = None, max_modules: int = 10000):
        self.modules: Dict[str, MemoryModule] = {}
        self.max_modules = max_modules
        self._pool_path = Path(pool_path) if pool_path else None
        if self._pool_path:
            self._pool_path.mkdir(parents=True, exist_ok=True)
            self._load()

    def contribute(self, module: MemoryModule) -> bool:
        """Add a memory module to the pool."""
        if module.revoked:
            return False
        if len(self.modules) >= self.max_modules:
            self._evict_lowest()
        self.modules[module.module_id] = module
        return True

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        category: Optional[str] = None,
    ) -> List[MemoryModule]:
        """Find the k most relevant modules by embedding similarity."""
        candidates = [
            m for m in self.modules.values()
            if not m.revoked and (category is None or m.category == category)
        ]
        if not candidates:
            return []

        # Compute similarities
        scored = []
        for mod in candidates:
            sim = float(np.dot(query_embedding, mod.embedding))
            scored.append((mod, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored[:k]]

    def adopt(self, module_id: str, adopter_agent_hash: str) -> Optional[MemoryModule]:
        """Mark a module as adopted. Returns the module if available."""
        mod = self.modules.get(module_id)
        if mod is None or mod.revoked:
            return None
        mod.adoption_count += 1
        mod.utility_score += 1.0
        return mod

    def rate(self, module_id: str, score: float) -> None:
        """Rate a module's utility (positive or negative)."""
        mod = self.modules.get(module_id)
        if mod:
            mod.utility_score += score

    def revoke(self, module_id: str, source_agent_hash: str) -> bool:
        """Revoke a previously shared module (GDPR right to erasure)."""
        mod = self.modules.get(module_id)
        if mod is None:
            return False
        if mod.source_agent_hash != source_agent_hash:
            logger.warning("Revocation denied: agent hash mismatch")
            return False
        mod.revoked = True
        mod.embedding = np.zeros_like(mod.embedding)  # Zero out data
        return True

    def decay_all(self, factor: float = 0.95) -> None:
        """Decay utility scores of all modules (evolutionary pressure)."""
        for mod in self.modules.values():
            mod.utility_score *= factor

    def _evict_lowest(self) -> None:
        """Remove the lowest-utility module to make room."""
        if not self.modules:
            return
        worst_id = min(self.modules, key=lambda k: self.modules[k].utility_score)
        del self.modules[worst_id]

    def get_statistics(self) -> Dict[str, Any]:
        active = [m for m in self.modules.values() if not m.revoked]
        return {
            "total_modules": len(self.modules),
            "active_modules": len(active),
            "revoked_modules": len(self.modules) - len(active),
            "avg_utility": float(np.mean([m.utility_score for m in active])) if active else 0,
            "avg_adoption": float(np.mean([m.adoption_count for m in active])) if active else 0,
            "categories": list({m.category for m in active if m.category}),
        }

    def save(self) -> None:
        if not self._pool_path:
            return
        data = {mid: m.to_dict() for mid, m in self.modules.items()}
        with open(self._pool_path / "pool.json", "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        if not self._pool_path:
            return
        path = self._pool_path / "pool.json"
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.modules = {k: MemoryModule.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.error("Failed to load memory pool: %s", e)


class FederationClient:
    """
    Agent-side federation client.

    Handles exporting local memories with differential privacy,
    importing useful modules from the pool, and managing revocations.
    """

    def __init__(
        self,
        agent_id: str,
        pool: MemoryPool,
        privacy_config: Optional[PrivacyConfig] = None,
        privacy_budget: float = 10.0,
    ):
        self.agent_id = agent_id
        self.agent_hash = hashlib.sha256(agent_id.encode()).hexdigest()[:16]
        self.pool = pool
        self._dp = DifferentialPrivacy(privacy_config)
        self._accountant = PrivacyAccountant(total_budget=privacy_budget)
        self._exported_ids: Set[str] = set()
        self._imported_ids: Set[str] = set()

    def export_memory(
        self,
        embedding: np.ndarray,
        category: str = "",
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Export a memory to the shared pool with differential privacy.
        Returns module_id or None if privacy budget is exhausted.

        Privacy budget is only spent AFTER the contribution succeeds,
        so failed exports don't waste privacy credits.
        """
        epsilon = self._dp.config.epsilon
        # Check budget availability first (don't spend yet)
        if self._accountant.remaining_budget < epsilon:
            return None

        # Add noise
        noised_embedding = self._dp.add_noise(embedding)

        # Anonymize metadata
        safe_meta = {}
        if metadata:
            for k, v in metadata.items():
                if k in ("topics", "category", "timestamp"):
                    safe_meta[k] = v
                elif isinstance(v, str):
                    safe_meta[k] = self._dp.anonymize_text(v)

        module = MemoryModule(
            module_id=f"mod_{uuid.uuid4().hex[:12]}",
            embedding=noised_embedding,
            category=category,
            source_agent_hash=self.agent_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=safe_meta,
        )

        if self.pool.contribute(module):
            # Only spend budget on successful contribution
            if self._accountant.spend(epsilon, f"export:{category}"):
                self._exported_ids.add(module.module_id)
                return module.module_id
            # Budget exhausted after contribution succeeded — still keep the module
            self._exported_ids.add(module.module_id)
            return module.module_id
        return None

    def import_relevant(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        category: Optional[str] = None,
    ) -> List[MemoryModule]:
        """Import the most relevant modules from the pool."""
        modules = self.pool.query(query_embedding, k=k, category=category)

        # Filter out own exports and already imported
        result = []
        for mod in modules:
            if mod.source_agent_hash == self.agent_hash:
                continue
            if mod.module_id in self._imported_ids:
                continue
            self.pool.adopt(mod.module_id, self.agent_hash)
            self._imported_ids.add(mod.module_id)
            result.append(mod)

        return result

    def rate_module(self, module_id: str, score: float) -> None:
        """Rate an imported module's usefulness."""
        self.pool.rate(module_id, score)

    def revoke_all(self) -> int:
        """Revoke all exported modules (right to be forgotten)."""
        count = 0
        for mid in list(self._exported_ids):
            if self.pool.revoke(mid, self.agent_hash):
                count += 1
        self._exported_ids.clear()
        return count

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_hash": self.agent_hash,
            "exported_count": len(self._exported_ids),
            "imported_count": len(self._imported_ids),
            "privacy": self._accountant.get_report(),
        }
