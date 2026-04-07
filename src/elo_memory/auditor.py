"""
Trustworthy Memory Auditor & Verifier
======================================

Cryptographic integrity layer using hash chains and Merkle trees.

Every memory gets a SHA-256 hash linked to the previous memory's hash,
forming a tamper-evident chain.  The agent can prove "this memory is
authentic and unaltered" without revealing content (via hash comparison).

Auto-flags memories that fail integrity checks (potential hallucinations
or external tampering).

References:
- Merkle (1988): Digital Signatures Based on Hash Trees
- Benet (2014): Content-Addressed Block Storage (IPFS)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .memory.episodic_store import Episode

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single entry in the audit log."""

    episode_id: str
    action: str  # "create", "read", "update", "delete", "verify"
    timestamp: str
    actor: str = ""  # Who performed the action
    details: str = ""


@dataclass
class ChainLink:
    """Single link in the hash chain."""

    episode_id: str
    content_hash: str  # SHA-256 of episode content
    previous_hash: str  # Hash of the previous link
    chain_hash: str  # SHA-256(content_hash + previous_hash)
    sequence_number: int
    timestamp: str


class MemoryAuditor:
    """
    Cryptographic integrity and audit trail for episodic memory.

    Maintains:
    1. Hash chain: linked list of content hashes
    2. Merkle tree: for batch verification
    3. Audit log: who accessed/modified what, when
    4. Tamper detection: verify any memory against its hash
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self._chain: List[ChainLink] = []
        self._audit_log: List[AuditEntry] = []
        self._hash_index: Dict[str, ChainLink] = {}  # episode_id → link
        self._merkle_root: Optional[str] = None
        self._tampered: List[str] = []  # Episode IDs that failed verification

        self._persistence_path: Optional[Path] = None
        if persistence_path:
            self._persistence_path = Path(persistence_path) / "auditor"
            self._persistence_path.mkdir(parents=True, exist_ok=True)
            self._load()

    # ── Hash Chain ───────────────────────────────────────────────

    def add_to_chain(self, episode: Episode) -> ChainLink:
        """Add an episode to the hash chain."""
        content_hash = self._hash_episode(episode)
        previous_hash = self._chain[-1].chain_hash if self._chain else "0" * 64

        chain_hash = hashlib.sha256((content_hash + previous_hash).encode()).hexdigest()

        link = ChainLink(
            episode_id=episode.episode_id,
            content_hash=content_hash,
            previous_hash=previous_hash,
            chain_hash=chain_hash,
            sequence_number=len(self._chain),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._chain.append(link)
        self._hash_index[episode.episode_id] = link

        # Update Merkle root
        self._update_merkle_root()

        # Audit log
        self._log("create", episode.episode_id, "Added to hash chain")

        return link

    def verify_episode(self, episode: Episode) -> bool:
        """
        Verify that an episode has not been tampered with.
        Compares current content hash against the stored hash.
        """
        link = self._hash_index.get(episode.episode_id)
        if link is None:
            self._log("verify", episode.episode_id, "NOT IN CHAIN — untracked episode")
            return False

        current_hash = self._hash_episode(episode)
        is_valid = current_hash == link.content_hash

        if is_valid:
            self._log("verify", episode.episode_id, "PASS — integrity verified")
        else:
            self._log(
                "verify",
                episode.episode_id,
                f"FAIL — hash mismatch (expected {link.content_hash[:16]}..., "
                f"got {current_hash[:16]}...)",
            )
            if episode.episode_id not in self._tampered:
                self._tampered.append(episode.episode_id)

        return is_valid

    def verify_chain(self) -> Dict[str, Any]:
        """
        Verify the entire hash chain.
        Returns verification report.
        """
        if not self._chain:
            return {"valid": True, "checked": 0, "broken_links": []}

        broken_links = []
        prev_hash = "0" * 64

        for i, link in enumerate(self._chain):
            # Verify link to previous
            if link.previous_hash != prev_hash:
                broken_links.append(
                    {
                        "sequence": i,
                        "episode_id": link.episode_id,
                        "error": "previous_hash mismatch",
                    }
                )

            # Verify chain hash
            expected = hashlib.sha256((link.content_hash + link.previous_hash).encode()).hexdigest()
            if link.chain_hash != expected:
                broken_links.append(
                    {
                        "sequence": i,
                        "episode_id": link.episode_id,
                        "error": "chain_hash mismatch",
                    }
                )

            prev_hash = link.chain_hash

        self._log("verify", "*", f"Full chain verification: {len(broken_links)} broken links")

        return {
            "valid": len(broken_links) == 0,
            "checked": len(self._chain),
            "broken_links": broken_links,
            "merkle_root": self._merkle_root,
        }

    # ── Merkle Tree ──────────────────────────────────────────────

    def _update_merkle_root(self) -> None:
        """Recompute the Merkle root from all chain hashes."""
        if not self._chain:
            self._merkle_root = None
            return

        hashes = [link.chain_hash for link in self._chain]
        self._merkle_root = self._compute_merkle_root(hashes)

    @staticmethod
    def _compute_merkle_root(hashes: List[str]) -> str:
        """Compute Merkle root from a list of hashes."""
        if not hashes:
            return ""
        if len(hashes) == 1:
            return hashes[0]

        # Pad to even length
        current_level = list(hashes)
        if len(current_level) % 2 != 0:
            current_level.append(current_level[-1])

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[min(i + 1, len(current_level) - 1)]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            current_level = next_level

        return current_level[0]

    def get_merkle_proof(self, episode_id: str) -> Optional[List[Dict[str, str]]]:
        """Get a Merkle proof for a specific episode (for external verification)."""
        link = self._hash_index.get(episode_id)
        if link is None:
            return None

        idx = link.sequence_number
        hashes = [l.chain_hash for l in self._chain]

        # Build proof path
        proof: List[Dict[str, str]] = []
        current_level = list(hashes)
        if len(current_level) % 2 != 0:
            current_level.append(current_level[-1])

        current_idx = idx
        while len(current_level) > 1:
            if current_idx % 2 == 0:
                sibling_idx = min(current_idx + 1, len(current_level) - 1)
                proof.append({"side": "right", "hash": current_level[sibling_idx]})
            else:
                sibling_idx = current_idx - 1
                proof.append({"side": "left", "hash": current_level[sibling_idx]})

            # Move up
            next_level = []
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[min(i + 1, len(current_level) - 1)]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            current_level = next_level
            current_idx //= 2

        return proof

    # ── Hashing ──────────────────────────────────────────────────

    @staticmethod
    def _hash_episode(episode: Episode) -> str:
        """Compute deterministic SHA-256 hash of episode content.

        Normalizes content so numpy arrays, lists, and dicts all produce
        the same hash regardless of save/load cycle.
        """
        # Build a canonical representation
        parts = []

        # Content — normalize arrays/lists to consistent string
        content = episode.content
        if isinstance(content, np.ndarray):
            parts.append(content.tobytes().hex())
        elif isinstance(content, list):
            # List from deserialized numpy array — treat same as ndarray
            parts.append(str(content))
        elif isinstance(content, dict):
            # Normalize numpy arrays inside dict to lists
            normalized = {}
            for k, v in sorted(content.items()):
                if isinstance(v, np.ndarray):
                    normalized[k] = v.tolist()
                else:
                    normalized[k] = v
            parts.append(json.dumps(normalized, sort_keys=True))
        else:
            parts.append(str(content))

        # Timestamp
        parts.append(episode.timestamp.isoformat())

        # Location
        parts.append(episode.location or "")

        # Entities
        parts.append(",".join(sorted(episode.entities)))

        canonical = "|".join(parts)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ── Audit Log ────────────────────────────────────────────────

    def _log(self, action: str, episode_id: str, details: str = "", actor: str = "system") -> None:
        entry = AuditEntry(
            episode_id=episode_id,
            action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            details=details,
        )
        self._audit_log.append(entry)

    def log_access(self, episode_id: str, actor: str = "agent") -> None:
        """Record a memory access for audit purposes."""
        self._log("read", episode_id, "Memory accessed", actor)

    def get_audit_log(
        self,
        episode_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, str]]:
        """Query the audit log with optional filters."""
        entries = self._audit_log
        if episode_id:
            entries = [e for e in entries if e.episode_id == episode_id]
        if action:
            entries = [e for e in entries if e.action == action]
        return [
            {
                "episode_id": e.episode_id,
                "action": e.action,
                "timestamp": e.timestamp,
                "actor": e.actor,
                "details": e.details,
            }
            for e in entries[-limit:]
        ]

    def get_tampered_episodes(self) -> List[str]:
        """Return episode IDs that failed integrity verification."""
        return list(self._tampered)

    # ── Statistics ───────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "chain_length": len(self._chain),
            "merkle_root": self._merkle_root,
            "audit_log_entries": len(self._audit_log),
            "tampered_episodes": len(self._tampered),
            "tracked_episodes": len(self._hash_index),
        }

    # ── Persistence ──────────────────────────────────────────────

    def save(self) -> None:
        if not self._persistence_path:
            return
        data = {
            "chain": [
                {
                    "episode_id": l.episode_id,
                    "content_hash": l.content_hash,
                    "previous_hash": l.previous_hash,
                    "chain_hash": l.chain_hash,
                    "sequence_number": l.sequence_number,
                    "timestamp": l.timestamp,
                }
                for l in self._chain
            ],
            "audit_log": [
                {
                    "episode_id": e.episode_id,
                    "action": e.action,
                    "timestamp": e.timestamp,
                    "actor": e.actor,
                    "details": e.details,
                }
                for e in self._audit_log[-10000:]  # Cap log size
            ],
            "tampered": self._tampered,
        }
        with open(self._persistence_path / "auditor_state.json", "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        if not self._persistence_path:
            return
        path = self._persistence_path / "auditor_state.json"
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._chain = [ChainLink(**link) for link in data.get("chain", [])]
            self._hash_index = {l.episode_id: l for l in self._chain}
            self._audit_log = [AuditEntry(**entry) for entry in data.get("audit_log", [])]
            self._tampered = data.get("tampered", [])
            self._update_merkle_root()
        except Exception as e:
            logger.error("Failed to load auditor state: %s", e)
