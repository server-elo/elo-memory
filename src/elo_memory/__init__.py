"""
Elo-Memory: Bio-inspired episodic memory system for AI agents.

Implements EM-LLM (ICLR 2025) with 8 core components:
- Bayesian Surprise Detection
- Event Segmentation
- Episodic Storage
- Two-Stage Retrieval
- Memory Consolidation
- Forgetting & Decay
- Interference Resolution
- Online Learning

License: BSL-1.1
"""

import logging

__version__ = "0.2.2"
__author__ = "Lorenc Ndoj, Elvi Zekaj"
__license__ = "BSL-1.1"

# Package-level logger. Defaults to NullHandler (silent) per Python best practice.
# Users can configure via: logging.getLogger("elo_memory").setLevel(logging.DEBUG)
logger = logging.getLogger("elo_memory")
logger.addHandler(logging.NullHandler())

try:
    from .surprise.bayesian_surprise import BayesianSurpriseEngine, SurpriseConfig
    from .segmentation.event_segmenter import EventSegmenter, SegmentationConfig
    from .memory.episodic_store import EpisodicMemoryStore, EpisodicMemoryConfig, Episode
    from .retrieval.two_stage_retriever import TwoStageRetriever, RetrievalConfig
    from .consolidation.memory_consolidation import MemoryConsolidationEngine, ConsolidationConfig
    from .memory.forgetting import ForgettingEngine, ForgettingConfig
    from .memory.interference import InterferenceResolver, InterferenceConfig
    from .online_learning import OnlineLearner, OnlineLearningConfig
    from .memory.user_memory import UserMemory
    from .brain import EloBrain

    __all__ = [
        "BayesianSurpriseEngine",
        "SurpriseConfig",
        "EventSegmenter",
        "SegmentationConfig",
        "EpisodicMemoryStore",
        "EpisodicMemoryConfig",
        "Episode",
        "TwoStageRetriever",
        "RetrievalConfig",
        "MemoryConsolidationEngine",
        "ConsolidationConfig",
        "ForgettingEngine",
        "ForgettingConfig",
        "InterferenceResolver",
        "InterferenceConfig",
        "OnlineLearner",
        "OnlineLearningConfig",
        "UserMemory",
        "EloBrain",
        "__version__",
        "__author__",
        "__license__",
    ]
except ImportError:
    __all__ = ["__version__", "__author__", "__license__"]
