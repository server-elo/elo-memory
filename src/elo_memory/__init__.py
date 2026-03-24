   1 | """
   2 | Elo-Memory: Bio-inspired episodic memory system for AI agents.
   3 | 
   4 | Implements EM-LLM (ICLR 2025) with 8 core components:
   5 | - Bayesian Surprise Detection
   6 | - Event Segmentation  
   7 | - Episodic Storage
   8 | - Two-Stage Retrieval
   9 | - Memory Consolidation
  10 | - Forgetting & Decay
  11 | - Interference Resolution
  12 | - Online Learning
  13 | 
  14 | License: MIT
  15 | """
  16 | 
  17 | __version__ = "0.1.1"
  18 | __author__ = "Lorenc Ndoj, Elvi Zekaj"
  19 | __license__ = "MIT"
  20 | 
  21 | # Import from component modules
  22 | # These assume the src/ directory is in PYTHONPATH or installed as package
  23 | 
  24 | try:
  25 |     from .surprise.bayesian_surprise import BayesianSurpriseEngine, SurpriseConfig