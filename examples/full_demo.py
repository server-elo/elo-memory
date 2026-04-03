"""
Elo Memory — Quick Demo
========================

Minimal example showing EloBrain in action.

    pip install elo-memory
    python examples/full_demo.py
"""

import shutil
from elo_memory import EloBrain


def main():
    shutil.rmtree("./demo_memories", ignore_errors=True)
    brain = EloBrain("user", persistence_path="./demo_memories")

    def echo_llm(prompt):
        # In real usage, replace with your LLM call
        return "Got it, I'll remember that."

    # Store some facts
    brain.think("I'm Alex, a data scientist at Netflix", echo_llm)
    brain.think("I work with PyTorch and Spark", echo_llm)
    brain.think("My email is alex@netflix.com", echo_llm)

    # What does the brain know?
    state = brain.what_i_know()
    print("Knowledge Base:")
    for k, v in state["knowledge"].items():
        print(f"  {k}: {v}")

    # Recall
    print("\nRecall 'What tools?':")
    for text, score in brain._memory.recall("What tools do I use?", k=3):
        print(f"  {score:.2f}: {text}")

    brain.close()
    shutil.rmtree("./demo_memories", ignore_errors=True)
    print("\nDone!")


if __name__ == "__main__":
    main()
