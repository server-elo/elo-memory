"""
Complete Elo Memory Demo — EloBrain + KnowledgeBase + Intelligence
==================================================================

Shows the full system: store, recall, KB, conflicts, gaps, causal reasoning.

    pip install elo-memory
    python examples/complete_system.py
"""

import shutil
from elo_memory import EloBrain


def main():
    print("=" * 60)
    print(" ELO MEMORY — Complete System Demo")
    print("=" * 60)

    shutil.rmtree("./demo_memories", ignore_errors=True)
    brain = EloBrain("demo_user", persistence_path="./demo_memories")

    def mock_llm(prompt):
        return "Understood."

    # ── Day 1: Introduction ──
    print("\n--- Day 1: Introduction ---")
    brain.think("I'm Sarah Chen, senior engineer at Shopify", mock_llm)
    brain.think("My email is sarah.chen@shopify.com", mock_llm)
    brain.think("We use Django backend, React frontend, PostgreSQL database", mock_llm)
    brain.think("Our team is 8 engineers, manager is Tom", mock_llm)

    state = brain.what_i_know()
    print(f"  KB facts: {len(state['knowledge'])}")
    for k, v in sorted(state["knowledge"].items()):
        print(f"    {k}: {v}")

    # ── Day 5: Things change ──
    print("\n--- Day 5: Changes ---")
    brain.think("Switched from Django to FastAPI because Django was too slow for websockets", mock_llm)
    brain.think("Moved from AWS ECS to Kubernetes for better scaling", mock_llm)
    brain.think("Raised Series A: $12M led by Sequoia, $180M valuation", mock_llm)

    state = brain.what_i_know()
    print(f"  Superseded: {state['superseded']}")
    print(f"  Causal links: {state['causal_links']}")

    # ── Recall ──
    print("\n--- Recall ---")
    for q in ["What backend?", "What database?", "Funding?", "Manager?"]:
        kb_answer = brain._kb.query(q)
        print(f"  {q:20s} -> {kb_answer or 'via episodes'}")

    # ── Why question ──
    print("\n--- Why Questions ---")
    enriched = brain.prepare("Why did we switch the backend?")
    for line in enriched["system"].split("\n"):
        if "->" in line or "→" in line:
            print(f"  {line.strip()}")

    # ── Knowledge gaps ──
    print("\n--- Knowledge Gaps ---")
    for gap in state["knowledge_gaps"][:3]:
        print(f"  [{gap['topic']}] Missing: {', '.join(gap['missing'])}")

    # ── Suggestions ──
    print("\n--- Proactive Suggestions ---")
    for s in state["suggestions"][:3]:
        print(f"  {s}")

    # ── Conflict resolution ──
    print("\n--- Implicit Conflict ---")
    brain.think("I drive a BMW 3 Series", mock_llm)
    brain.think("Just picked up my new Tesla Model 3", mock_llm)
    facts = brain._memory.get_facts()
    has_bmw = any("bmw" in t.lower() for t, _ in facts)
    has_tesla = any("tesla" in t.lower() for t, _ in facts)
    print(f"  BMW in facts: {has_bmw} (should be False)")
    print(f"  Tesla in facts: {has_tesla} (should be True)")

    brain.close()
    shutil.rmtree("./demo_memories", ignore_errors=True)
    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
