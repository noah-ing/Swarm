#!/usr/bin/env python3
"""Run a single task through the Supervisor with full cognitive display."""

import sys
from agents import Supervisor
from brain import get_brain
from evolution import get_evolution


def run_task(task: str, model: str = "sonnet"):
    """Run a task with full cognitive visibility."""
    print(f"\n{'='*60}")
    print(f"🐝 SWARM COGNITIVE EXECUTION")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    supervisor = Supervisor(working_dir=".", model=model)

    # Callbacks for visibility
    def on_thinking(t):
        print(f"\n💭 THINKING PHASE")
        print(f"   Understanding: {t.understanding}")
        print(f"   Strategy: {t.strategy}")
        print(f"   Confidence: {t.confidence:.0%}")
        print(f"   Complexity: {t.estimated_complexity}")
        if t.risks:
            print(f"   Risks: {[r.get('risk', '')[:40] for r in t.risks[:2]]}")
        if t.missing_info:
            print(f"   Missing info: {t.missing_info[:2]}")
        print()

    def on_strategy(s):
        print(f"🎯 {s}")

    def on_learning(l):
        print(f"📚 Learned: {l}")

    supervisor.on_thinking = on_thinking
    supervisor.on_strategy = on_strategy
    supervisor.on_learning = on_learning

    # Execute
    print("⚡ EXECUTING...")
    result = supervisor.run(task, allow_ask=False)

    # Results
    print(f"\n{'='*60}")
    if result.success:
        print(f"✅ SUCCESS in {result.duration_seconds:.1f}s (${result.cost_usd:.4f})")
        print(f"\n📝 Result:\n{result.message[:500]}")
    else:
        print(f"❌ FAILED: {result.message}")

    if result.learnings:
        print(f"\n💡 Learnings:")
        for l in result.learnings:
            print(f"   - {l}")

    # Show updated stats
    print(f"\n{'='*60}")
    print("📊 LEARNING STATS AFTER TASK")
    print(f"{'='*60}")

    brain = get_brain()
    evo = get_evolution()
    brain_stats = brain.get_stats()
    evo_stats = evo.get_stats()

    print(f"Brain reflections: {brain_stats['total_reflections']}")
    print(f"Skills learned: {brain_stats['skills_learned']}")
    print(f"Evolution mutations: {evo_stats['total_mutations']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_task.py 'your task here' [model]")
        sys.exit(1)

    task = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "sonnet"
    run_task(task, model)
