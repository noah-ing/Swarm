#!/usr/bin/env python3
"""
Swarm Training Loop - Continuous learning through task execution.

Modes:
1. benchmark - Run against coding challenges
2. self-improve - Let Swarm improve its own code
3. interactive - REPL for continuous tasks
4. autonomous - Self-directed exploration and improvement
"""

import sys
import time
import random
import argparse
from pathlib import Path

from agents import Supervisor
from brain import get_brain
from evolution import get_evolution
from memory import get_memory_store


# Benchmark tasks for training
BENCHMARK_TASKS = [
    # Simple (should use direct strategy)
    ("Create a Python function that reverses a string", "low"),
    ("Write a function to check if a number is prime", "low"),
    ("Create a function that finds the maximum in a list", "low"),
    ("Write a Python function to calculate factorial", "low"),
    ("Create a function that counts vowels in a string", "low"),

    # Medium (might decompose)
    ("Create a Python class for a basic linked list with insert and delete", "medium"),
    ("Write a function that finds all duplicates in a list and returns their indices", "medium"),
    ("Create a decorator that caches function results", "medium"),
    ("Write a function that merges two sorted lists efficiently", "medium"),
    ("Create a context manager for timing code execution", "medium"),

    # Complex (should decompose)
    ("Create a simple REST API client class with GET, POST, PUT, DELETE methods and error handling", "high"),
    ("Write a recursive function to flatten arbitrarily nested lists and add tests", "high"),
    ("Create a thread-safe singleton pattern implementation with examples", "high"),
]

# Self-improvement prompts
SELF_IMPROVE_PROMPTS = [
    "Review {file} and suggest 3 specific improvements to make it more efficient",
    "Find potential bugs or edge cases in {file} and fix them",
    "Add better error handling to {file}",
    "Optimize the most complex function in {file}",
    "Add type hints to any untyped functions in {file}",
]

SWARM_FILES = [
    "agents/base.py",
    "agents/grunt.py",
    "agents/orchestrator.py",
    "agents/qa.py",
    "agents/thinker.py",
    "agents/supervisor.py",
    "brain.py",
    "evolution.py",
    "memory.py",
    "codebase.py",
]


def print_stats():
    """Print current learning statistics."""
    brain = get_brain()
    evolution = get_evolution()
    memory = get_memory_store()

    brain_stats = brain.get_stats()
    evo_stats = evolution.get_stats()
    mem_stats = memory.get_stats()

    print("\n" + "="*60)
    print("📊 SWARM LEARNING STATS")
    print("="*60)
    print(f"🧠 Brain:")
    print(f"   Reflections: {brain_stats['total_reflections']} ({brain_stats['successful_reflections']} successful)")
    print(f"   Skills learned: {brain_stats['skills_learned']}")
    print(f"   Failure patterns tracked: {brain_stats['failure_patterns_tracked']}")
    print(f"   Avg confidence: {brain_stats['avg_confidence']:.1%}")

    print(f"\n🧬 Evolution:")
    print(f"   Total mutations: {evo_stats['total_mutations']}")
    print(f"   Retirements: {evo_stats['total_retirements']}")
    for agent_type, stats in evo_stats.get('agents', {}).items():
        print(f"   {agent_type}: {stats['active_variants']} active variants, {stats['avg_success_rate']:.1%} success")

    print(f"\n💾 Memory:")
    print(f"   Solutions stored: {mem_stats['total_memories']}")
    print(f"   Successful solutions: {mem_stats['successful_memories']}")
    print(f"   Models tracked: {', '.join(mem_stats.get('models_used', [])) or 'none yet'}")
    print("="*60 + "\n")


def run_benchmark(rounds: int = 1, model: str = "sonnet"):
    """Run benchmark tasks to train the system."""
    print(f"🏋️ Starting benchmark training ({rounds} rounds, {len(BENCHMARK_TASKS)} tasks each)")
    print(f"   Model: {model}")
    print(f"   Total tasks: {rounds * len(BENCHMARK_TASKS)}\n")

    supervisor = Supervisor(working_dir=str(Path.cwd()), model=model)

    # Callbacks for visibility
    supervisor.on_thinking = lambda t: print(f"   💭 Strategy: {t.strategy} (confidence: {t.confidence:.0%})")
    supervisor.on_strategy = lambda s: print(f"   🎯 {s}")
    supervisor.on_learning = lambda l: print(f"   📚 Learned: {l}")

    results = {"success": 0, "fail": 0, "total_cost": 0.0, "total_time": 0.0}

    for round_num in range(rounds):
        print(f"\n{'='*60}")
        print(f"📍 ROUND {round_num + 1}/{rounds}")
        print("="*60)

        # Shuffle tasks for variety
        tasks = BENCHMARK_TASKS.copy()
        random.shuffle(tasks)

        for i, (task, complexity) in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {complexity.upper()}: {task[:50]}...")

            try:
                result = supervisor.run(
                    task=task,
                    context=f"This is a {complexity} complexity training task. Create the code in a file called training_output.py",
                    skip_qa=True,  # Faster training
                    allow_ask=False,  # Don't pause for questions
                )

                if result.success:
                    print(f"   ✅ SUCCESS in {result.duration_seconds:.1f}s (${result.cost_usd:.4f})")
                    results["success"] += 1
                else:
                    print(f"   ❌ FAILED: {result.message[:50]}...")
                    results["fail"] += 1

                results["total_cost"] += result.cost_usd
                results["total_time"] += result.duration_seconds

            except Exception as e:
                print(f"   💥 ERROR: {e}")
                results["fail"] += 1

            # Brief pause between tasks
            time.sleep(1)

    # Final stats
    print("\n" + "="*60)
    print("🏁 BENCHMARK COMPLETE")
    print("="*60)
    print(f"Success rate: {results['success']}/{results['success']+results['fail']} ({results['success']/(results['success']+results['fail'])*100:.1f}%)")
    print(f"Total cost: ${results['total_cost']:.4f}")
    print(f"Total time: {results['total_time']:.1f}s")

    print_stats()

    # Cleanup
    Path("training_output.py").unlink(missing_ok=True)


def run_self_improve(iterations: int = 3, model: str = "sonnet"):
    """Let Swarm improve its own code."""
    print(f"🔄 Starting self-improvement loop ({iterations} iterations)")
    print(f"   Target files: {len(SWARM_FILES)}")
    print(f"   Model: {model}\n")

    supervisor = Supervisor(working_dir=str(Path.cwd()), model=model)

    supervisor.on_thinking = lambda t: print(f"   💭 {t.understanding[:60]}...")
    supervisor.on_learning = lambda l: print(f"   📚 {l}")

    for iteration in range(iterations):
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {iteration + 1}/{iterations}")
        print("="*60)

        # Pick random file and improvement type
        target_file = random.choice(SWARM_FILES)
        prompt_template = random.choice(SELF_IMPROVE_PROMPTS)
        task = prompt_template.format(file=target_file)

        print(f"\n📝 Task: {task}")

        try:
            result = supervisor.run(
                task=task,
                context="You are improving the Swarm codebase. Be careful not to break existing functionality. Make minimal, targeted changes.",
                allow_ask=False,
            )

            if result.success:
                print(f"   ✅ Improvement suggested/applied")
                if result.learnings:
                    for learning in result.learnings[:2]:
                        print(f"   💡 {learning}")
            else:
                print(f"   ⚠️ No improvement made: {result.message[:50]}...")

        except Exception as e:
            print(f"   💥 Error: {e}")

        time.sleep(2)

    print_stats()


def run_interactive(model: str = "sonnet"):
    """Interactive REPL for continuous learning."""
    print("🎮 SWARM INTERACTIVE MODE")
    print("="*60)
    print("Commands:")
    print("  /stats  - Show learning statistics")
    print("  /reset  - Reset to fresh state")
    print("  /model  - Change model (haiku/sonnet/opus)")
    print("  /quit   - Exit")
    print("="*60 + "\n")

    supervisor = Supervisor(working_dir=str(Path.cwd()), model=model)

    # Verbose callbacks
    supervisor.on_thinking = lambda t: print(f"\n💭 Thinking: {t.strategy} strategy, {t.confidence:.0%} confidence")
    supervisor.on_strategy = lambda s: print(f"🎯 {s}")
    supervisor.on_learning = lambda l: print(f"📚 Learned: {l}")

    task_count = 0

    while True:
        try:
            user_input = input("\n🐝 swarm> ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("👋 Goodbye!")
                break
            elif user_input == "/stats":
                print_stats()
                continue
            elif user_input == "/reset":
                print("⚠️ This would reset all learning. Not implemented yet.")
                continue
            elif user_input.startswith("/model"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] in ("haiku", "sonnet", "opus"):
                    supervisor = Supervisor(working_dir=str(Path.cwd()), model=parts[1])
                    print(f"✅ Switched to {parts[1]}")
                else:
                    print("Usage: /model haiku|sonnet|opus")
                continue

            # Execute task
            task_count += 1
            print(f"\n[Task #{task_count}] Processing...")

            result = supervisor.run(
                task=user_input,
                allow_ask=True,
                stream=False,
            )

            if result.success:
                print(f"\n✅ Success ({result.duration_seconds:.1f}s, ${result.cost_usd:.4f})")
                print(f"\n{result.message}")
            else:
                print(f"\n❌ Failed: {result.message}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n💥 Error: {e}")


def run_autonomous(duration_minutes: int = 10, model: str = "sonnet"):
    """Autonomous self-directed improvement."""
    print(f"🤖 AUTONOMOUS MODE - {duration_minutes} minutes")
    print("="*60)
    print("Swarm will self-direct its own learning and improvement.")
    print("Press Ctrl+C to stop.\n")

    supervisor = Supervisor(working_dir=str(Path.cwd()), model=model)

    supervisor.on_thinking = lambda t: print(f"   💭 {t.understanding[:50]}...")
    supervisor.on_learning = lambda l: print(f"   📚 {l}")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    iteration = 0

    # Meta-prompt for self-direction
    meta_prompt = """You are in autonomous self-improvement mode.

Based on the current state of the Swarm codebase and your learning history, decide what to do next.
Options:
1. Fix a bug or issue you've noticed
2. Improve performance of a slow component
3. Add a missing feature that would be useful
4. Write a test for untested code
5. Improve documentation
6. Refactor unclear code

Choose ONE specific, actionable task and execute it. Be precise."""

    try:
        while time.time() < end_time:
            iteration += 1
            elapsed = (time.time() - start_time) / 60
            remaining = duration_minutes - elapsed

            print(f"\n{'='*60}")
            print(f"🤖 AUTONOMOUS ITERATION {iteration} ({remaining:.1f} min remaining)")
            print("="*60)

            result = supervisor.run(
                task=meta_prompt,
                context=f"Iteration {iteration}. Focus on high-impact improvements.",
                allow_ask=False,
            )

            if result.success:
                print(f"   ✅ Completed: {result.message[:60]}...")
            else:
                print(f"   ⚠️ Issue: {result.message[:60]}...")

            # Adaptive delay based on result
            time.sleep(5 if result.success else 10)

    except KeyboardInterrupt:
        print("\n\n⏹️ Autonomous mode stopped by user")

    print_stats()


def main():
    parser = argparse.ArgumentParser(description="Swarm Training Loop")
    parser.add_argument("mode", choices=["benchmark", "self-improve", "interactive", "autonomous", "stats"],
                       help="Training mode")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds (benchmark mode)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations (self-improve mode)")
    parser.add_argument("--duration", type=int, default=10, help="Duration in minutes (autonomous mode)")
    parser.add_argument("--model", default="sonnet", choices=["haiku", "sonnet", "opus"],
                       help="Model to use")

    args = parser.parse_args()

    if args.mode == "stats":
        print_stats()
    elif args.mode == "benchmark":
        run_benchmark(rounds=args.rounds, model=args.model)
    elif args.mode == "self-improve":
        run_self_improve(iterations=args.iterations, model=args.model)
    elif args.mode == "interactive":
        run_interactive(model=args.model)
    elif args.mode == "autonomous":
        run_autonomous(duration_minutes=args.duration, model=args.model)


if __name__ == "__main__":
    main()
