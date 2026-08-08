#!/usr/bin/env python3
"""
Interactive CLI interface for Gemma local agent in SenaAIgent.
Run directly from terminal: python cli.py
"""

import sys
import argparse
from agents import GemmaAgent, ModelAgent, OrchestratorAgent, TaskPriority


def interactive_chat(agent: GemmaAgent):
    """Run an interactive chat REPL in the terminal."""
    print("=" * 60)
    print("      GEMMA LOCAL LLM AGENT - INTERACTIVE TERMINAL CLI      ")
    print("=" * 60)
    status = agent.get_status()
    print(f"Model: {status['model']} | Host: {status['ollama_host']} | Status: {status['status']}")
    print(f"Acceleration: {status['hardware_acceleration']}")
    print("Commands: '/exit' to quit, '/telemetry' for sample risk analysis, '/optimize' for meta-orchestrator optimization.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n\033[1;34mYou > \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/exit", "quit", "exit"]:
                print("Exiting Gemma CLI. Goodbye!")
                break
            elif user_input.lower() == "/telemetry":
                print("\033[1;32mRunning Gemma Telemetry Risk Analysis...\033[0m")
                result = agent.analyze_telemetry({"ph": 6.2, "turbidity": 4.5, "temperature": 24.0, "dissolved_oxygen": 5.5})
                print(f"\nResult:\n{result}")
            elif user_input.lower() == "/optimize":
                print("\033[1;32mRunning Gemma Autonomous Queue Self-Optimization...\033[0m")
                result = agent.self_optimize_workflow(
                    queue_status={"pending": 3, "processing": 2},
                    load_metrics={"load_score": 78.5, "queue_length": 5}
                )
                print(f"\nOptimization Plan:\n{result}")
            else:
                print("\033[1;35mGemma > \033[0m", end="", flush=True)
                response = agent.generate(user_input)
                print(response)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break


def main():
    parser = argparse.ArgumentParser(description="Gemma Local Agent CLI")
    parser.add_argument("--model", type=str, default="gemma2:9b", help="Gemma model name")
    parser.add_argument("--prompt", type=str, help="Single-shot prompt execution")
    args = parser.parse_args()

    agent = GemmaAgent(model_name=args.model)

    if args.prompt:
        response = agent.generate(args.prompt)
        print(f"Gemma: {response}")
    else:
        interactive_chat(agent)


if __name__ == "__main__":
    main()
