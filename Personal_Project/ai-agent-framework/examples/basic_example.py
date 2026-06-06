"""
EXAMPLE 1: Basic Agent Usage
Contoh paling sederhana untuk memulai
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.utils import setup_logging, print_result


def main():
    """
    Run basic agent example
    """
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    print("\n" + "="*60)
    print("BASIC AGENT EXAMPLE")
    print("="*60)
    print("\nInitializing agent with tools: search, calculate, summarize")
    
    # Create agent
    agent = Agent(
        model="gpt-4",
        memory_path="./memory",
        max_iterations=5,
        token_budget=2000,
        use_persistent_memory=True
    )
    
    # Task
    task = "Research AI agents and summarize key points"
    
    print(f"\nTask: {task}\n")
    
    # Run agent
    result = agent.run(task=task, max_retries=2)
    
    # Display result
    print_result(result, verbose=True)
    
    # Access session info
    session_info = agent.get_session_info()
    print("\nSession Info:")
    for key, value in session_info.items():
        print(f"  {key}: {value}")
    
    # Show execution history
    print("\nExecution History:")
    for i, execution in enumerate(agent.tool_executor.execution_history):
        print(f"\n  {i+1}. {execution.tool}")
        print(f"     Success: {execution.success}")
        print(f"     Duration: {execution.duration_seconds:.2f}s")
    
    return result


if __name__ == "__main__":
    result = main()
    
    # Return exit code based on success
    sys.exit(0 if result.success else 1)
