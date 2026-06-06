"""
EXAMPLE 3: Advanced Multi-Tool Agent
Menunjukkan: retry logic, error recovery, multi-tool coordination
"""

import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.tools import ToolRegistry, ToolParameter
from src.retry import TemporaryError
from src.utils import setup_logging, print_result


# Unreliable tools yang demonstrate retry logic
def unreliable_api_call(endpoint: str, retries_left: int = 2) -> dict:
    """
    Tool yang sometimes fail (simulate flaky API)
    Demonstrate retry logic dengan exponential backoff
    """
    
    # Randomly fail 60% of time (to show retries working)
    if random.random() < 0.6:
        error_types = [
            "TimeoutError: Connection timed out",
            "ConnectionError: Failed to connect",
            "RateLimitError: Too many requests"
        ]
        raise TemporaryError(random.choice(error_types))
    
    # Success
    return {
        "endpoint": endpoint,
        "status": 200,
        "data": f"Data from {endpoint}",
        "retries_needed": 2 - retries_left
    }


def data_processing(raw_data: str, operations: list = None) -> dict:
    """Process data dengan multiple operations"""
    
    if operations is None:
        operations = ["clean", "validate"]
    
    # Simulate processing
    result = {
        "raw_input": raw_data[:30],
        "operations_applied": operations,
        "output": f"Processed: {raw_data[:50]}...",
        "success": True
    }
    
    return result


def database_write(record_id: str, data: dict) -> dict:
    """Write data ke database"""
    
    # Simulate occasional write failures
    if random.random() < 0.3:
        raise TemporaryError("Database connection lost")
    
    return {
        "record_id": record_id,
        "status": "written",
        "bytes": len(str(data))
    }


def error_logger(error_message: str) -> dict:
    """Log errors untuk debugging"""
    
    return {
        "error": error_message,
        "logged": True,
        "timestamp": "2024-01-01T00:00:00Z"
    }


def validation_check(data: dict) -> dict:
    """Validate data"""
    
    validation_rules = [
        ("has_required_fields", "data" in str(data).lower()),
        ("format_valid", len(data) > 0),
        ("no_nulls", "none" not in str(data).lower())
    ]
    
    all_valid = all(result for _, result in validation_rules)
    
    return {
        "validated": all_valid,
        "checks": [
            {"rule": name, "passed": result}
            for name, result in validation_rules
        ]
    }


def create_advanced_agent() -> Agent:
    """Create agent dengan advanced tools"""
    
    registry = ToolRegistry()
    
    # Register unreliable tool (demonstrates retry)
    registry.register(
        name="api_call",
        description="Call external API (may timeout, demonstrates retry logic)",
        implementation=unreliable_api_call,
        parameters=[
            ToolParameter(
                name="endpoint",
                type="string",
                required=True,
                description="API endpoint"
            )
        ]
    )
    
    # Register processing tool
    registry.register(
        name="process_data",
        description="Process and clean data",
        implementation=data_processing,
        parameters=[
            ToolParameter(
                name="raw_data",
                type="string",
                required=True,
                description="Raw data to process"
            ),
            ToolParameter(
                name="operations",
                type="list",
                required=False,
                description="List of operations"
            )
        ]
    )
    
    # Register database tool (also unreliable)
    registry.register(
        name="save_to_db",
        description="Save data ke database",
        implementation=database_write,
        parameters=[
            ToolParameter(
                name="record_id",
                type="string",
                required=True,
                description="Record ID"
            ),
            ToolParameter(
                name="data",
                type="dict",
                required=True,
                description="Data to save"
            )
        ]
    )
    
    # Register error logging tool
    registry.register(
        name="log_error",
        description="Log error untuk debugging",
        implementation=error_logger,
        parameters=[
            ToolParameter(
                name="error_message",
                type="string",
                required=True,
                description="Error message to log"
            )
        ]
    )
    
    # Register validation tool
    registry.register(
        name="validate",
        description="Validate data",
        implementation=validation_check,
        parameters=[
            ToolParameter(
                name="data",
                type="dict",
                required=True,
                description="Data to validate"
            )
        ]
    )
    
    # Create agent dengan aggressive retry
    agent = Agent(
        model="gpt-4",
        memory_path="./memory/advanced",
        max_iterations=10,
        token_budget=4000,
        use_persistent_memory=True,
        tool_registry=registry
    )
    
    return agent


def demonstrate_retry_logic():
    """
    Demonstrate retry logic dengan flaky tools
    """
    
    print("\n" + "="*60)
    print("ADVANCED AGENT - RETRY LOGIC DEMO")
    print("="*60)
    print("\nNote: api_call tool may fail, triggering retry with exponential backoff")
    
    logger = setup_logging(level="DEBUG")  # Show detailed logs
    
    agent = create_advanced_agent()
    
    # Task yang akan use unreliable tool
    task = """
    Fetch data dari API endpoint, process it, validate it, dan save to database.
    Handle any failures gracefully dengan retry logic.
    """
    
    result = agent.run(
        task=task,
        session_id="advanced_retry_demo",
        max_retries=3  # Aggressive retry
    )
    
    print_result(result, verbose=True)
    
    # Show execution details
    print("\nExecution Details:")
    for i, execution in enumerate(agent.tool_executor.execution_history):
        print(f"\n  Step {i+1}: {execution.tool}")
        print(f"    Success: {execution.success}")
        print(f"    Duration: {execution.duration_seconds:.2f}s")
        if execution.error:
            print(f"    Error: {execution.error}")
        if execution.output:
            output_str = str(execution.output)
            print(f"    Output: {output_str[:100]}...")
    
    return result


def demonstrate_error_recovery():
    """
    Demonstrate error recovery dengan fallback planning
    """
    
    print("\n" + "="*60)
    print("ADVANCED AGENT - ERROR RECOVERY DEMO")
    print("="*60)
    print("\nAgent akan recovery dari errors menggunakan fallback strategies")
    
    logger = setup_logging(level="INFO")
    
    agent = create_advanced_agent()
    
    task = """
    Complex workflow:
    1. Call potentially failing API
    2. Process the retrieved data
    3. Validate all results
    4. Save to database
    5. Log any errors untuk analysis
    
    If any step fails, use fallback strategy dan retry.
    """
    
    result = agent.run(
        task=task,
        session_id="advanced_recovery_demo",
        max_retries=3
    )
    
    print_result(result, verbose=True)
    
    # Analyze recovery
    history = agent.tool_executor.execution_history
    failed_attempts = sum(1 for ex in history if not ex.success)
    successful_attempts = sum(1 for ex in history if ex.success)
    
    print(f"\nRecovery Analysis:")
    print(f"  Total tool calls: {len(history)}")
    print(f"  Successful: {successful_attempts}")
    print(f"  Failed: {failed_attempts}")
    print(f"  Success rate: {(successful_attempts/len(history)*100) if history else 0:.1f}%")
    
    # Show learned patterns
    print(f"\nMemory Learning:")
    facts = agent.memory.get_facts()
    print(f"  Facts stored: {len(facts)}")
    for fact in facts[:3]:
        print(f"    - {fact['key']}: {str(fact['value'])[:50]}...")
    
    return result


def demonstrate_multi_tool_workflow():
    """
    Demonstrate complex workflow dengan multiple tools
    """
    
    print("\n" + "="*60)
    print("ADVANCED AGENT - MULTI-TOOL WORKFLOW")
    print("="*60)
    
    logger = setup_logging(level="INFO")
    
    agent = create_advanced_agent()
    
    task = """
    Execute complete data pipeline:
    1. Fetch data dari multiple API endpoints
    2. Process dan clean setiap response
    3. Validate results
    4. Save processed data to database
    5. Log all steps
    
    Optimize untuk reliability dengan proper error handling.
    """
    
    result = agent.run(
        task=task,
        session_id="advanced_workflow_demo",
        max_retries=2
    )
    
    print_result(result, verbose=True)
    
    print("\nWorkflow Metrics:")
    history = agent.tool_executor.execution_history
    
    # Group by tool
    tool_stats = {}
    for execution in history:
        tool = execution.tool
        if tool not in tool_stats:
            tool_stats[tool] = {"count": 0, "success": 0, "total_time": 0}
        
        tool_stats[tool]["count"] += 1
        if execution.success:
            tool_stats[tool]["success"] += 1
        tool_stats[tool]["total_time"] += execution.duration_seconds
    
    print("\n  Tool Statistics:")
    for tool, stats in tool_stats.items():
        success_rate = (stats["success"] / stats["count"] * 100) if stats["count"] > 0 else 0
        print(f"    {tool}:")
        print(f"      Calls: {stats['count']}")
        print(f"      Success rate: {success_rate:.1f}%")
        print(f"      Total time: {stats['total_time']:.2f}s")
    
    return result


def main():
    """Main demo function"""
    
    demo_type = sys.argv[1] if len(sys.argv) > 1 else "retry"
    
    if demo_type == "retry":
        return demonstrate_retry_logic()
    elif demo_type == "recovery":
        return demonstrate_error_recovery()
    elif demo_type == "workflow":
        return demonstrate_multi_tool_workflow()
    else:
        print("Usage: python multi_tool_agent.py [retry|recovery|workflow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
