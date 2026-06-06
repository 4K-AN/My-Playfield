"""
Utility functions untuk Agent Framework
"""

import logging
import sys
from typing import Any, Dict
import json
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: str = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    
    Returns:
        Configured logger
    """
    
    logger = logging.getLogger("agent_framework")
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def print_result(result: Any, verbose: bool = False):
    """
    Pretty print agent result
    
    Args:
        result: AgentResult object
        verbose: Include detailed information
    """
    
    print("\n" + "="*60)
    print("AGENT EXECUTION COMPLETE")
    print("="*60)
    
    if hasattr(result, 'success'):
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        print(f"\nStatus: {status}")
    
    if hasattr(result, 'error') and result.error:
        print(f"Error: {result.error}")
    
    if hasattr(result, 'output') and result.output:
        print(f"\nOutput:")
        if isinstance(result.output, dict):
            print(json.dumps(result.output, indent=2))
        else:
            print(result.output)
    
    if hasattr(result, 'iterations'):
        print(f"\nMetrics:")
        print(f"  Iterations: {result.iterations}")
    
    if hasattr(result, 'total_tokens'):
        print(f"  Tokens: {result.total_tokens}")
    
    if hasattr(result, 'execution_time_seconds'):
        print(f"  Time: {result.execution_time_seconds:.2f}s")
    
    if hasattr(result, 'session_id'):
        print(f"  Session: {result.session_id}")
    
    if verbose and hasattr(result, 'memory_used'):
        print(f"\nMemory Usage:")
        for key, value in result.memory_used.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)


def format_context_for_llm(context: Dict[str, Any]) -> str:
    """
    Format context dictionary menjadi readable string untuk LLM
    """
    
    lines = []
    
    for key, value in context.items():
        if key == "memory_summary":
            lines.append(f"\n{key.upper()}:")
            lines.append(str(value))
        elif key == "conversation_history":
            lines.append(f"\n{key.upper()}:")
            for msg in value:
                lines.append(f"  {msg.get('role', 'unknown')}: {msg.get('content', '')[:60]}...")
        elif key == "facts":
            lines.append(f"\n{key.upper()}:")
            for fact in value[:5]:  # Top 5
                lines.append(f"  - {fact.get('key')}: {str(fact.get('value'))[:50]}...")
        elif isinstance(value, list):
            lines.append(f"\n{key.upper()}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"\n{key.upper()}: {value}")
    
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """
    Rough estimate token count
    Approximation: ~1.3 chars per token untuk English
    """
    return max(1, int(len(text) / 4))


def merge_results(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple tool execution results
    """
    
    merged = {
        "all_successful": True,
        "tool_results": {},
        "errors": []
    }
    
    for result in results:
        tool_name = result.get("tool", "unknown")
        
        if result.get("success"):
            merged["tool_results"][tool_name] = result.get("output")
        else:
            merged["all_successful"] = False
            merged["errors"].append({
                "tool": tool_name,
                "error": result.get("error")
            })
    
    return merged


def sanitize_output(output: Any) -> str:
    """
    Sanitize output untuk logging/display
    Hindari exposing sensitive info
    """
    
    if isinstance(output, dict):
        sanitized = {}
        for key, value in output.items():
            if any(keyword in key.lower() for keyword in ['key', 'token', 'secret', 'password']):
                sanitized[key] = "***"
            else:
                sanitized[key] = sanitize_output(value)
        return json.dumps(sanitized)
    
    elif isinstance(output, list):
        return json.dumps([sanitize_output(item) for item in output[:5]])
    
    elif isinstance(output, str):
        return output[:200] + ("..." if len(output) > 200 else "")
    
    else:
        return str(output)


class ProgressTracker:
    """Track progress of agent execution"""
    
    def __init__(self, total_steps: int = 0):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.step_times = []
    
    def update(self, step: int):
        """Update progress"""
        self.current_step = step
        self.step_times.append(
            (datetime.now() - self.start_time).total_seconds()
        )
    
    def get_progress(self) -> float:
        """Get progress percentage"""
        if self.total_steps == 0:
            return 0
        return (self.current_step / self.total_steps) * 100
    
    def get_eta(self) -> str:
        """Get estimated time to completion"""
        if len(self.step_times) < 2:
            return "Unknown"
        
        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        remaining_time = avg_step_time * remaining_steps
        
        minutes = int(remaining_time / 60)
        seconds = int(remaining_time % 60)
        
        return f"{minutes}m {seconds}s"
    
    def get_summary(self) -> str:
        """Get progress summary"""
        return f"Progress: {self.get_progress():.1f}% | ETA: {self.get_eta()}"


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text dengan ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def compare_sessions(
    session1: Dict[str, Any],
    session2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two sessions untuk learning
    """
    
    comparison = {
        "same_task": session1.get("task") == session2.get("task"),
        "iteration_diff": session2.get("iterations", 0) - session1.get("iterations", 0),
        "success_improvement": (
            session2.get("success", False) and
            not session1.get("success", False)
        ),
        "metrics": {
            "session1": {
                "iterations": session1.get("iterations"),
                "status": session1.get("status")
            },
            "session2": {
                "iterations": session2.get("iterations"),
                "status": session2.get("status")
            }
        }
    }
    
    return comparison
