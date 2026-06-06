"""
Tool System untuk agent
Registry, execution, dan error handling untuk tools
"""

from typing import Any, Callable, Dict, List, Optional
import logging
from .models import Tool, ToolParameter, ExecutionResult
from .retry import retry_with_backoff, RetryConfig, TemporaryError
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry untuk manage semua available tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.implementations: Dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        description: str,
        implementation: Callable,
        parameters: List[ToolParameter] = None
    ):
        """Register sebuah tool"""
        
        if parameters is None:
            parameters = []
        
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters
        )
        
        self.tools[name] = tool
        self.implementations[name] = implementation
        
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool definition"""
        return self.tools.get(name)
    
    def get_implementation(self, name: str) -> Optional[Callable]:
        """Get tool implementation function"""
        return self.implementations.get(name)
    
    def list_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tools formatted untuk OpenAI function calling"""
        return [tool.to_tool_definition() for tool in self.list_tools()]


class ToolExecutor:
    """Execute tools dengan error handling dan retry logic"""
    
    def __init__(
        self,
        registry: ToolRegistry,
        retry_config: Optional[RetryConfig] = None
    ):
        self.registry = registry
        self.retry_config = retry_config or RetryConfig()
        self.execution_history: List[ExecutionResult] = []
    
    def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        max_retries: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute sebuah tool
        
        Args:
            tool_name: Nama tool
            params: Parameters untuk tool
            max_retries: Override default retry config
        
        Returns:
            ExecutionResult dengan output atau error
        """
        
        start_time = time.time()
        
        # Validate tool exists
        implementation = self.registry.get_implementation(tool_name)
        if not implementation:
            return ExecutionResult(
                step_number=0,
                tool=tool_name,
                params=params,
                output=None,
                error=f"Tool not found: {tool_name}",
                success=False,
                duration_seconds=0
            )
        
        # Prepare retry config
        config = self.retry_config
        if max_retries:
            config = RetryConfig(max_retries=max_retries)
        
        try:
            # Execute dengan retry
            output = retry_with_backoff(
                self._execute_tool,
                tool_name=tool_name,
                implementation=implementation,
                params=params,
                config=config,
                on_retry=self._on_retry
            )
            
            duration = time.time() - start_time
            
            result = ExecutionResult(
                step_number=0,
                tool=tool_name,
                params=params,
                output=output,
                success=True,
                duration_seconds=duration
            )
            
            logger.info(
                f"Tool {tool_name} executed successfully in {duration:.2f}s"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            result = ExecutionResult(
                step_number=0,
                tool=tool_name,
                params=params,
                error=str(e),
                success=False,
                duration_seconds=duration
            )
            
            logger.error(
                f"Tool {tool_name} failed after {duration:.2f}s: {e}"
            )
        
        # Save to history
        self.execution_history.append(result)
        
        return result
    
    def _execute_tool(
        self,
        tool_name: str,
        implementation: Callable,
        params: Dict[str, Any]
    ) -> Any:
        """Internal method untuk execute tool"""
        
        try:
            logger.debug(f"Executing {tool_name} with params: {params}")
            result = implementation(**params)
            return result
        
        except TimeoutError as e:
            raise TemporaryError(f"Tool timeout: {e}") from e
        except ConnectionError as e:
            raise TemporaryError(f"Connection error: {e}") from e
        except Exception as e:
            raise
    
    def _on_retry(self, attempt: int, error: Exception):
        """Callback when retry happens"""
        logger.warning(f"Retry attempt {attempt}: {error}")
    
    def execute_batch(
        self,
        tools_params: List[tuple[str, Dict[str, Any]]]
    ) -> List[ExecutionResult]:
        """
        Execute multiple tools
        
        Args:
            tools_params: List of (tool_name, params) tuples
        
        Returns:
            List of ExecutionResult
        """
        
        results = []
        
        for tool_name, params in tools_params:
            result = self.execute(tool_name, params)
            results.append(result)
            
            # Stop on critical failure
            if not result.success:
                logger.warning(f"Batch execution stopped at {tool_name}")
                break
        
        return results
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get execution history"""
        return self.execution_history.copy()
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()


# Built-in tools untuk common tasks

def builtin_search(query: str) -> Dict[str, Any]:
    """
    Simulated search tool
    Dalam aplikasi real, ini akan call ke real API
    """
    return {
        "query": query,
        "results": [
            {"title": f"Result about {query}", "snippet": "..."}
        ],
        "count": 1
    }


def builtin_calculate(expression: str) -> Dict[str, Any]:
    """
    Simulated calculator tool
    """
    try:
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }


def builtin_summarize(text: str) -> Dict[str, Any]:
    """
    Simulated summarization tool
    """
    words = text.split()
    summary = " ".join(words[:min(20, len(words))]) + "..."
    
    return {
        "original_length": len(text),
        "summary": summary,
        "summary_length": len(summary)
    }


def create_default_registry() -> ToolRegistry:
    """Create registry dengan built-in tools"""
    
    registry = ToolRegistry()
    
    # Register built-in tools
    registry.register(
        name="search",
        description="Search untuk informasi",
        implementation=builtin_search,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                required=True,
                description="Search query"
            )
        ]
    )
    
    registry.register(
        name="calculate",
        description="Calculate mathematical expression",
        implementation=builtin_calculate,
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                required=True,
                description="Math expression (e.g., '2 + 2')"
            )
        ]
    )
    
    registry.register(
        name="summarize",
        description="Summarize text",
        implementation=builtin_summarize,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                required=True,
                description="Text to summarize"
            )
        ]
    )
    
    return registry


# Example usage
"""
from tools import ToolRegistry, ToolExecutor, create_default_registry

# Option 1: Use default registry
registry = create_default_registry()

# Option 2: Create custom registry
registry = ToolRegistry()
registry.register(
    name="my_tool",
    description="My custom tool",
    implementation=my_function,
    parameters=[
        ToolParameter(name="param1", type="string", required=True)
    ]
)

# Execute
executor = ToolExecutor(registry)
result = executor.execute("search", {"query": "AI"})
print(result.output)

# Batch execution
results = executor.execute_batch([
    ("search", {"query": "AI"}),
    ("calculate", {"expression": "2 + 2"}),
])
"""
