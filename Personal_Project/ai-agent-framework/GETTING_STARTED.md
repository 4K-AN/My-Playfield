# Getting Started Guide - AI Agent Framework

Panduan langkah-demi-langkah untuk memulai dengan Agent Framework.

## Prerequisites

- Python 3.10 atau lebih tinggi
- pip untuk package management
- Basic pemahaman tentang Python

## Step 1: Setup Lingkungan

### 1.1 Install Dependencies

```bash
# Navigate ke project folder
cd ai-agent-framework

# Install requirements
pip install -r requirements.txt
```

### 1.2 Setup Environment Variables

```bash
# Copy example file
cp .env.example .env

# Edit .env file dengan settings Anda
# IMPORTANT: Tambahkan OpenAI API key Anda
OPENAI_API_KEY=sk-your-api-key-here
```

## Step 2: Pahami Architecture

Sebelum coding, baca dan pahami:

1. **[README.md](README.md)** - Overview project
2. **[AGENT_LOOP_EXPLAINED.md](AGENT_LOOP_EXPLAINED.md)** - Detail tentang agent loop

Key concepts:
- **OBSERVE**: Kumpulkan context
- **PLAN**: Generate strategy
- **EXECUTE**: Jalankan tools
- **REFLECT**: Learn from results
- **DECIDE**: Continue atau stop
- **RETURN**: Package hasil

## Step 3: Run Basic Example

### 3.1 Jalankan contoh paling sederhana

```bash
cd examples
python basic_example.py
```

**Output:**
```
============================================================
BASIC AGENT EXAMPLE
============================================================

Initializing agent with tools: search, calculate, summarize

Task: Research AI agents and summarize key points

============================================================
AGENT EXECUTION COMPLETE
============================================================

Status: ✓ SUCCESS
Output: {...}
Metrics:
  Iterations: 3
  Tokens: 850
  Time: 5.23s
  Session: session_20240115_103015

============================================================
```

### 3.2 Explore execution

```python
# Di dalam basic_example.py, lihat section:
# - Session Info
# - Execution History
```

## Step 4: Explore Examples

Setiap example mendemonstrasikan fitur berbeda:

### Research Agent
Menunjukkan: Multi-step planning, memory persistence, session continuation

```bash
python research_agent.py continuation
python research_agent.py memory_reuse
```

### Multi-Tool Agent
Menunjukkan: Retry logic, error recovery, batch execution

```bash
python multi_tool_agent.py retry
python multi_tool_agent.py recovery
python multi_tool_agent.py workflow
```

### Task Automation
Menunjukkan: Practical workflow automation, integration

```bash
python task_automation.py triage
python task_automation.py processing
python task_automation.py custom
```

## Step 5: Pahami Key Components

### Memory System

```python
from src.memory import Memory

# Create memory
memory = Memory(memory_path="./memory", use_persistent=True)

# Start session
memory.start_session(task="My task", session_id="session_001")

# Add facts (learned information)
memory.add_fact(
    key="important_info",
    value="Information about topic",
    category="fact"
)

# Save untuk next session
memory.save_session()

# Load existing session
memory.start_session(task="Continuation", session_id="session_001")
facts = memory.get_facts()  # Load dari previous session!
```

### Tool System

```python
from src.tools import ToolRegistry, ToolExecutor

# Create registry
registry = ToolRegistry()

# Add tool
def my_tool(param1: str) -> dict:
    return {"result": f"Processed {param1}"}

registry.register(
    name="my_tool",
    description="My custom tool",
    implementation=my_tool,
    parameters=[...]
)

# Execute tool
executor = ToolExecutor(registry)
result = executor.execute("my_tool", {"param1": "value"})
print(result.output)
```

### Retry Logic

```python
from src.retry import retry_decorator, RetryConfig

# Option 1: Decorator
@retry_decorator(max_retries=3)
def unreliable_function():
    # Might fail, will retry automatically
    pass

# Option 2: Direct call
from src.retry import retry_with_backoff

result = retry_with_backoff(
    unreliable_function,
    config=RetryConfig(max_retries=3, initial_wait=1.0)
)
```

### Output Parsing

```python
from src.parser import OutputParser

parser = OutputParser(strict=False)

# Parse JSON dari text (dengan error recovery)
json_data = parser.parse_json("""
The answer is:
```json
{
    "status": "success",
    "data": "result"
}
```
""")

# Parse ke Pydantic model
from src.models import Plan
plan = parser.parse_to_model(json_text, Plan)
```

## Step 6: Build Custom Agent

### 6.1 Simple Custom Agent

```python
from src.agent import Agent
from src.tools import ToolRegistry, ToolParameter

# Create registry dengan custom tools
registry = ToolRegistry()

# Add your tools
def my_search(query: str) -> dict:
    # Your implementation
    return {"results": [...]}

registry.register(
    name="search",
    description="Search for information",
    implementation=my_search,
    parameters=[
        ToolParameter(name="query", type="string", required=True)
    ]
)

# Create agent
agent = Agent(
    model="gpt-4",
    tool_registry=registry,
    max_iterations=10
)

# Run
result = agent.run(task="Your task here")
print(result.output)
```

### 6.2 Advanced: With Memory & Retry

```python
from src.agent import Agent
from src.retry import RetryConfig

agent = Agent(
    model="gpt-4",
    memory_path="./my_memory",
    max_iterations=15,
    token_budget=4000,
    use_persistent_memory=True,
    retry_config=RetryConfig(
        max_retries=5,
        initial_wait=2.0,
        backoff_multiplier=2.0
    ),
    tool_registry=registry
)

# Run dengan session continuity
result = agent.run(
    task="Continue research",
    session_id="research_001",
    max_retries=3
)

# Access learned facts
facts = agent.memory.get_facts()
for fact in facts:
    print(f"{fact['key']}: {fact['value']}")
```

## Step 7: Run Tests

```bash
# Run all tests
cd tests
python test_agent.py

# Output:
# test_add_tool (__main__.TestAgent) ... ok
# test_agent_initialization (__main__.TestAgent) ... ok
# ... more tests ...
# Ran 18 tests in 2.34s - OK
```

## Step 8: Debugging Tips

### Enable Detailed Logging

```python
from src.utils import setup_logging

logger = setup_logging(level="DEBUG", log_file="agent.log")
```

### Monitor Session State

```python
# Access session info
session_info = agent.get_session_info()
print(session_info)

# Check execution history
for execution in agent.tool_executor.execution_history:
    print(f"{execution.tool}: {execution.success} ({execution.duration_seconds}s)")
```

### Inspect Memory

```python
# See all facts
print(agent.memory.get_facts())

# See conversation history
print(agent.memory.get_conversation_history())

# Get context summary
print(agent.memory.get_context_summary())
```

## Step 9: Common Patterns

### Pattern 1: Multi-Step Research

```python
task = """
Research topic and:
1. Find latest information
2. Analyze findings
3. Extract key insights
4. Create summary
"""

result = agent.run(task=task)
```

### Pattern 2: Error Recovery

```python
# Agent automatically handles:
# - Tool failures dengan retry
# - Temporary network errors
# - Token limits
# - Max iterations

# Setup aggressive retry
result = agent.run(task=task, max_retries=5)
```

### Pattern 3: Session Continuation

```python
# Session 1: Initial work
result1 = agent.run(task="Start research", session_id="research_001")

# Session 2: Continue (same session_id)
# - Loads previous facts
# - Continues from where it left off
result2 = agent.run(task="Deepen research", session_id="research_001")
```

## Step 10: Best Practices

### ✅ DO

1. **Use descriptive session IDs** untuk easy tracking
2. **Set max_iterations** sesuai kebutuhan
3. **Monitor token usage** untuk cost control
4. **Save results** ke persistent memory
5. **Log execution** untuk debugging
6. **Test tools** sebelum integrate ke agent
7. **Use retry logic** untuk unreliable operations

### ❌ DON'T

1. Jangan set max_iterations terlalu tinggi
2. Jangan forget to save session
3. Jangan hardcode credentials
4. Jangan ignore error logs
5. Jangan set token_budget terlalu rendah

## Troubleshooting

### Issue: "Tool not found"

```python
# Check available tools
tools = agent.tool_registry.list_tools()
print([t.name for t in tools])

# Ensure tool is registered
agent.add_tool(name="my_tool", ...)
```

### Issue: "Memory not persisting"

```python
# Make sure to save session
agent.memory.save_session()

# Check memory path exists
import os
assert os.path.exists(agent.memory.memory_path)
```

### Issue: "Agent looping infinitely"

```python
# Check max_iterations
agent.max_iterations = 10  # Set reasonable limit

# Check decision logic
# Look at AGENT_LOOP_EXPLAINED.md for decision tree
```

### Issue: "High token usage"

```python
# Reduce max_iterations
agent.max_iterations = 5

# Set token_budget
agent.token_budget = 2000

# Simplify tool complexity
```

## Next Steps

1. ✓ Complete all steps di guide ini
2. Read [AGENT_LOOP_EXPLAINED.md](AGENT_LOOP_EXPLAINED.md) in detail
3. Explore semua examples
4. Build custom agent untuk your use case
5. Implement in production

## Resources

- **Examples**: `/examples` folder
- **Tests**: `/tests` folder (good reference implementation)
- **Architecture**: [AGENT_LOOP_EXPLAINED.md](AGENT_LOOP_EXPLAINED.md)
- **API Reference**: Docstrings di `/src` files

## Support

Untuk issues atau questions:
1. Check troubleshooting section
2. Review relevant example
3. Check tests untuk usage patterns
4. Read detailed documentation

---

**Happy learning! Start dengan basic_example.py dan build up from there.**
