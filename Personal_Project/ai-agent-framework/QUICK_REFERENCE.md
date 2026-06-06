# AI Agent Framework - Quick Reference

**Complete Multi-Step AI Agent Implementation with Tool Use, Persistent Memory, Structured Output Parsing, and Failure Recovery**

## 🚀 Quick Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env

# Run Examples
python examples/basic_example.py
python examples/research_agent.py continuation
python examples/multi_tool_agent.py retry
python examples/task_automation.py triage

# Run Tests
python tests/test_agent.py
```

## 📊 Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AGENT LOOP (6 Phases)                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐      ┌──────────────┐                 │
│  │   OBSERVE   │─────>│     PLAN     │                 │
│  └─────────────┘      └──────────────┘                 │
│        ↑                      │                         │
│        │                      ▼                         │
│  ┌─────────────┐      ┌──────────────┐                 │
│  │   REFLECT   │<─────│   EXECUTE    │                 │
│  └─────────────┘      └──────────────┘                 │
│        ↑                      ▲                         │
│        │                      │                         │
│        └──────────────────────┘                         │
│              ▼                                          │
│        ┌──────────────┐                                 │
│        │    DECIDE    │ ──> Loop/Stop/Complete         │
│        └──────────────┘                                 │
│              │                                          │
│              ▼                                          │
│        ┌──────────────┐                                 │
│        │    RETURN    │ ──> AgentResult                │
│        └──────────────┘                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Core Components

### 1. Agent (`src/agent.py`)
- Orchestrates the 6-phase loop
- Manages iteration and state
- Coordinates all components

### 2. Memory (`src/memory.py`)
- In-memory conversation history
- Persistent fact storage
- Session management
- Cross-session learning

### 3. Tools (`src/tools.py`)
- Tool registry
- Tool executor
- Built-in tools (search, calculate, summarize)
- Extensible for custom tools

### 4. Retry Logic (`src/retry.py`)
- Exponential backoff
- Jitter support
- Automatic recovery
- Configurable strategies

### 5. Parser (`src/parser.py`)
- JSON extraction from text
- Error recovery
- Pydantic validation
- Schema enforcement

## 💻 Minimal Example

```python
from src.agent import Agent

# Create agent
agent = Agent(
    model="gpt-4",
    memory_path="./memory",
    max_iterations=10
)

# Run task
result = agent.run(task="Research AI agents and summarize")

# Access results
print(result.output)
print(result.iterations)
print(result.total_tokens)
```

## 📁 File Overview

| File | Purpose | Key Classes |
|------|---------|------------|
| `agent.py` | Main agent loop | `Agent` |
| `memory.py` | Persistent storage | `Memory`, `FileStore` |
| `tools.py` | Tool execution | `ToolRegistry`, `ToolExecutor` |
| `retry.py` | Resilience | `RetryConfig`, `retry_with_backoff` |
| `parser.py` | Output parsing | `OutputParser`, `JsonExtractor` |
| `models.py` | Data schemas | `Tool`, `Plan`, `AgentResult` |

## 🎯 Use Cases

| Use Case | Example |
|----------|---------|
| **Simple Tasks** | `basic_example.py` |
| **Research Workflow** | `research_agent.py` |
| **Error Recovery** | `multi_tool_agent.py` |
| **Task Automation** | `task_automation.py` |

## 📖 Documentation Structure

```
README.md                          ← Start here
    ↓
AGENT_LOOP_EXPLAINED.md           ← Understand architecture
    ↓
GETTING_STARTED.md                ← Setup & first run
    ↓
examples/basic_example.py         ← See it in action
    ↓
examples/research_agent.py        ← Explore features
    ↓
PROJECT_STRUCTURE.md              ← Navigate codebase
    ↓
IMPLEMENTATION_CHECKLIST.md       ← Build your own
```

## 🔑 Key Features

✅ **Multi-step Agent Loop**
- OBSERVE → PLAN → EXECUTE → REFLECT → DECIDE → RETURN

✅ **Tool Use & Integration**
- Register custom tools
- Execute with parameters
- Chain multiple tools

✅ **Persistent Memory**
- In-session memory
- Cross-session facts
- Automatic learning

✅ **Structured Output**
- JSON parsing from text
- Error recovery
- Schema validation

✅ **Failure Recovery**
- Exponential backoff
- Configurable retries
- Graceful degradation

✅ **Full Tracing**
- Execution history
- Memory tracking
- Token counting

## 🎓 Learning Path

```
1. Read README.md (5 min)
   ↓
2. Read AGENT_LOOP_EXPLAINED.md (20 min)
   ↓
3. Follow GETTING_STARTED.md (30 min)
   ↓
4. Run basic_example.py (5 min)
   ↓
5. Run research_agent.py (10 min)
   ↓
6. Explore source code (45 min)
   ↓
7. Build custom agent (60 min)
   ↓
Total: ~2.5 hours to mastery
```

## 📊 Agent Loop Phases Explained

### Phase 1: OBSERVE (Gather Context)
```python
context = {
    "current_task": task,
    "conversation_history": [...],
    "known_facts": [...],
    "available_tools": [...],
    "iteration": current,
}
```

### Phase 2: PLAN (Generate Strategy)
```python
plan = {
    "reasoning": "why this approach",
    "next_steps": [
        {"tool": "search", "params": {...}},
        {"tool": "analyze", "params": {...}}
    ],
    "expected_outcome": "..."
}
```

### Phase 3: EXECUTE (Run Tools)
```python
results = [
    tool_executor.execute("search", params),
    tool_executor.execute("analyze", previous_output)
]
```

### Phase 4: REFLECT (Learn)
```python
memory.add_fact("key", "value", category="success")
memory.save_session()
```

### Phase 5: DECIDE (Continue?)
```python
if success or timeout or max_iterations:
    return COMPLETE/STOP
else:
    return LOOP
```

### Phase 6: RETURN (Package)
```python
return AgentResult(
    output=...,
    iterations=...,
    tokens=...,
    memory_used=...
)
```

## 🔄 Memory Persistence

```
Session 1: Learn facts
  ├── Add facts to memory
  └── Save session

Session 2: Reuse facts (same session_id)
  ├── Load previous facts
  └── Use in planning
     └── Better decisions!

Session 3: Continue learning
  ├── Add new facts
  └── Combine with old facts
```

## ⚡ Retry Strategy

```
Attempt 1: Fail immediately
  └── Wait: 1 second

Attempt 2: Still fails
  └── Wait: 2 seconds (exponential)

Attempt 3: Still fails
  └── Wait: 4 seconds (exponential)

Attempt 4: Success!
  └── Total retries: 3
```

## 📈 Metrics & Monitoring

```python
result = agent.run(task)

# Execution metrics
print(f"Iterations: {result.iterations}")
print(f"Tokens: {result.total_tokens}")
print(f"Time: {result.execution_time_seconds}s")

# Memory metrics
print(f"Conversation messages: {result.memory_used['conversation_history']}")
print(f"Facts stored: {result.memory_used['facts']}")
print(f"Session: {result.session_id}")
```

## 🛠️ Extending the Framework

### Add Custom Tool
```python
def my_tool(param: str) -> dict:
    return {"result": f"Processed {param}"}

agent.add_tool(
    name="my_tool",
    implementation=my_tool,
    description="My custom tool"
)
```

### Custom Retry Strategy
```python
config = RetryConfig(
    max_retries=5,
    initial_wait=2.0,
    backoff_multiplier=3.0
)
agent = Agent(retry_config=config)
```

### Custom Parser
```python
parser = OutputParser(strict=True)  # Strict validation
result = parser.parse_to_model(text, ModelClass)
```

## 🧪 Testing

```bash
# Run all tests
python tests/test_agent.py

# Or individually
python -m pytest tests/test_agent.py -v
python -m pytest tests/test_memory.py -v
python -m pytest tests/test_tools.py -v
```

## 📝 Best Practices

✅ Always set `max_iterations` limit
✅ Set reasonable `token_budget`
✅ Use meaningful `session_id` for tracking
✅ Enable logging for debugging
✅ Save sessions for persistence
✅ Test tools before integrating
✅ Monitor token usage
✅ Handle errors gracefully

❌ Don't set unlimited iterations
❌ Don't ignore error logs
❌ Don't hardcode API keys
❌ Don't forget to save sessions
❌ Don't use tools without testing
❌ Don't ignore token budget

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Tool not found | Check registry: `agent.tool_registry.list_tools()` |
| Memory not persisting | Enable: `use_persistent_memory=True` |
| High token usage | Reduce `max_iterations` or simplify tasks |
| Infinite loop | Check `max_iterations` and decision logic |
| Retry not working | Check `RetryConfig` and error types |

## 📚 Further Reading

- **Agent Loop**: AGENT_LOOP_EXPLAINED.md
- **Setup**: GETTING_STARTED.md
- **Structure**: PROJECT_STRUCTURE.md
- **Implementation**: IMPLEMENTATION_CHECKLIST.md
- **Examples**: /examples folder
- **Tests**: /tests folder

## 💡 Tips & Tricks

1. **Debug with logging**
   ```python
   setup_logging(level="DEBUG", log_file="debug.log")
   ```

2. **Monitor execution**
   ```python
   for execution in agent.tool_executor.execution_history:
       print(f"{execution.tool}: {execution.duration_seconds}s")
   ```

3. **Inspect memory**
   ```python
   facts = agent.memory.get_facts()
   history = agent.memory.get_conversation_history()
   ```

4. **Test tools independently**
   ```python
   executor = ToolExecutor(registry)
   result = executor.execute(tool_name, params)
   ```

## 🎯 Next Steps

1. ✅ Read README.md
2. ✅ Read AGENT_LOOP_EXPLAINED.md
3. ✅ Follow GETTING_STARTED.md
4. ✅ Run examples
5. ✅ Build custom agent
6. ✅ Deploy to production

---

**Happy building! Start with `examples/basic_example.py` and go from there.**

Questions? Check documentation or review examples.

Last Updated: 2024
Version: 1.0.0
