# Multi-Step AI Agent Framework

Belajar membuat AI agent yang sophisticated dengan kemampuan multi-step execution, tool integration, persistent memory, structured output parsing, dan robust error handling.

## 🎯 Konsep Utama

### Agent Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT LOOP                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. [OBSERVE] Baca konteks dan memory                        │
│         ↓                                                    │
│  2. [PLAN] LLM generate steps dengan tool calls             │
│         ↓                                                    │
│  3. [EXECUTE] Jalankan tools, tangkap output                │
│         ↓                                                    │
│  4. [REFLECT] Update memory, parse output                   │
│         ↓                                                    │
│  5. [DECIDE] Selesai atau loop? (max iterations)            │
│         ↓                                                    │
│  6. [RETURN] Output terstruktur atau error handling         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔑 Key Components

### 1. **Core Agent** (`agent.py`)
- Orchestrates the agent loop
- Manages state and iteration
- Implements retry logic

### 2. **Memory System** (`memory.py`)
- Session memory (dalam-memory)
- Persistent memory (file-based)
- Conversation history
- Task context tracking

### 3. **Tool System** (`tools.py`)
- Tool registry
- Tool execution with error handling
- Structured parameter validation

### 4. **Output Parser** (`parser.py`)
- JSON parsing dari LLM output
- Validation dan error recovery
- Schema enforcement

### 5. **Retry Logic** (`retry.py`)
- Exponential backoff
- Specific error handling
- Max attempts management

## 📁 Project Structure

```
ai-agent-framework/
├── README.md (this file)
├── AGENT_LOOP_EXPLAINED.md (detailed explanation)
├── requirements.txt
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── agent.py (main agent loop)
│   ├── memory.py (memory system)
│   ├── tools.py (tool registry & execution)
│   ├── parser.py (output parsing)
│   ├── retry.py (retry logic)
│   ├── models.py (data models)
│   └── utils.py (utilities)
│
├── memory/
│   ├── persistent/ (stored memories)
│   └── sessions/ (session data)
│
├── examples/
│   ├── basic_example.py (simple agent)
│   ├── research_agent.py (research workflow)
│   ├── task_automation.py (task executor)
│   └── multi_tool_agent.py (advanced example)
│
└── tests/
    ├── test_agent.py
    ├── test_memory.py
    ├── test_tools.py
    └── test_parser.py
```

## 🚀 Quick Start

```python
from src.agent import Agent
from src.memory import PersistentMemory

# Initialize
agent = Agent(
    model="gpt-4",
    memory=PersistentMemory("./memory"),
    max_iterations=10
)

# Add tools
agent.add_tool("search", search_function)
agent.add_tool("calculate", calculate_function)

# Run agent
result = agent.run(
    task="Research AI agents and summarize findings",
    max_retries=3
)

print(result.output)
print(result.memory_used)
```

## 📚 Learning Path

1. **Baca** [AGENT_LOOP_EXPLAINED.md](AGENT_LOOP_EXPLAINED.md) - Pahami architecture
2. **Lihat** `examples/basic_example.py` - Simple case
3. **Implementasi** `src/agent.py` - Core logic
4. **Explore** `examples/research_agent.py` - Complex case
5. **Test** dengan `tests/` - Verify implementation

## 🔧 Technologies

- **Python 3.10+**
- **OpenAI API** (gpt-4, gpt-3.5-turbo)
- **Pydantic** (validation)
- **SQLite** (persistent memory)
- **Requests** (HTTP tools)

## 🎓 Key Learnings

✅ How to design agent loops
✅ Tool calling and execution
✅ Memory persistence across sessions
✅ JSON parsing dari LLM
✅ Retry logic dan error recovery
✅ Structured thinking dan planning
✅ State management

## 📝 Notes

- Semua examples bisa dijalankan standalone
- Memory system support multi-session tracking
- Retry logic dengan exponential backoff
- Full logging untuk debugging

---

Mulai dengan membaca [AGENT_LOOP_EXPLAINED.md](AGENT_LOOP_EXPLAINED.md)!
