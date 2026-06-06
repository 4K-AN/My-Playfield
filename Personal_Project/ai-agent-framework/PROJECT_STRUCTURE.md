# Project Structure & Learning Map

Panduan navigasi untuk framework dan learning path yang recommended.

## 📁 Complete Project Structure

```
ai-agent-framework/
│
├── 📖 DOCUMENTATION
│   ├── README.md                          # Project overview
│   ├── AGENT_LOOP_EXPLAINED.md           # Deep dive into agent loop
│   ├── GETTING_STARTED.md                # Step-by-step setup guide
│   ├── PROJECT_STRUCTURE.md              # This file
│   ├── requirements.txt                  # Python dependencies
│   └── .env.example                      # Environment template
│
├── 💻 SOURCE CODE (src/)
│   ├── __init__.py                       # Package exports
│   ├── agent.py                          # Main Agent class
│   ├── memory.py                         # Memory system
│   ├── tools.py                          # Tool registry & execution
│   ├── parser.py                         # Output parsing
│   ├── retry.py                          # Retry logic
│   ├── models.py                         # Pydantic models
│   └── utils.py                          # Utility functions
│
├── 🎓 EXAMPLES (examples/)
│   ├── __init__.py
│   ├── basic_example.py                  # Simple usage
│   ├── research_agent.py                 # Multi-step with memory
│   ├── multi_tool_agent.py               # Advanced retry & error handling
│   └── task_automation.py                # Practical automation
│
├── 🧪 TESTS (tests/)
│   ├── __init__.py
│   ├── test_agent.py                     # Unit tests
│   └── (More test files for each component)
│
└── 📦 DATA
    └── memory/
        ├── persistent/                   # Persistent storage
        └── sessions/                     # Session data
```

## 🎯 Learning Path

### 1️⃣ **Foundation** (30 min)
- [ ] Read [README.md](README.md)
- [ ] Understand agent loop concept
- [ ] Read [AGENT_LOOP_EXPLAINED.md](AGENT_LOOP_EXPLAINED.md)

**Key Takeaway:** Agent = Loop that observes, plans, executes, reflects, decides

### 2️⃣ **Setup** (15 min)
- [ ] Follow [GETTING_STARTED.md](GETTING_STARTED.md) Steps 1-2
- [ ] Install dependencies
- [ ] Setup environment

**Key Takeaway:** Environment ready, dependencies installed

### 3️⃣ **Run Examples** (45 min)
- [ ] Run `basic_example.py`
- [ ] Run `research_agent.py`
- [ ] Run `multi_tool_agent.py`
- [ ] Run `task_automation.py`

**Key Takeaway:** Understand different use cases and capabilities

### 4️⃣ **Core Components** (90 min)

#### 4a. Agent Class (`src/agent.py`)
```python
# Key methods:
agent.run(task)          # Main execution
agent._observe()         # Phase 1: Gather context
agent._plan()            # Phase 2: Generate strategy
agent._execute()         # Phase 3: Run tools
agent._reflect()         # Phase 4: Learn from results
agent._decide()          # Phase 5: Continue or stop?
```

#### 4b. Memory System (`src/memory.py`)
```python
# Key methods:
memory.start_session()   # Begin new session
memory.add_message()     # Add to conversation
memory.add_fact()        # Store learned info
memory.get_facts()       # Retrieve stored facts
memory.save_session()    # Persist to disk
```

#### 4c. Tool System (`src/tools.py`)
```python
# Key classes:
ToolRegistry             # Manage available tools
ToolExecutor             # Execute tools with retry
Tool                     # Tool definition
ExecutionResult          # Tool execution result
```

#### 4d. Retry Logic (`src/retry.py`)
```python
# Key features:
retry_decorator          # Decorator for auto-retry
retry_with_backoff       # Direct function call
exponential_backoff      # Wait strategy
RetryConfig              # Configuration
```

#### 4e. Output Parsing (`src/parser.py`)
```python
# Key methods:
parser.parse_json()      # Extract JSON dari text
parser.parse_to_model()  # Validate to Pydantic model
extract_json_blocks()    # Find JSON dalam text
```

### 5️⃣ **Build Custom Agent** (60 min)
- [ ] Create custom tools
- [ ] Register tools dengan agent
- [ ] Configure memory
- [ ] Setup retry logic
- [ ] Run custom workflow

**Key Takeaway:** Can build agent untuk specific use case

### 6️⃣ **Advanced Topics** (120 min)
- [ ] Multi-tool orchestration
- [ ] Session continuation
- [ ] Error recovery strategies
- [ ] Memory optimization
- [ ] Performance tuning

**Key Takeaway:** Expert level understanding

## 🔍 Code Navigation Guide

### Starting a New Agent

```python
from src.agent import Agent

# Step 1: Create agent
agent = Agent(
    model="gpt-4",                      # See: src/agent.py
    memory_path="./memory",             # See: src/memory.py
    max_iterations=10,
    use_persistent_memory=True          # See: src/memory.py
)

# Step 2: Add tools (optional)
def my_function(param: str):            # See: examples/
    return {"result": "..."}

agent.add_tool(                         # See: src/tools.py
    name="my_tool",
    implementation=my_function,
    description="Description"
)

# Step 3: Run
result = agent.run(task="Your task")    # See: src/agent.py -> run()

# Step 4: Access results
print(result.output)                    # See: src/models.py -> AgentResult
print(result.memory_used)
```

### Understanding Agent Loop

```python
# In src/agent.py, main loop at run() method:

while iteration < max_iterations:
    # 1. OBSERVE
    context = agent._observe()          # Line ~XXX
    
    # 2. PLAN
    plan = agent._plan(context)         # Line ~XXX
    
    # 3. EXECUTE
    results = agent._execute(plan)      # Line ~XXX
    
    # 4. REFLECT
    reflection = agent._reflect(plan)   # Line ~XXX
    
    # 5. DECIDE
    decision = agent._decide()          # Line ~XXX
    
    # 6. LOOP or RETURN
    if decision != "LOOP":
        break
```

### Memory Persistence

```python
# In src/memory.py:

# Save session
memory.save_session()                   # Line ~XXX

# Load session
memory.start_session(
    session_id="same_id"                # Line ~XXX
)
# Automatically loads previous facts!

# Query facts
facts = memory.get_facts()              # Line ~XXX
```

### Tool Execution with Retry

```python
# In src/tools.py -> execute():

result = retry_with_backoff(            # Calls: src/retry.py
    tool_func,
    config=retry_config
)

# Retry strategy in src/retry.py:
# - Exponential backoff
# - Optional jitter
# - Max retries limit
```

## 📚 Component Dependency Map

```
agent.py
  ├── memory.py (persistent learning)
  ├── tools.py (tool execution)
  │   └── retry.py (resilience)
  ├── parser.py (output validation)
  │   └── models.py (schemas)
  ├── models.py (data structures)
  └── utils.py (helpers)
```

## 🎯 Use Case Guide

| Use Case | Start Here | Key Files |
|----------|-----------|-----------|
| **Simple agent** | `examples/basic_example.py` | `agent.py`, `tools.py` |
| **Multi-step research** | `examples/research_agent.py` | `agent.py`, `memory.py` |
| **Error recovery** | `examples/multi_tool_agent.py` | `retry.py`, `agent.py` |
| **Task automation** | `examples/task_automation.py` | Full stack |
| **Custom tools** | `examples/basic_example.py` | `tools.py` |
| **Memory persistence** | `examples/research_agent.py` | `memory.py` |
| **Batch processing** | `examples/task_automation.py` | `tools.py` |

## 📖 Documentation Map

| Document | Purpose | Audience |
|----------|---------|----------|
| README.md | Overview | Everyone |
| AGENT_LOOP_EXPLAINED.md | Deep technical | Developers |
| GETTING_STARTED.md | Setup & basics | New users |
| PROJECT_STRUCTURE.md | Navigation | Everyone |
| Docstrings in code | API reference | Developers |
| Examples/ | Working code | Learners |
| Tests/ | Usage patterns | Advanced users |

## 🔧 Configuration Guide

### Agent Configuration

```python
Agent(
    model="gpt-4",               # LLM model
    memory_path="./memory",      # Storage location
    max_iterations=10,           # Loop limit
    token_budget=4000,           # Token limit
    use_persistent_memory=True   # Persistence
)
```

### Memory Configuration

```python
Memory(
    memory_path="./memory",
    use_persistent=True          # Enable file storage
)
```

### Retry Configuration

```python
RetryConfig(
    max_retries=3,               # Attempts
    initial_wait=1.0,            # First wait (seconds)
    backoff_multiplier=2.0,      # Exponential growth
    max_wait=60.0,               # Max wait time
    jitter=True                  # Random variance
)
```

## 📊 Performance Tips

### Optimization

1. **Reduce iterations**: Set reasonable `max_iterations`
2. **Batch tools**: Use `execute_batch()` untuk parallel tasks
3. **Memory pruning**: Regularly clean old facts
4. **Token monitoring**: Track `total_tokens_used`

### Monitoring

```python
# Track metrics
print(result.iterations)          # How many loops
print(result.total_tokens)        # Token consumption
print(result.execution_time_seconds)  # Runtime
print(result.memory_used)         # Memory stats
```

## 🐛 Debugging Strategy

1. **Enable DEBUG logging**
   ```python
   setup_logging(level="DEBUG", log_file="debug.log")
   ```

2. **Check execution history**
   ```python
   agent.tool_executor.execution_history
   ```

3. **Inspect memory**
   ```python
   agent.memory.get_facts()
   agent.memory.get_conversation_history()
   ```

4. **Test tools individually**
   ```python
   executor = ToolExecutor(registry)
   result = executor.execute(tool_name, params)
   ```

## 🎓 Knowledge Checklist

- [ ] Understand agent loop (6 phases)
- [ ] Know memory persistence mechanism
- [ ] Know tool execution & retry logic
- [ ] Know output parsing strategies
- [ ] Can build custom tools
- [ ] Can create custom agents
- [ ] Can debug failures
- [ ] Know best practices
- [ ] Familiar dengan all examples
- [ ] Can run full test suite

## 📚 Next Learning Resources

1. **Advanced topics**
   - Multi-agent coordination
   - Hierarchical planning
   - Knowledge graphs
   - Semantic search

2. **Production topics**
   - Monitoring & logging
   - Error tracking
   - Performance optimization
   - Scalability

3. **Integration topics**
   - OpenAI API integration
   - Vector databases
   - Message queues
   - Microservices

---

**You are here:** Understanding framework → Building agents → Production deployment

**Start with:** [GETTING_STARTED.md](GETTING_STARTED.md)
