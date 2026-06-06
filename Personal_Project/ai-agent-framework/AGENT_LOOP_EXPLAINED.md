# Agent Loop Explained - Deep Dive

## 🔄 Complete Agent Loop Breakdown

### Phase 1: OBSERVE (Konteks & Memory)

```python
def observe():
    """
    Kumpulkan semua informasi yang relevan untuk membuat keputusan
    """
    context = {
        "current_task": task,
        "history": memory.get_conversation_history(),
        "facts": memory.get_facts(),
        "previous_attempts": memory.get_failed_attempts(),
        "iteration": current_iteration,
        "tokens_used": token_counter.total(),
    }
```

**Apa yang terjadi:**
- Baca task/goal dari input
- Load conversation history dari memory
- Ambil facts dan context yang telah disimpan
- Lihat error/failure sebelumnya untuk dihindari
- Check remaining iterations dan tokens

**Mengapa penting:**
- LLM butuh context untuk reasoning
- Memory enables learning dari mistakes
- Prevents infinite loops

---

### Phase 2: PLAN (LLM Generate Steps)

```python
def plan(context):
    """
    LLM membaca context dan generate next action(s)
    """
    prompt = f"""
    Current Task: {context['current_task']}
    
    History:
    {format_history(context['history'])}
    
    Previous Failed Attempts:
    {context['previous_attempts']}
    
    Available Tools: {list_tools()}
    
    Think step by step. Return JSON with:
    {{
        "reasoning": "why this approach",
        "next_steps": [
            {{
                "step": 1,
                "tool": "tool_name",
                "params": {{...}},
                "description": "why this step"
            }}
        ],
        "expected_outcome": "what should happen",
        "fallback_plan": "if this fails, do this instead"
    }}
    """
    
    response = llm.generate(prompt)
    return response
```

**Output yang diharapkan:**
```json
{
    "reasoning": "Need to search for information first, then process it",
    "next_steps": [
        {
            "step": 1,
            "tool": "web_search",
            "params": {"query": "latest AI trends 2024"},
            "description": "Gather current information"
        },
        {
            "step": 2,
            "tool": "text_analyzer",
            "params": {"text": "will_come_from_step_1"},
            "description": "Analyze search results"
        }
    ],
    "expected_outcome": "Comprehensive summary of AI trends",
    "fallback_plan": "If search fails, use local knowledge base"
}
```

**Mengapa penting:**
- LLM bisa think ahead daripada reactive
- Fallback plan mengurangi need untuk retry
- Reasoning membuat debugging lebih mudah

---

### Phase 3: EXECUTE (Jalankan Tools)

```python
def execute_plan(steps):
    """
    Jalankan setiap step dengan error handling
    """
    results = {}
    
    for step in steps:
        tool_name = step["tool"]
        params = resolve_params(step["params"], results)
        
        try:
            # Call tool dengan retry logic
            result = execute_with_retry(
                tool=tool_name,
                params=params,
                max_retries=3,
                backoff_strategy="exponential"
            )
            
            results[f"step_{step['step']}"] = result
            
            # Save to memory immediately
            memory.log_step(
                step_number=step["step"],
                tool=tool_name,
                params=params,
                result=result,
                success=True
            )
            
        except ToolError as e:
            results[f"step_{step['step']}_error"] = str(e)
            memory.log_step(
                step_number=step["step"],
                tool=tool_name,
                error=str(e),
                success=False
            )
            
            if step.get("critical"):
                return {"error": str(e), "partial_results": results}
    
    return results
```

**Karakteristik penting:**
- Sequential atau parallel execution (bergantung dependency)
- Parameter resolution dari previous steps
- Immediate memory logging
- Graceful error handling
- Partial success support

**Contoh Execution:**
```
Step 1: web_search(query="AI trends")
  ✓ Found 15 articles
  → saved to step_1_result
  
Step 2: text_analyzer(text=<step_1_result[0]>)
  ✓ Extracted 5 key insights
  → saved to step_2_result
  
Step 3: summarize(texts=<step_2_result>)
  ✓ Generated summary
  → final result ready
```

---

### Phase 4: REFLECT (Update Memory & Parse)

```python
def reflect(execution_results, original_plan):
    """
    Analisis hasil execution dan update memory
    """
    # Parse structured output
    parsed_output = parse_json_output(
        raw_output=execution_results,
        schema=OutputSchema
    )
    
    # Evaluate success
    success = evaluate_results(
        results=parsed_output,
        expected_outcome=original_plan["expected_outcome"]
    )
    
    if not success:
        # Learn from failure
        memory.add_failure_case(
            task=current_task,
            plan=original_plan,
            results=execution_results,
            reason=evaluate_reason_for_failure()
        )
    else:
        # Celebrate success
        memory.add_successful_case(
            task=current_task,
            solution=original_plan,
            results=parsed_output
        )
    
    # Update facts based on new information
    memory.update_facts(extracted_facts(parsed_output))
    
    return {
        "success": success,
        "parsed_output": parsed_output,
        "insights": extract_insights(execution_results)
    }
```

**Apa yang disimpan:**
- ✓ Successful patterns untuk reuse
- ✗ Failed patterns untuk avoid
- 📊 Facts dan insights untuk reference
- 📈 Metrics (tokens, time, attempts)

---

### Phase 5: DECIDE (Loop atau End?)

```python
def decide_next_action(reflection, context):
    """
    Tentukan apakah perlu loop lagi atau stop
    """
    
    # Condition 1: Goal achieved?
    if reflection["success"] and is_goal_complete():
        return {"action": "COMPLETE", "reason": "Goal achieved"}
    
    # Condition 2: Max iterations reached?
    if context["iteration"] >= max_iterations:
        return {"action": "STOP", "reason": "Max iterations reached"}
    
    # Condition 3: Token limit exceeded?
    if context["tokens_used"] > token_budget:
        return {"action": "STOP", "reason": "Token budget exceeded"}
    
    # Condition 4: Unrecoverable error?
    if reflection.get("error") and not reflection.get("recoverable"):
        return {"action": "FAIL", "reason": "Unrecoverable error"}
    
    # Otherwise, continue
    return {"action": "LOOP", "reason": "More steps needed"}
```

**Decision Tree:**
```
                    ┌─ Goal Complete? → COMPLETE
                    │
    Decision Point ─┼─ Max Iterations? → STOP
                    │
                    ├─ Token Limit? → STOP
                    │
                    ├─ Unrecoverable Error? → FAIL
                    │
                    └─ Otherwise → LOOP (back to PLAN phase)
```

---

### Phase 6: RETURN (Final Output)

```python
def return_result(final_reflection, context):
    """
    Format dan return hasil akhir
    """
    return AgentResult(
        # What the agent accomplished
        output=final_reflection["parsed_output"],
        
        # Execution metrics
        iterations=context["iteration"],
        total_tokens=context["tokens_used"],
        execution_time=time.time() - context["start_time"],
        
        # Memory usage
        memory_used={
            "conversation_history": len(memory.conversation_history),
            "facts_stored": len(memory.facts),
            "sessions": len(memory.sessions),
        },
        
        # Reasoning trail
        reasoning_trace=[
            step["reasoning"] for step in all_steps
        ],
        
        # Success status
        success=final_reflection["success"],
        error=final_reflection.get("error"),
        
        # For next session
        session_id=memory.session_id,
        saved_facts=memory.get_all_facts(),
    )
```

---

## 🔁 Loop Example: Real Scenario

**Task:** "Research and summarize the latest developments in quantum computing"

### Iteration 1
```
OBSERVE: No prior attempts, start fresh
↓
PLAN: 
  - Tool: web_search("quantum computing latest 2024")
  - Tool: text_analyzer on top 5 results
↓
EXECUTE: 
  - web_search returns 15 articles
  - text_analyzer extracts key points
↓
REFLECT: 
  - Good start, but need more depth
  - Save findings to memory
↓
DECIDE: Goal not complete, continue
↓
LOOP → Iteration 2
```

### Iteration 2
```
OBSERVE: 
  - Previous results in memory
  - We have 15 articles analyzed
  - Need: deeper technical analysis
↓
PLAN:
  - Tool: deep_technical_analyzer on complex articles
  - Tool: fact_checker on claims
↓
EXECUTE:
  - Technical analyzer identifies key breakthroughs
  - Fact checker validates claims
↓
REFLECT:
  - Good progress, have technical details
  - Facts verified and saved
↓
DECIDE: Goal almost complete, do final synthesis
↓
LOOP → Iteration 3
```

### Iteration 3
```
OBSERVE:
  - Comprehensive data from iterations 1-2
  - All facts verified
↓
PLAN:
  - Tool: summarizer(all_findings)
  - Tool: format_report(summary)
↓
EXECUTE:
  - Generate comprehensive summary
  - Format into readable report
↓
REFLECT:
  - Goal complete!
  - All information synthesized
  - Facts saved for future reference
↓
DECIDE: Goal achieved
↓
RETURN: Final report with metadata
```

---

## 🧠 Memory Persistence Across Sessions

### Session 1: Initial Research
```
Task: "Learn about AI agents"
Agent gathers info and saves:
  - facts: ["Agents use tool calling", "Memory improves performance"]
  - session_id: "sess_001"
  - completion: partial
```

### Session 2: Continuation
```
Task: "Continue learning about AI agents, focus on implementations"
Agent:
  - Loads facts from Session 1
  - Uses facts as context
  - Avoids re-researching known information
  - Builds on previous work
  - Adds new facts to memory
```

### Session 3: Later Reference
```
Task: "Build an AI agent for X"
Agent:
  - Loads ALL facts from Sessions 1-2
  - Reference patterns that worked
  - Avoid patterns that failed
  - Dramatically faster learning
```

---

## ⚡ Error Recovery & Retry Logic

### Exponential Backoff Strategy

```python
attempt = 1
wait_time = initial_wait  # e.g., 1 second

while attempt <= max_retries:
    try:
        result = execute_tool(params)
        return result
    except TemporaryError as e:
        if attempt >= max_retries:
            raise
        
        wait_time = initial_wait * (2 ** (attempt - 1))
        # Plus random jitter: 0-25%
        wait_time *= (1 + random(0, 0.25))
        
        time.sleep(wait_time)
        attempt += 1
```

### Retry Timeline Example
```
Attempt 1: Fails immediately
Wait: 1.0-1.25 seconds

Attempt 2: Still fails
Wait: 2.0-2.50 seconds

Attempt 3: Still fails
Wait: 4.0-5.00 seconds

Attempt 4: Success! ✓
Total time: ~8 seconds, but recovered
```

---

## 📊 Summary

| Phase | Purpose | Output |
|-------|---------|--------|
| OBSERVE | Gather context | Enriched context object |
| PLAN | Generate strategy | Structured action plan |
| EXECUTE | Do the work | Execution results |
| REFLECT | Learn & save | Updated memory |
| DECIDE | Continue? | Loop/Stop/Complete decision |
| RETURN | Wrap up | Final AgentResult |

**Key Insight:** Setiap phase meng-improve kualitas untuk phase berikutnya, creating virtuous cycle of learning!

---

Mari implementasi! Lihat `src/agent.py` untuk kode utama.
