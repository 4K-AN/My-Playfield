"""
Main Agent class yang mengimplementasikan agent loop
OBSERVE → PLAN → EXECUTE → REFLECT → DECIDE → RETURN
"""

import logging
from typing import Any, Dict, Optional, List
import os
from datetime import datetime

from .models import AgentResult, Plan, ExecutionStep, Reflection
from .memory import Memory
from .tools import ToolRegistry, ToolExecutor, create_default_registry
from .parser import OutputParser, StructuredOutputSchema
from .retry import RetryConfig

logger = logging.getLogger(__name__)


class Agent:
    """
    Multi-step AI Agent dengan tool use, memory, dan retry logic
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        memory_path: str = "./memory",
        max_iterations: int = 10,
        token_budget: int = 4000,
        use_persistent_memory: bool = True,
        tool_registry: Optional[ToolRegistry] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize Agent
        
        Args:
            model: LLM model name (gpt-4, gpt-3.5-turbo, etc)
            memory_path: Path untuk persistent memory storage
            max_iterations: Maximum agent loop iterations
            token_budget: Maximum tokens untuk execution
            use_persistent_memory: Enable persistent memory across sessions
            tool_registry: Custom tool registry (or use default)
            retry_config: Custom retry configuration
        """
        
        self.model = model
        self.max_iterations = max_iterations
        self.token_budget = token_budget
        
        # Initialize components
        self.memory = Memory(
            memory_path=memory_path,
            use_persistent=use_persistent_memory
        )
        
        # Tool system
        self.tool_registry = tool_registry or create_default_registry()
        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            retry_config=retry_config or RetryConfig()
        )
        
        # Output parsing
        self.output_parser = OutputParser(strict=False)
        
        # Execution state
        self.current_iteration = 0
        self.total_tokens_used = 0
        self.execution_start_time = None
    
    def add_tool(
        self,
        name: str,
        implementation: Any,
        description: str,
        parameters: Optional[List[Any]] = None
    ):
        """Add custom tool ke agent"""
        
        self.tool_registry.register(
            name=name,
            description=description,
            implementation=implementation,
            parameters=parameters or []
        )
    
    def run(
        self,
        task: str,
        session_id: Optional[str] = None,
        max_retries: int = 3
    ) -> AgentResult:
        """
        Run agent untuk complete task
        
        Main entry point yang orchestrate agent loop
        
        Args:
            task: Task description
            session_id: Session ID (for continuation)
            max_retries: Max retries per tool call
        
        Returns:
            AgentResult dengan output dan metadata
        """
        
        # Initialize
        self.execution_start_time = datetime.now()
        self.current_iteration = 0
        self.total_tokens_used = 0
        
        # Setup session
        self.memory.start_session(
            task=task,
            session_id=session_id,
            max_iterations=self.max_iterations
        )
        
        logger.info(f"Agent started: {task}")
        logger.info(f"Session: {self.memory.session_id}")
        
        # Main agent loop
        final_reflection = None
        
        while self.current_iteration < self.max_iterations:
            try:
                # PHASE 1: OBSERVE
                context = self._observe()
                
                # PHASE 2: PLAN
                plan = self._plan(context)
                
                if not plan:
                    logger.error("Failed to generate plan")
                    break
                
                # PHASE 3: EXECUTE
                execution_results = self._execute(plan.next_steps, max_retries)
                
                # PHASE 4: REFLECT
                final_reflection = self._reflect(plan, execution_results)
                
                # PHASE 5: DECIDE
                decision = self._decide(final_reflection, context)
                
                # PHASE 6: RETURN or LOOP?
                if decision["action"] != "LOOP":
                    logger.info(f"Agent loop ended: {decision['reason']}")
                    break
                
                self.current_iteration += 1
                
            except Exception as e:
                logger.error(f"Agent error at iteration {self.current_iteration}: {e}")
                final_reflection = Reflection(
                    success=False,
                    error=str(e),
                    recoverable=False
                )
                break
        
        # Save session
        self.memory.end_session(
            status="completed" if final_reflection and final_reflection.success else "failed"
        )
        
        # Build and return result
        return self._build_result(final_reflection)
    
    def _observe(self) -> Dict[str, Any]:
        """
        PHASE 1: OBSERVE - Gather context
        """
        
        logger.debug(f"OBSERVE: Iteration {self.current_iteration}")
        
        context = {
            "task": self.memory.current_session.task if self.memory.current_session else "",
            "iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "tokens_used": self.total_tokens_used,
            "token_budget": self.token_budget,
            "memory_summary": self.memory.get_context_summary(),
            "conversation_history": self.memory.get_conversation_history(last_n=5),
            "facts": self.memory.get_facts(),
            "available_tools": [t.name for t in self.tool_registry.list_tools()],
        }
        
        return context
    
    def _plan(self, context: Dict[str, Any]) -> Optional[Plan]:
        """
        PHASE 2: PLAN - Generate action plan
        """
        
        logger.debug("PLAN: Generating plan")
        
        # Build prompt untuk LLM
        prompt = self._build_plan_prompt(context)
        
        # Call LLM (simulated untuk demo)
        plan_json = self._call_llm(prompt)
        
        # Parse response
        try:
            plan = self.output_parser.parse_to_model(plan_json, Plan)
            
            self.memory.add_message(
                role="assistant",
                content=f"Plan: {plan.reasoning}"
            )
            
            logger.info(f"Generated plan with {len(plan.next_steps)} steps")
            
            return plan
        
        except Exception as e:
            logger.error(f"Failed to parse plan: {e}")
            return None
    
    def _execute(
        self,
        steps: List[ExecutionStep],
        max_retries: int
    ) -> Dict[str, Any]:
        """
        PHASE 3: EXECUTE - Run tools
        """
        
        logger.debug("EXECUTE: Running steps")
        
        results = {}
        step_outputs = {}
        
        for step in steps:
            try:
                # Resolve parameters (might reference previous outputs)
                resolved_params = self._resolve_parameters(
                    step.params,
                    step_outputs
                )
                
                logger.info(f"Executing step {step.step}: {step.tool}")
                
                # Execute tool
                execution_result = self.tool_executor.execute(
                    tool_name=step.tool,
                    params=resolved_params,
                    max_retries=max_retries
                )
                
                results[f"step_{step.step}"] = {
                    "success": execution_result.success,
                    "output": execution_result.output,
                    "error": execution_result.error,
                    "duration": execution_result.duration_seconds
                }
                
                # Store untuk next steps
                step_outputs[f"step_{step.step}_output"] = execution_result.output
                
                # Update token count (rough estimate)
                self.total_tokens_used += 100
                
                # Log to memory
                self.memory.add_message(
                    role="tool",
                    content=f"Step {step.step} result: {str(execution_result.output)[:100]}...",
                    tool_call_id=f"step_{step.step}"
                )
                
                # Check if critical
                if step.critical and not execution_result.success:
                    logger.error(f"Critical step {step.step} failed!")
                    break
            
            except Exception as e:
                logger.error(f"Error executing step {step.step}: {e}")
                results[f"step_{step.step}"] = {
                    "success": False,
                    "error": str(e)
                }
                
                if step.critical:
                    break
        
        return results
    
    def _reflect(
        self,
        plan: Plan,
        execution_results: Dict[str, Any]
    ) -> Reflection:
        """
        PHASE 4: REFLECT - Learn and analyze
        """
        
        logger.debug("REFLECT: Analyzing results")
        
        # Check success
        success = all(
            result.get("success", False)
            for result in execution_results.values()
        )
        
        # Extract insights
        insights = [plan.reasoning]
        
        if success:
            # Save successful pattern
            self.memory.add_fact(
                key=f"successful_pattern_{self.current_iteration}",
                value={
                    "plan": plan.reasoning,
                    "steps": len(plan.next_steps),
                    "iteration": self.current_iteration
                },
                category="success"
            )
        else:
            # Save failure pattern
            failed_steps = [
                k for k, v in execution_results.items()
                if not v.get("success", False)
            ]
            
            self.memory.add_failed_attempt(
                task=self.memory.current_session.task if self.memory.current_session else "",
                attempt={"plan": plan.dict(), "iteration": self.current_iteration},
                error=str(execution_results),
                reason="Execution failed"
            )
            
            insights.append(f"Failed steps: {failed_steps}")
        
        # Build reflection
        reflection = Reflection(
            success=success,
            parsed_output=execution_results,
            insights=insights,
            recoverable=True,
            suggested_next_action="continue" if not success else "complete"
        )
        
        return reflection
    
    def _decide(
        self,
        reflection: Reflection,
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        PHASE 5: DECIDE - Continue or stop?
        """
        
        logger.debug("DECIDE: Making decision")
        
        # Decision logic
        if reflection.success:
            return {
                "action": "COMPLETE",
                "reason": "Goal achieved"
            }
        
        if context["iteration"] >= context["max_iterations"] - 1:
            return {
                "action": "STOP",
                "reason": "Max iterations reached"
            }
        
        if context["tokens_used"] >= context["token_budget"]:
            return {
                "action": "STOP",
                "reason": "Token budget exceeded"
            }
        
        if reflection.error and not reflection.recoverable:
            return {
                "action": "FAIL",
                "reason": f"Unrecoverable error: {reflection.error}"
            }
        
        return {
            "action": "LOOP",
            "reason": "More steps needed"
        }
    
    def _build_result(self, reflection: Optional[Reflection]) -> AgentResult:
        """
        PHASE 6: BUILD RESULT - Package everything
        """
        
        execution_time = (
            (datetime.now() - self.execution_start_time).total_seconds()
            if self.execution_start_time
            else 0
        )
        
        return AgentResult(
            output=reflection.parsed_output if reflection else None,
            success=reflection.success if reflection else False,
            error=reflection.error if reflection else "Unknown error",
            iterations=self.current_iteration,
            total_tokens=self.total_tokens_used,
            execution_time_seconds=execution_time,
            memory_used={
                "conversation_history": len(self.memory.conversation_history),
                "facts": len(self.memory.facts),
                "execution_history": len(self.tool_executor.execution_history)
            },
            session_id=self.memory.session_id or "unknown",
            reasoning_trace=[],
            execution_history=self.tool_executor.execution_history,
            saved_facts=self.memory.facts
        )
    
    # Helper methods
    
    def _build_plan_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt untuk plan generation"""
        
        prompt = f"""
You are an intelligent AI agent. Your task is to plan the next steps.

Task: {context['task']}
Iteration: {context['iteration']}/{context['max_iterations']}

Available Tools:
{chr(10).join(f"  - {t.name}: {t.description}" for t in self.tool_registry.list_tools())}

Context:
{context['memory_summary']}

Recent History:
{chr(10).join(f"  {msg['role']}: {msg['content'][:50]}..." for msg in context['conversation_history'])}

Generate a plan in JSON format:
{{
    "reasoning": "why this approach",
    "next_steps": [
        {{
            "step": 1,
            "tool": "tool_name",
            "params": {{}},
            "description": "what this step does",
            "critical": false
        }}
    ],
    "expected_outcome": "what should happen",
    "fallback_plan": "if this fails, do this",
    "confidence": 0.8
}}
"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM untuk planning
        Untuk demo, return simulated response
        """
        
        # Simulated LLM response
        # Dalam real application, ini akan call OpenAI API
        
        return """
{
    "reasoning": "First, gather information about the task, then process it",
    "next_steps": [
        {
            "step": 1,
            "tool": "search",
            "params": {"query": "information about task"},
            "description": "Gather relevant information",
            "critical": true
        }
    ],
    "expected_outcome": "Successful task completion",
    "fallback_plan": "Use alternative approach",
    "confidence": 0.85
}
"""
    
    def _resolve_parameters(
        self,
        params: Dict[str, Any],
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve parameter values yang mungkin reference ke outputs sebelumnya
        
        Example:
            Input: {"text": "$.step_1_output"}
            Returns: {"text": <actual output dari step 1>}
        """
        
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference ke previous output
                ref_key = value[1:]  # Remove $
                resolved[key] = previous_outputs.get(ref_key, value)
            else:
                resolved[key] = value
        
        return resolved
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get info tentang current session"""
        
        if not self.memory.current_session:
            return {}
        
        session = self.memory.current_session
        
        return {
            "session_id": session.session_id,
            "task": session.task,
            "iterations": session.iterations,
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "facts_count": len(session.facts),
            "history_length": len(session.conversation_history)
        }
