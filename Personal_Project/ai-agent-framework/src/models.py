"""
Data models untuk Agent Framework
Menggunakan Pydantic untuk validation dan serialization
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json


class ToolParameter(BaseModel):
    """Definisi parameter untuk sebuah tool"""
    name: str
    type: str  # "string", "int", "float", "bool", "list", "dict"
    required: bool = True
    description: str = ""
    default: Optional[Any] = None


class Tool(BaseModel):
    """Registrasi sebuah tool yang bisa digunakan agent"""
    name: str
    description: str
    parameters: List[ToolParameter]
    
    def to_tool_definition(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class ExecutionStep(BaseModel):
    """Satu step dalam execution plan"""
    step: int
    tool: str
    params: Dict[str, Any]
    description: str
    critical: bool = False
    depends_on: Optional[List[int]] = None  # step numbers yang harus selesai duluan


class Plan(BaseModel):
    """Plan yang di-generate oleh LLM"""
    reasoning: str = Field(..., description="Alasan mengapa plan ini dipilih")
    next_steps: List[ExecutionStep]
    expected_outcome: str
    fallback_plan: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0, le=1)


class ExecutionResult(BaseModel):
    """Hasil execution dari sebuah step"""
    step_number: int
    tool: str
    params: Dict[str, Any]
    output: Optional[Any] = None
    error: Optional[str] = None
    success: bool
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.now)


class Reflection(BaseModel):
    """Hasil analisis setelah execution"""
    success: bool
    parsed_output: Optional[Dict[str, Any]] = None
    insights: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    recoverable: bool = True
    suggested_next_action: Optional[str] = None


class MemoryEntry(BaseModel):
    """Satu entry dalam memory"""
    key: str
    value: Any
    category: str  # "fact", "pattern", "failure", "success"
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    relevance_score: float = 1.0  # How relevant this entry is


class AgentResult(BaseModel):
    """Final result dari agent execution"""
    # Output
    output: Optional[Dict[str, Any]] = None
    success: bool
    error: Optional[str] = None
    
    # Metrics
    iterations: int
    total_tokens: int
    execution_time_seconds: float
    
    # Memory
    memory_used: Dict[str, int]
    session_id: str
    
    # Tracing
    reasoning_trace: List[str] = Field(default_factory=list)
    execution_history: List[ExecutionResult] = Field(default_factory=list)
    
    # For continuation
    saved_facts: List[MemoryEntry] = Field(default_factory=list)
    
    def to_json(self) -> str:
        """Serialize ke JSON"""
        return self.model_dump_json(indent=2)
    
    def summary(self) -> str:
        """Get summary string"""
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        return f"""
{status}
Iterations: {self.iterations}
Tokens: {self.total_tokens}
Time: {self.execution_time_seconds:.2f}s
Session: {self.session_id}
"""


class ConversationMessage(BaseModel):
    """Satu message dalam conversation history"""
    role: str  # "user", "assistant", "tool"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """State sebuah session"""
    session_id: str
    created_at: datetime
    last_updated: datetime = Field(default_factory=datetime.now)
    task: str
    iterations: int = 0
    max_iterations: int
    tokens_used: int = 0
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    facts: List[MemoryEntry] = Field(default_factory=list)
    status: str  # "in_progress", "completed", "failed", "paused"
    metadata: Dict[str, Any] = Field(default_factory=dict)
