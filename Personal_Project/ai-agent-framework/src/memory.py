"""
Memory System untuk persistent memory across sessions
Supports dalam-memory dan file-based storage
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .models import (
    MemoryEntry,
    ConversationMessage,
    SessionState
)

logger = logging.getLogger(__name__)


class InMemoryStore:
    """In-memory storage (tidak persistent)"""
    
    def __init__(self):
        self.entries: Dict[str, Any] = {}
    
    def store(self, key: str, value: Any):
        self.entries[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        return self.entries.get(key)
    
    def get_all(self) -> Dict[str, Any]:
        return self.entries.copy()
    
    def clear(self):
        self.entries.clear()


class FileStore:
    """File-based storage untuk persistence"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store(self, key: str, value: Any):
        """Store value ke file"""
        file_path = self.base_path / f"{key}.json"
        
        try:
            # Handle serialization
            if isinstance(value, list) and value and hasattr(value[0], 'dict'):
                # Pydantic models
                data = [item.dict() if hasattr(item, 'dict') else item for item in value]
            elif hasattr(value, 'dict'):
                data = value.dict()
            else:
                data = value
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Stored {key} to {file_path}")
        
        except Exception as e:
            logger.error(f"Error storing {key}: {e}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value dari file"""
        file_path = self.base_path / f"{key}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving {key}: {e}")
            return None
    
    def get_all(self) -> Dict[str, Any]:
        """Get semua entries"""
        result = {}
        
        for file_path in self.base_path.glob("*.json"):
            key = file_path.stem
            data = self.retrieve(key)
            if data:
                result[key] = data
        
        return result
    
    def delete(self, key: str):
        """Delete entry"""
        file_path = self.base_path / f"{key}.json"
        
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted {key}")


class Memory:
    """Main memory system yang combine in-memory + persistent storage"""
    
    def __init__(
        self,
        memory_path: str = "./memory",
        use_persistent: bool = True
    ):
        self.memory_path = memory_path
        self.use_persistent = use_persistent
        
        # Initialize stores
        self.in_memory = InMemoryStore()
        self.persistent = FileStore(
            os.path.join(memory_path, "persistent")
        ) if use_persistent else None
        
        # Session management
        self.session_id: Optional[str] = None
        self.current_session: Optional[SessionState] = None
        
        # Conversation history
        self.conversation_history: List[ConversationMessage] = []
        
        # Facts dan knowledge
        self.facts: List[MemoryEntry] = []
        
        # Failed attempts untuk learning
        self.failed_attempts: List[Dict[str, Any]] = []
    
    def start_session(
        self,
        task: str,
        session_id: Optional[str] = None,
        max_iterations: int = 10
    ):
        """Start sebuah session baru atau load existing"""
        
        if session_id is None:
            # Generate new session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        
        # Try load existing session
        existing = self._load_session_from_disk()
        
        if existing:
            self.current_session = existing
            logger.info(f"Loaded existing session: {session_id}")
        else:
            # Create new session
            self.current_session = SessionState(
                session_id=session_id,
                created_at=datetime.now(),
                task=task,
                max_iterations=max_iterations,
                status="in_progress"
            )
            logger.info(f"Created new session: {session_id}")
        
        # Load session facts
        self.facts = self.current_session.facts
        self.conversation_history = self.current_session.conversation_history
    
    def add_message(
        self,
        role: str,
        content: str,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add message ke conversation history"""
        
        message = ConversationMessage(
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            metadata=metadata or {}
        )
        
        self.conversation_history.append(message)
        
        if self.current_session and self.current_session.conversation_history is not self.conversation_history:
            self.current_session.conversation_history.append(message)
    
    def add_fact(
        self,
        key: str,
        value: Any,
        category: str = "fact",
        relevance_score: float = 1.0
    ):
        """Add fact ke memory"""
        
        entry = MemoryEntry(
            key=key,
            value=value,
            category=category,
            session_id=self.session_id or "unknown",
            relevance_score=relevance_score
        )
        
        self.facts.append(entry)
        
        if self.current_session and self.current_session.facts is not self.facts:
            self.current_session.facts.append(entry)
    
    def get_facts(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all facts, optionally filtered by category"""
        
        facts = self.facts
        
        if category:
            facts = [f for f in facts if f.category == category]
        
        # Sort by relevance
        facts = sorted(facts, key=lambda f: f.relevance_score, reverse=True)
        
        return [
            {
                "key": f.key,
                "value": f.value,
                "category": f.category,
                "relevance": f.relevance_score
            }
            for f in facts
        ]
    
    def add_failed_attempt(
        self,
        task: str,
        attempt: Dict[str, Any],
        error: str,
        reason: str
    ):
        """Log failed attempt untuk learning"""
        
        failure = {
            "task": task,
            "attempt": attempt,
            "error": error,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        
        self.failed_attempts.append(failure)
        
        # Save to disk
        if self.use_persistent and self.persistent:
            key = f"failure_{datetime.now().timestamp()}"
            self.persistent.store(key, failure)
    
    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        
        history = self.conversation_history
        
        if last_n:
            history = history[-last_n:]
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in history
        ]
    
    def get_context_summary(self) -> str:
        """Get summary dari memory untuk LLM context"""
        
        facts_text = ""
        if self.facts:
            facts_text = "Known Facts:\n"
            for fact in self.get_facts()[:5]:  # Top 5 by relevance
                facts_text += f"  - {fact['key']}: {fact['value']}\n"
        
        history_text = ""
        if self.conversation_history:
            history_text = "\nRecent History:\n"
            for msg in self.conversation_history[-3:]:  # Last 3 messages
                history_text += f"  {msg.role}: {msg.content[:100]}...\n"
        
        return facts_text + history_text
    
    def save_session(self):
        """Save current session ke disk"""
        
        if not self.current_session:
            return
        
        # Update session state
        self.current_session.last_updated = datetime.now()
        self.current_session.iterations += 1
        
        # Save to persistent storage
        if self.use_persistent and self.persistent:
            self.persistent.store(
                f"session_{self.session_id}",
                self.current_session.dict()
            )
            logger.debug(f"Saved session: {self.session_id}")
    
    def end_session(self, status: str = "completed"):
        """End current session"""
        
        if self.current_session:
            self.current_session.status = status
            self.current_session.last_updated = datetime.now()
            self.save_session()
            logger.info(f"Session ended: {self.session_id} ({status})")
    
    def _load_session_from_disk(self) -> Optional[SessionState]:
        """Load session dari disk"""
        
        if not self.use_persistent or not self.persistent or not self.session_id:
            return None
        
        data = self.persistent.retrieve(f"session_{self.session_id}")
        
        if data:
            try:
                return SessionState(**data)
            except Exception as e:
                logger.error(f"Error loading session: {e}")
        
        return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List semua sessions"""
        
        if not self.use_persistent or not self.persistent:
            return []
        
        sessions = []
        for key, value in self.persistent.get_all().items():
            if key.startswith("session_"):
                sessions.append({
                    "session_id": value.get("session_id"),
                    "created_at": value.get("created_at"),
                    "status": value.get("status"),
                    "task": value.get("task")
                })
        
        return sorted(
            sessions,
            key=lambda s: s.get("created_at", ""),
            reverse=True
        )
    
    def clear_session(self):
        """Clear current session memory (in-memory only)"""
        
        self.conversation_history.clear()
        self.facts.clear()
        self.failed_attempts.clear()
        logger.info("Cleared current session memory")
