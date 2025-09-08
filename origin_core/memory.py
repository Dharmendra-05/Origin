"""
Agent Memory module managing short-term conversation context and long-term vector storage.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class MemoryMessage(BaseModel):
    """Represents a single message turn in agent memory."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMemory:
    """Sliding-window short-term memory buffer for AI agents."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[MemoryMessage] = []

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Appends a new message turn to memory."""
        self.messages.append(MemoryMessage(role=role, content=content, metadata=metadata))
        if len(self.messages) > self.max_turns:
            # Preserve system prompt if present at index 0
            if self.messages[0].role == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_turns - 1):]
            else:
                self.messages = self.messages[-self.max_turns:]

    def get_messages(self) -> List[MemoryMessage]:
        """Returns current conversation history."""
        return self.messages

    def clear(self) -> None:
        """Clears memory state."""
        self.messages.clear()
