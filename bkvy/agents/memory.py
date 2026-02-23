"""
Conversation memory management for agents
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logging import setup_logging

logger = setup_logging()


@dataclass
class ConversationMessage:
    """A single message in the conversation"""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None  # For tool results
    tool_calls: Optional[List[Dict]] = None  # For assistant tool calls


class ConversationMemory:
    """
    Manages conversation history for an agent session.
    Handles message storage, token counting, and truncation.
    """

    # Approximate tokens per character (conservative estimate)
    CHARS_PER_TOKEN = 4

    def __init__(self, max_tokens: int = 100000, system_prompt: Optional[str] = None):
        """
        Initialize conversation memory.

        Args:
            max_tokens: Maximum tokens to retain in history
            system_prompt: Optional system prompt to prepend
        """
        self._messages: List[ConversationMessage] = []
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    @property
    def messages(self) -> List[ConversationMessage]:
        """Get all messages"""
        return self._messages.copy()

    @property
    def system_prompt(self) -> Optional[str]:
        """Get the system prompt"""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        """Set the system prompt"""
        self._system_prompt = value

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None
    ) -> None:
        """Add a message to the conversation"""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
            tool_call_id=tool_call_id,
            tool_calls=tool_calls
        )
        self._messages.append(message)

        # Check if truncation needed
        self._truncate_if_needed()

        logger.debug(
            "Added message to memory",
            role=role,
            content_length=len(content),
            total_messages=len(self._messages)
        )

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user message"""
        self.add_message("user", content, metadata)

    def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict]] = None
    ) -> None:
        """Add an assistant message"""
        self.add_message("assistant", content, metadata, tool_calls=tool_calls)

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a tool result message"""
        self.add_message("tool", content, metadata, tool_call_id=tool_call_id)

    def get_messages(self) -> List[ConversationMessage]:
        """Get all messages"""
        return self._messages.copy()

    def get_last_message(self) -> Optional[ConversationMessage]:
        """Get the last message"""
        return self._messages[-1] if self._messages else None

    def get_last_assistant_message(self) -> Optional[ConversationMessage]:
        """Get the last assistant message"""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg
        return None

    def clear(self) -> None:
        """Clear all messages (keeps system prompt)"""
        self._messages.clear()

    def estimate_tokens(self) -> int:
        """Estimate total tokens in the conversation"""
        total_chars = 0

        if self._system_prompt:
            total_chars += len(self._system_prompt)

        for msg in self._messages:
            total_chars += len(msg.content)
            # Add overhead for role and structure
            total_chars += 20

        return total_chars // self.CHARS_PER_TOKEN

    def _truncate_if_needed(self) -> None:
        """Truncate old messages if over token limit"""
        while self.estimate_tokens() > self._max_tokens and len(self._messages) > 1:
            # Remove oldest non-system message
            removed = self._messages.pop(0)
            logger.debug(
                "Truncated message from memory",
                role=removed.role,
                content_length=len(removed.content)
            )

    def format_for_llm(self, provider: str = "openai") -> List[Dict[str, Any]]:
        """
        Format messages for specific LLM provider.

        Args:
            provider: LLM provider (openai, anthropic, gemini)

        Returns:
            List of formatted messages
        """
        messages = []

        # Add system prompt if present
        if self._system_prompt:
            if provider == "anthropic":
                # Anthropic handles system separately, but we include it in messages
                # The router will extract it
                messages.append({
                    "role": "system",
                    "content": self._system_prompt
                })
            else:
                messages.append({
                    "role": "system",
                    "content": self._system_prompt
                })

        # Add conversation messages
        for msg in self._messages:
            formatted = {"role": msg.role, "content": msg.content}

            # Handle tool calls in assistant messages
            if msg.tool_calls and provider in ("openai", "anthropic"):
                if provider == "openai":
                    formatted["tool_calls"] = msg.tool_calls
                # Anthropic uses a different format handled by the executor

            # Handle tool results
            if msg.role == "tool" and msg.tool_call_id:
                if provider == "openai":
                    formatted["tool_call_id"] = msg.tool_call_id

            messages.append(formatted)

        return messages

    def format_for_react(self) -> str:
        """
        Format conversation for ReAct-style text prompting.
        Returns a string representation of the conversation.
        """
        lines = []

        if self._system_prompt:
            lines.append(f"System: {self._system_prompt}")
            lines.append("")

        for msg in self._messages:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                tool_name = msg.metadata.get("tool_name", "tool")
                lines.append(f"Observation [{tool_name}]: {msg.content}")

            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary"""
        return {
            "system_prompt": self._system_prompt,
            "max_tokens": self._max_tokens,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                    "tool_call_id": msg.tool_call_id,
                    "tool_calls": msg.tool_calls
                }
                for msg in self._messages
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Deserialize memory from dictionary"""
        memory = cls(
            max_tokens=data.get("max_tokens", 100000),
            system_prompt=data.get("system_prompt")
        )

        for msg_data in data.get("messages", []):
            memory.add_message(
                role=msg_data["role"],
                content=msg_data["content"],
                metadata=msg_data.get("metadata", {}),
                tool_call_id=msg_data.get("tool_call_id"),
                tool_calls=msg_data.get("tool_calls")
            )

        return memory
