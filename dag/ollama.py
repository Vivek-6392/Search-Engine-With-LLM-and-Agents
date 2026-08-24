import json
import urllib.request
import uuid
from collections.abc import Sequence
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


class ChatOllama(BaseChatModel):
    """Chat model client for local Ollama instances with tool-calling support."""

    model: str = "mistral-nemo:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: int = 300
    bound_tools: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        converted = [convert_to_openai_tool(t) for t in tools]
        return self.__class__(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout,
            bound_tools=converted,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                m: dict[str, Any] = {"role": "assistant", "content": str(msg.content or "")}
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["args"],
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                ollama_messages.append(m)
            elif isinstance(msg, ToolMessage):
                ollama_messages.append({
                    "role": "tool",
                    "content": str(msg.content),
                })
            else:
                ollama_messages.append({"role": "user", "content": str(msg.content)})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.bound_tools:
            payload["tools"] = self.bound_tools

        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg_data = data.get("message", {})
            content = msg_data.get("content", "")
            tool_calls = []

            for tc in msg_data.get("tool_calls", []):
                func = tc.get("function", {})
                tool_calls.append({
                    "name": func.get("name"),
                    "args": func.get("arguments", {}),
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                })

            ai_msg = AIMessage(
                content=content,
                tool_calls=tool_calls,
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def get_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Fetch installed local models from Ollama."""
    fallback_models = [
        "mistral-nemo:latest",
        "llama3:latest",
        "mistral:latest",
        "phi3:latest",
        "codellama:latest",
        "gemma:2b",
    ]
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [
                m["name"]
                for m in data.get("models", [])
                if not any(x in m["name"] for x in ["embed", "mxbai"])
            ]
            if models:
                return models
    except Exception:
        pass
    return fallback_models
