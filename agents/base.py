"""Base agent with LLM integration."""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
import openai

from config import get_settings


@dataclass
class Message:
    """A message in the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str | list[dict]


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
    ):
        self.settings = get_settings()
        self.model = model or self.settings.default_model
        self.system_prompt = system_prompt or ""
        self.messages: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Initialize clients
        self._anthropic_client = None
        self._openai_client = None
        self._anthropic_async_client = None
        self._openai_async_client = None

    @property
    def anthropic_client(self) -> anthropic.Anthropic:
        """Lazy-load Anthropic client."""
        if self._anthropic_client is None:
            api_key = self.settings.anthropic_api_key
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._anthropic_client = anthropic.Anthropic(
                api_key=api_key.get_secret_value()
            )
        return self._anthropic_client

    @property
    def anthropic_async_client(self) -> anthropic.AsyncAnthropic:
        """Lazy-load async Anthropic client."""
        if self._anthropic_async_client is None:
            api_key = self.settings.anthropic_api_key
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._anthropic_async_client = anthropic.AsyncAnthropic(
                api_key=api_key.get_secret_value()
            )
        return self._anthropic_async_client

    @property
    def openai_client(self) -> openai.OpenAI:
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            api_key = self.settings.openai_api_key
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not set")
            self._openai_client = openai.OpenAI(
                api_key=api_key.get_secret_value()
            )
        return self._openai_client

    @property
    def openai_async_client(self) -> openai.AsyncOpenAI:
        """Lazy-load async OpenAI client."""
        if self._openai_async_client is None:
            api_key = self.settings.openai_api_key
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not set")
            self._openai_async_client = openai.AsyncOpenAI(
                api_key=api_key.get_secret_value()
            )
        return self._openai_async_client

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            model: Override the default model

        Returns:
            LLMResponse with content and/or tool calls
        """
        model = model or self.model
        provider = self.settings.get_provider(model)
        model_id = self.settings.get_model_id(model)

        if provider == "anthropic":
            return self._chat_anthropic(messages, tools, model_id)
        else:
            return self._chat_openai(messages, tools, model_id)

    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Async version of chat."""
        model = model or self.model
        provider = self.settings.get_provider(model)
        model_id = self.settings.get_model_id(model)

        if provider == "anthropic":
            return await self._chat_anthropic_async(messages, tools, model_id)
        else:
            return await self._chat_openai_async(messages, tools, model_id)

    def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Call Anthropic API."""
        # Separate system message
        system = self.system_prompt
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {
            "model": model_id,
            "max_tokens": 8192,
            "messages": chat_messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = tools

        response = self.anthropic_client.messages.create(**kwargs)

        # Track tokens
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        # Parse response
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def _chat_anthropic_async(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Async call to Anthropic API."""
        system = self.system_prompt
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {
            "model": model_id,
            "max_tokens": 8192,
            "messages": chat_messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = tools

        response = await self.anthropic_async_client.messages.create(**kwargs)

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _chat_openai(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Call OpenAI API."""
        chat_messages = []
        has_system = any(m["role"] == "system" for m in messages)

        if self.system_prompt and not has_system:
            chat_messages.append({"role": "system", "content": self.system_prompt})

        chat_messages.extend(messages)

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]

        kwargs = {
            "model": model_id,
            "messages": chat_messages,
        }

        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.openai_client.chat.completions.create(**kwargs)

        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        message = response.choices[0].message
        content = message.content
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    async def _chat_openai_async(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Async call to OpenAI API."""
        chat_messages = []
        has_system = any(m["role"] == "system" for m in messages)

        if self.system_prompt and not has_system:
            chat_messages.append({"role": "system", "content": self.system_prompt})

        chat_messages.extend(messages)

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]

        kwargs = {
            "model": model_id,
            "messages": chat_messages,
        }

        if openai_tools:
            kwargs["tools"] = openai_tools

        response = await self.openai_async_client.chat.completions.create(**kwargs)

        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        message = response.choices[0].message
        content = message.content
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def get_cost(self) -> float:
        """Calculate total cost based on token usage."""
        costs = self.settings.cost_per_million.get(self.model, {"input": 0, "output": 0})
        input_cost = (self.total_input_tokens / 1_000_000) * costs["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost

    @abstractmethod
    def run(self, task: str, **kwargs) -> Any:
        """Run the agent on a task. Must be implemented by subclasses."""
        pass

    async def run_async(self, task: str, **kwargs) -> Any:
        """Async version of run. Override in subclasses for async support."""
        # Default: run sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(task, **kwargs))
