"""Base agent with LLM integration and streaming support."""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Callable, Iterator, AsyncIterator

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


@dataclass
class StreamEvent:
    """Event emitted during streaming."""

    type: Literal["thinking", "text", "tool_start", "tool_input", "tool_end", "done"]
    content: str = ""
    tool_name: str | None = None
    tool_id: str | None = None


class BaseAgent(ABC):
    """Base class for all agents with streaming support."""

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

        # Streaming callbacks
        self.on_stream: Callable[[StreamEvent], None] | None = None

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

    def _emit(self, event: StreamEvent):
        """Emit a stream event if callback is set."""
        if self.on_stream:
            self.on_stream(event)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Send messages to the LLM and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            model: Override the default model
            stream: Enable streaming output

        Returns:
            LLMResponse with content and/or tool calls
        """
        model = model or self.model
        provider = self.settings.get_provider(model)
        model_id = self.settings.get_model_id(model)

        if provider == "anthropic":
            if stream:
                return self._chat_anthropic_stream(messages, tools, model_id)
            return self._chat_anthropic(messages, tools, model_id)
        else:
            if stream:
                return self._chat_openai_stream(messages, tools, model_id)
            return self._chat_openai(messages, tools, model_id)

    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Async version of chat."""
        model = model or self.model
        provider = self.settings.get_provider(model)
        model_id = self.settings.get_model_id(model)

        if provider == "anthropic":
            if stream:
                return await self._chat_anthropic_stream_async(messages, tools, model_id)
            return await self._chat_anthropic_async(messages, tools, model_id)
        else:
            if stream:
                return await self._chat_openai_stream_async(messages, tools, model_id)
            return await self._chat_openai_async(messages, tools, model_id)

    def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Call Anthropic API."""
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

    def _chat_anthropic_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Call Anthropic API with streaming."""
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

        content_parts = []
        tool_calls = []
        current_tool_id = None
        current_tool_name = None
        current_tool_input = ""
        input_tokens = 0
        output_tokens = 0

        with self.anthropic_client.messages.stream(**kwargs) as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'message_start':
                        if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                            input_tokens = event.message.usage.input_tokens

                    elif event.type == 'content_block_start':
                        if hasattr(event, 'content_block'):
                            block = event.content_block
                            if block.type == 'tool_use':
                                current_tool_id = block.id
                                current_tool_name = block.name
                                current_tool_input = ""
                                self._emit(StreamEvent(
                                    type="tool_start",
                                    tool_name=current_tool_name,
                                    tool_id=current_tool_id,
                                ))

                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            delta = event.delta
                            if delta.type == 'text_delta':
                                content_parts.append(delta.text)
                                self._emit(StreamEvent(type="text", content=delta.text))
                            elif delta.type == 'input_json_delta':
                                current_tool_input += delta.partial_json
                                self._emit(StreamEvent(
                                    type="tool_input",
                                    content=delta.partial_json,
                                    tool_name=current_tool_name,
                                    tool_id=current_tool_id,
                                ))

                    elif event.type == 'content_block_stop':
                        if current_tool_id:
                            try:
                                args = json.loads(current_tool_input) if current_tool_input else {}
                            except json.JSONDecodeError:
                                args = {}
                            tool_calls.append(ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=args,
                            ))
                            self._emit(StreamEvent(
                                type="tool_end",
                                tool_name=current_tool_name,
                                tool_id=current_tool_id,
                            ))
                            current_tool_id = None
                            current_tool_name = None
                            current_tool_input = ""

                    elif event.type == 'message_delta':
                        if hasattr(event, 'usage'):
                            output_tokens = event.usage.output_tokens

        self._emit(StreamEvent(type="done"))

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return LLMResponse(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            stop_reason="end_turn",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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

    async def _chat_anthropic_stream_async(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Async streaming call to Anthropic API."""
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

        content_parts = []
        tool_calls = []
        current_tool_id = None
        current_tool_name = None
        current_tool_input = ""
        input_tokens = 0
        output_tokens = 0

        async with self.anthropic_async_client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'message_start':
                        if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                            input_tokens = event.message.usage.input_tokens

                    elif event.type == 'content_block_start':
                        if hasattr(event, 'content_block'):
                            block = event.content_block
                            if block.type == 'tool_use':
                                current_tool_id = block.id
                                current_tool_name = block.name
                                current_tool_input = ""
                                self._emit(StreamEvent(
                                    type="tool_start",
                                    tool_name=current_tool_name,
                                    tool_id=current_tool_id,
                                ))

                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            delta = event.delta
                            if delta.type == 'text_delta':
                                content_parts.append(delta.text)
                                self._emit(StreamEvent(type="text", content=delta.text))
                            elif delta.type == 'input_json_delta':
                                current_tool_input += delta.partial_json
                                self._emit(StreamEvent(
                                    type="tool_input",
                                    content=delta.partial_json,
                                    tool_name=current_tool_name,
                                    tool_id=current_tool_id,
                                ))

                    elif event.type == 'content_block_stop':
                        if current_tool_id:
                            try:
                                args = json.loads(current_tool_input) if current_tool_input else {}
                            except json.JSONDecodeError:
                                args = {}
                            tool_calls.append(ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=args,
                            ))
                            self._emit(StreamEvent(
                                type="tool_end",
                                tool_name=current_tool_name,
                                tool_id=current_tool_id,
                            ))
                            current_tool_id = None
                            current_tool_name = None
                            current_tool_input = ""

                    elif event.type == 'message_delta':
                        if hasattr(event, 'usage'):
                            output_tokens = event.usage.output_tokens

        self._emit(StreamEvent(type="done"))

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return LLMResponse(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            stop_reason="end_turn",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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

    def _chat_openai_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Call OpenAI API with streaming."""
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
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if openai_tools:
            kwargs["tools"] = openai_tools

        content_parts = []
        tool_calls_data = {}  # id -> {name, arguments}
        input_tokens = 0
        output_tokens = 0

        stream = self.openai_client.chat.completions.create(**kwargs)

        for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            if chunk.choices:
                delta = chunk.choices[0].delta

                if delta.content:
                    content_parts.append(delta.content)
                    self._emit(StreamEvent(type="text", content=delta.content))

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.id:
                            tool_calls_data[tc.id] = {
                                "name": tc.function.name if tc.function else "",
                                "arguments": ""
                            }
                            self._emit(StreamEvent(
                                type="tool_start",
                                tool_name=tc.function.name if tc.function else "",
                                tool_id=tc.id,
                            ))
                        if tc.function and tc.function.arguments:
                            # Find the tool call to update
                            for tid, data in tool_calls_data.items():
                                if tc.id == tid or (not tc.id and data["name"]):
                                    data["arguments"] += tc.function.arguments
                                    self._emit(StreamEvent(
                                        type="tool_input",
                                        content=tc.function.arguments,
                                        tool_name=data["name"],
                                        tool_id=tid,
                                    ))
                                    break

        self._emit(StreamEvent(type="done"))

        # Build tool calls
        tool_calls = []
        for tid, data in tool_calls_data.items():
            try:
                args = json.loads(data["arguments"]) if data["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tid,
                name=data["name"],
                arguments=args,
            ))
            self._emit(StreamEvent(type="tool_end", tool_name=data["name"], tool_id=tid))

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return LLMResponse(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            stop_reason="stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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

    async def _chat_openai_stream_async(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model_id: str,
    ) -> LLMResponse:
        """Async streaming call to OpenAI API."""
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
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if openai_tools:
            kwargs["tools"] = openai_tools

        content_parts = []
        tool_calls_data = {}
        input_tokens = 0
        output_tokens = 0

        stream = await self.openai_async_client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            if chunk.choices:
                delta = chunk.choices[0].delta

                if delta.content:
                    content_parts.append(delta.content)
                    self._emit(StreamEvent(type="text", content=delta.content))

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.id:
                            tool_calls_data[tc.id] = {
                                "name": tc.function.name if tc.function else "",
                                "arguments": ""
                            }
                            self._emit(StreamEvent(
                                type="tool_start",
                                tool_name=tc.function.name if tc.function else "",
                                tool_id=tc.id,
                            ))
                        if tc.function and tc.function.arguments:
                            for tid, data in tool_calls_data.items():
                                if tc.id == tid or (not tc.id and data["name"]):
                                    data["arguments"] += tc.function.arguments
                                    self._emit(StreamEvent(
                                        type="tool_input",
                                        content=tc.function.arguments,
                                        tool_name=data["name"],
                                        tool_id=tid,
                                    ))
                                    break

        self._emit(StreamEvent(type="done"))

        tool_calls = []
        for tid, data in tool_calls_data.items():
            try:
                args = json.loads(data["arguments"]) if data["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tid,
                name=data["name"],
                arguments=args,
            ))
            self._emit(StreamEvent(type="tool_end", tool_name=data["name"], tool_id=tid))

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return LLMResponse(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            stop_reason="stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(task, **kwargs))
