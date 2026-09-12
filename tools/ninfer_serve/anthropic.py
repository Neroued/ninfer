"""Anthropic Messages request encoding and semantic SSE classification."""

from __future__ import annotations

import json
from typing import Any, Sequence

from tools.ninfer_serve.client import ProtocolEvent, ProtocolRequest, ServeProtocolError
from tools.streaming_http.sse import SseEvent


def anthropic_request(
    model: str,
    messages: Sequence[dict[str, Any]],
    max_output_tokens: int,
    *,
    system: str | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    stream: bool = True,
) -> ProtocolRequest:
    if max_output_tokens <= 0:
        raise ValueError("Anthropic max_output_tokens must be positive")
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "stream": stream,
    }
    if system is not None:
        payload["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    if tools is not None:
        payload["tools"] = list(tools)
        payload["tool_choice"] = {"type": "auto"}
    return ProtocolRequest(
        protocol="anthropic_messages",
        path="/v1/messages",
        payload=payload,
        stream=stream,
    )


class AnthropicStreamAdapter:
    def consume(self, event: SseEvent) -> list[ProtocolEvent]:
        try:
            payload = json.loads(event.data)
        except json.JSONDecodeError as error:
            raise ServeProtocolError("Anthropic SSE data is not JSON") from error
        if not isinstance(payload, dict):
            raise ServeProtocolError("Anthropic SSE payload is not an object")
        event_type = payload.get("type")
        if not isinstance(event_type, str):
            raise ServeProtocolError("Anthropic SSE payload has no type")
        if event.event != "message" and event.event != event_type:
            raise ServeProtocolError("Anthropic event header and payload type differ")
        if event_type == "message_start":
            message = payload.get("message")
            response_id = message.get("id") if isinstance(message, dict) else None
            return [
                ProtocolEvent(
                    "accepted",
                    event_type,
                    event.received_ns,
                    payload=payload,
                    response_id=response_id if isinstance(response_id, str) else None,
                )
            ]
        if event_type == "content_block_delta":
            delta = payload.get("delta")
            output = ""
            if isinstance(delta, dict):
                for key in ("text", "thinking", "partial_json"):
                    value = delta.get(key)
                    if isinstance(value, str) and value:
                        output += value
            if output:
                return [
                    ProtocolEvent(
                        "model_output",
                        event_type,
                        event.received_ns,
                        payload=payload,
                        output=output,
                    )
                ]
        if event_type == "message_stop":
            return [ProtocolEvent("terminal", event_type, event.received_ns, payload=payload)]
        if event_type == "error":
            error = payload.get("error")
            return [
                ProtocolEvent(
                    "error",
                    event_type,
                    event.received_ns,
                    payload=payload,
                    error_code=(
                        error.get("type") if isinstance(error, dict) and isinstance(error.get("type"), str) else None
                    ),
                    error_message=(
                        error.get("message")
                        if isinstance(error, dict) and isinstance(error.get("message"), str)
                        else None
                    ),
                )
            ]
        return [
            ProtocolEvent("metadata", event_type, event.received_ns, payload=payload)
        ]
