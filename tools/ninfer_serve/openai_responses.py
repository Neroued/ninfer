"""OpenAI Responses request encoding and semantic SSE classification."""

from __future__ import annotations

import json
from typing import Any, Sequence

from tools.ninfer_serve.client import ProtocolEvent, ProtocolRequest, ServeProtocolError
from tools.streaming_http.sse import SseEvent


def responses_request(
    model: str,
    input_value: str | Sequence[dict[str, Any]],
    max_output_tokens: int,
    *,
    store: bool,
    previous_response_id: str | None = None,
    stream: bool = True,
) -> ProtocolRequest:
    if max_output_tokens < 0:
        raise ValueError("Responses max_output_tokens must be non-negative")
    payload: dict[str, Any] = {
        "model": model,
        "input": input_value,
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "store": store,
        "stream": stream,
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    return ProtocolRequest(
        protocol="openai_responses",
        path="/v1/responses",
        payload=payload,
        stream=stream,
    )


class ResponsesStreamAdapter:
    _OUTPUT_TYPES = {
        "response.output_text.delta",
        "response.reasoning_text.delta",
        "response.function_call_arguments.delta",
    }
    _TERMINAL_TYPES = {
        "response.completed",
        "response.incomplete",
        "response.failed",
    }

    def consume(self, event: SseEvent) -> list[ProtocolEvent]:
        try:
            payload = json.loads(event.data)
        except json.JSONDecodeError as error:
            raise ServeProtocolError("Responses SSE data is not JSON") from error
        if not isinstance(payload, dict):
            raise ServeProtocolError("Responses SSE payload is not an object")
        event_type = payload.get("type")
        if not isinstance(event_type, str):
            raise ServeProtocolError("Responses SSE payload has no type")
        if event.event != "message" and event.event != event_type:
            raise ServeProtocolError("Responses event header and payload type differ")
        response = payload.get("response")
        response_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(response_id, str):
            response_id = None
        if event_type == "response.created":
            return [
                ProtocolEvent(
                    "accepted",
                    event_type,
                    event.received_ns,
                    payload=payload,
                    response_id=response_id,
                )
            ]
        if event_type in self._OUTPUT_TYPES:
            delta = payload.get("delta")
            if isinstance(delta, str) and delta:
                return [
                    ProtocolEvent(
                        "model_output",
                        event_type,
                        event.received_ns,
                        payload=payload,
                        response_id=response_id,
                        output=delta,
                    )
                ]
        if event_type in self._TERMINAL_TYPES:
            if event_type == "response.failed":
                error = response.get("error") if isinstance(response, dict) else None
                return [
                    ProtocolEvent(
                        "error",
                        event_type,
                        event.received_ns,
                        payload=payload,
                        response_id=response_id,
                        error_code=(
                            error.get("code") if isinstance(error, dict) and isinstance(error.get("code"), str) else None
                        ),
                        error_message=(
                            error.get("message")
                            if isinstance(error, dict) and isinstance(error.get("message"), str)
                            else None
                        ),
                    )
                ]
            return [
                ProtocolEvent(
                    "terminal",
                    event_type,
                    event.received_ns,
                    payload=payload,
                    response_id=response_id,
                )
            ]
        return [
            ProtocolEvent(
                "metadata",
                event_type,
                event.received_ns,
                payload=payload,
                response_id=response_id,
            )
        ]
