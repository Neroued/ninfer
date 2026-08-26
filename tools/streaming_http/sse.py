"""Incremental Server-Sent Events decoding with frame-completion timestamps."""

from __future__ import annotations

import codecs
from dataclasses import dataclass


class SseDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: str
    received_ns: int
    event_id: str | None = None
    retry_ms: int | None = None


class SseDecoder:
    """Decode arbitrarily fragmented UTF-8 SSE bytes.

    `received_ns` on an emitted event is the timestamp of the chunk that completed its terminating
    blank line.  This is the only timestamp a protocol adapter may use for first-output timing.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._buffer = ""
        self._pending_cr = False

    def _normalize(self, text: str, *, final: bool) -> str:
        if self._pending_cr:
            text = "\r" + text
            self._pending_cr = False
        if not final and text.endswith("\r"):
            self._pending_cr = True
            text = text[:-1]
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _parse(block: str, received_ns: int) -> SseEvent | None:
        event_type = "message"
        data: list[str] = []
        event_id: str | None = None
        retry_ms: int | None = None
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_type = value
            elif field == "data":
                data.append(value)
            elif field == "id" and "\x00" not in value:
                event_id = value
            elif field == "retry" and value.isdigit():
                retry_ms = int(value)
        if not data and event_type == "message" and event_id is None and retry_ms is None:
            return None
        return SseEvent(
            event=event_type,
            data="\n".join(data),
            received_ns=received_ns,
            event_id=event_id,
            retry_ms=retry_ms,
        )

    def feed(self, data: bytes, received_ns: int) -> list[SseEvent]:
        try:
            text = self._decoder.decode(data, final=False)
        except UnicodeDecodeError as error:
            raise SseDecodeError(f"SSE stream is not valid UTF-8: {error}") from error
        self._buffer += self._normalize(text, final=False)
        events: list[SseEvent] = []
        while True:
            boundary = self._buffer.find("\n\n")
            if boundary < 0:
                break
            block = self._buffer[:boundary]
            self._buffer = self._buffer[boundary + 2 :]
            event = self._parse(block, received_ns)
            if event is not None:
                events.append(event)
        return events

    def finish(self, received_ns: int) -> list[SseEvent]:
        try:
            tail = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise SseDecodeError(f"SSE stream ends in invalid UTF-8: {error}") from error
        self._buffer += self._normalize(tail, final=True)
        if self._pending_cr:
            self._buffer += "\n"
            self._pending_cr = False
        if not self._buffer:
            return []
        # SSE dispatches the last event at EOF even when the final blank line is omitted.
        block = self._buffer.rstrip("\n")
        self._buffer = ""
        event = self._parse(block, received_ns)
        return [] if event is None else [event]
