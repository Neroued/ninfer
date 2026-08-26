"""HTTP/1.1 request timing for an already-running local service."""

from __future__ import annotations

import http.client
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping
from urllib.parse import urlsplit


class HttpClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpResponseHead:
    status: int
    reason: str
    headers: dict[str, str]
    received_ns: int


@dataclass
class HttpExchangeResult:
    sent_ns: int | None = None
    body_sent_ns: int | None = None
    headers_ns: int | None = None
    ended_ns: int | None = None
    status: int | None = None
    reason: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    error: str | None = None
    cancel_requested: bool = False
    cancelled: bool = False
    cancel_ns: int | None = None


class PreparedExchange:
    """One preconnected, single-use HTTP/1.1 exchange."""

    def __init__(
        self,
        connection: http.client.HTTPConnection,
        method: str,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> None:
        self._connection = connection
        self._transport = connection.sock
        if self._transport is None:
            raise HttpClientError("prepared HTTP exchange has no connected transport")
        self._body = body
        self._lock = threading.Lock()
        self._cancel_requested = False
        self._cancel_ns: int | None = None
        self._started = False
        self._finished = False

        # Request-line/header construction and connection establishment are outside t0.  CPython's
        # HTTPConnection buffers these values until endheaders(), where the first socket write
        # occurs.
        self._connection.putrequest(method, path)
        for name, value in headers.items():
            self._connection.putheader(name, value)

    @property
    def body_bytes(self) -> int:
        return len(self._body)

    @property
    def cancel_ns(self) -> int | None:
        with self._lock:
            return self._cancel_ns

    def cancel(self) -> int:
        now = time.perf_counter_ns()
        with self._lock:
            if self._cancel_ns is None:
                self._cancel_ns = now
            self._cancel_requested = True
            transport = None if self._finished else self._transport
        if transport is not None:
            try:
                # HTTPConnection drops its `sock` reference for a will-close response after
                # getresponse(), while HTTPResponse keeps reading from the same transport.  Keep
                # and shut down that exact preconnected socket so cancellation wakes either
                # getresponse() or response.read1().
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return self._cancel_ns

    def execute(
        self,
        *,
        on_sent: Callable[[int], None] | None = None,
        on_body_sent: Callable[[int], None] | None = None,
        on_headers: Callable[[HttpResponseHead], None] | None = None,
        on_chunk: Callable[[bytes, int], None] | None = None,
        chunk_size: int = 64 * 1024,
    ) -> HttpExchangeResult:
        with self._lock:
            if self._started:
                raise HttpClientError("prepared HTTP exchange is single-use")
            self._started = True
            cancelled_before_start = self._cancel_requested
        result = HttpExchangeResult()
        if cancelled_before_start:
            result.cancel_requested = True
            result.cancelled = True
            result.cancel_ns = self.cancel_ns
            result.ended_ns = time.perf_counter_ns()
            with self._lock:
                self._finished = True
            self._connection.close()
            return result

        chunks: list[bytes] = []
        response: http.client.HTTPResponse | None = None
        try:
            result.sent_ns = time.perf_counter_ns()
            if on_sent is not None:
                on_sent(result.sent_ns)
            self._connection.endheaders(self._body)
            result.body_sent_ns = time.perf_counter_ns()
            if on_body_sent is not None:
                on_body_sent(result.body_sent_ns)

            response = self._connection.getresponse()
            result.headers_ns = time.perf_counter_ns()
            result.status = int(response.status)
            result.reason = str(response.reason or "")
            result.headers = {name.lower(): value for name, value in response.getheaders()}
            if on_headers is not None:
                on_headers(
                    HttpResponseHead(
                        status=result.status,
                        reason=result.reason,
                        headers=dict(result.headers),
                        received_ns=result.headers_ns,
                    )
                )

            while True:
                block = response.read1(chunk_size)
                if not block:
                    break
                received_ns = time.perf_counter_ns()
                chunks.append(block)
                if on_chunk is not None:
                    on_chunk(block, received_ns)
            result.ended_ns = time.perf_counter_ns()
        except (OSError, http.client.HTTPException, TimeoutError, ssl.SSLError) as error:
            result.ended_ns = time.perf_counter_ns()
            with self._lock:
                cancel_requested = self._cancel_requested
                cancel_ns = self._cancel_ns
            result.cancel_requested = cancel_requested
            result.cancelled = cancel_requested
            result.cancel_ns = cancel_ns
            if not cancel_requested:
                result.error = f"{type(error).__name__}: {error}"
        finally:
            if response is not None:
                response.close()
            self._connection.close()
            with self._lock:
                self._finished = True

        result.body = b"".join(chunks)
        with self._lock:
            result.cancel_requested = self._cancel_requested
            result.cancel_ns = self._cancel_ns
        return result


class StreamingHttpClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise HttpClientError("base URL must be an absolute http:// or https:// URL")
        if parsed.query or parsed.fragment:
            raise HttpClientError("base URL must not contain a query or fragment")
        if timeout_seconds <= 0:
            raise HttpClientError("timeout must be positive")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._prefix = parsed.path.rstrip("/")
        self._timeout = float(timeout_seconds)

    def _new_connection(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                self._host, self._port, timeout=self._timeout
            )
        else:
            connection = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout
            )
        connection.connect()
        return connection

    def prepare(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
    ) -> PreparedExchange:
        if not path.startswith("/"):
            raise HttpClientError("request path must start with '/'")
        request_headers = dict(headers or {})
        lower = {name.lower() for name in request_headers}
        if "content-length" not in lower:
            request_headers["Content-Length"] = str(len(body))
        if "connection" not in lower:
            request_headers["Connection"] = "close"
        connection = self._new_connection()
        return PreparedExchange(
            connection,
            method.upper(),
            self._prefix + path,
            body,
            request_headers,
        )
