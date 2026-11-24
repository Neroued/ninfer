from __future__ import annotations

import errno
import http.client
import socket
import threading

import pytest

from tools.bench.ttft.execution import RequestHandle
from tools.ninfer_serve.client import PreparedServeExchange, ProtocolRequest
from tools.streaming_http.client import PreparedExchange


@pytest.fixture
def exchange():
    transport, peer = socket.socketpair()
    transport.settimeout(2)
    peer.settimeout(2)
    connection = http.client.HTTPConnection("test")
    connection.sock = transport
    request = ProtocolRequest("openai_chat", "/test", {})
    raw = PreparedExchange(connection, "POST", "/test", b"{}", {"Content-Length": "2"})
    prepared = PreparedServeExchange(request, b"{}", raw)
    handle = RequestHandle("subject", 0, prepared, None)
    try:
        yield handle, peer, connection
    finally:
        handle.cancel()
        handle.wait_done(3)
        peer.close()
        connection.close()


@pytest.mark.parametrize("phase", ["headers", "body", "chunked_body"])
def test_local_shutdown_cancels_blocked_http_read(exchange, phase):
    handle, peer, _ = exchange
    handle.start()
    handle.wait_body_sent(2)
    if phase != "headers":
        payload = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        if phase == "chunked_body":
            peer.sendall(header + b"Transfer-Encoding: chunked\r\n\r\n")
            peer.sendall(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        else:
            peer.sendall(header + b"Connection: close\r\n\r\n" + payload)
        handle.wait_first_output(2)
    assert handle.cancel(only_if_unrequested=True) is not None
    handle.wait_done(2)
    record = handle.as_record()
    assert record["outcome"] == "cancelled"
    assert record["transport_cancelled"] and record["cancel_requested"]
    assert record["transport_error"] is None


@pytest.mark.parametrize("error", [
    TimeoutError("read timed out"),
    ConnectionResetError(errno.ECONNRESET, "peer reset"),
    http.client.BadStatusLine("malformed response"),
])
def test_cancellation_does_not_replace_concurrent_read_failure(exchange, monkeypatch, error):
    handle, _, connection = exchange
    reading = threading.Event()
    release = threading.Event()

    def failed_response():
        reading.set()
        assert release.wait(3)
        raise error

    monkeypatch.setattr(connection, "getresponse", failed_response)
    handle.start()
    try:
        assert reading.wait(2)
        handle.cancel()
    finally:
        release.set()
    handle.wait_done(2)
    record = handle.as_record()
    assert record["outcome"] == "transport_error"
    assert record["cancel_requested"]
    assert not record["transport_cancelled"]
    assert record["transport_error"] == f"{type(error).__name__}: {error}"


@pytest.mark.parametrize(("headers", "outcome"), [
    (b"HTTP/1.1 503 Unavailable\r\nContent-Type: text/event-stream\r\n", "rejected"),
    (b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n", "protocol_error"),
])
def test_cancellation_preserves_received_http_and_protocol_failure(
    exchange, monkeypatch, headers, outcome,
):
    handle, peer, connection = exchange
    received = threading.Event()
    original_response = connection.getresponse

    def response():
        value = original_response()
        received.set()
        return value

    monkeypatch.setattr(connection, "getresponse", response)
    handle.start()
    handle.wait_body_sent(2)
    peer.sendall(headers + b"Connection: close\r\n\r\n")
    assert received.wait(2)
    handle.cancel()
    handle.wait_done(2)
    record = handle.as_record()
    assert record["outcome"] == outcome
    assert record["cancel_requested"] and not record["transport_cancelled"]


def test_cancellation_preserves_malformed_chunk_size(exchange, monkeypatch):
    handle, peer, connection = exchange
    malformed = threading.Event()
    release = threading.Event()
    original_read = connection.response_class.read1

    def read(response, amount):
        try:
            return original_read(response, amount)
        except http.client.HTTPException:
            malformed.set()
            assert release.wait(3)
            raise

    monkeypatch.setattr(connection.response_class, "read1", read)
    handle.start()
    handle.wait_body_sent(2)
    peer.sendall(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n\r\ninvalid\r\n"
    )
    try:
        assert malformed.wait(2)
        handle.cancel()
    finally:
        release.set()
    handle.wait_done(2)
    record = handle.as_record()
    assert record["outcome"] == "transport_error"
    assert record["cancel_requested"] and not record["transport_cancelled"]
    assert "invalid chunk size" in record["transport_error"]
