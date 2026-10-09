"""No-network oracles for replay accounting and SystemOne response validation."""
import asyncio
import copy
import json
import threading
import unittest
from unittest.mock import patch

from decision_live import CHAT, chat_request, replay, replay_http, systemone_cases, validate_systemone


def response_for(request):
    answers = {}
    for name, question in request["questions"].items():
        kind = question["type"]
        if kind == "noul":
            answers[name] = {"type": "noul", "noul": .25}
        else:
            criteria = question["criteria"]
            labels = list(criteria) if kind == "choice" else [str(i) for i in range(len(criteria))]
            p = {k: float(i == 0) for i, k in enumerate(labels)}
            answers[name] = {"type": kind, "probabilities": p, "confidence": 1}
            if kind == "choice":
                answers[name]["choice"] = labels[0]
            else:
                answers[name].update(score=0, legend=dict(zip(labels, criteria)))
    return {"model": "synthetic", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}}


class ReplayTest(unittest.IsolatedAsyncioTestCase):
    async def test_backlog_never_drops_and_every_event_is_independent(self):
        calls, active, peak = [], 0, 0

        async def invoke(event):
            nonlocal active, peak
            calls.append(event)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.001)
            active -= 1
            return {"answer": event}

        # Exceeds the old 64-event limit; all duplicate inputs still get a call.
        events = [i % 4 for i in range(160)]
        stage = await replay(events, 0, 2, invoke)
        self.assertEqual(calls, events)
        self.assertEqual(stage["completed"], 160)
        self.assertEqual(stage["dropped"], 0)
        self.assertGreater(stage["peak_pending"], 64)
        self.assertEqual(peak, 2)
        self.assertEqual([r["event"] for r in stage["records"]], list(range(160)))
        self.assertTrue(all(r["e2e_ms"] >= r["request_ms"] for r in stage["records"]))

    async def test_errors_are_counted_without_losing_later_events(self):
        async def invoke(event):
            if event == 2:
                raise TimeoutError("explicit request timeout")
            return {"answer": event}
        stage = await replay(list(range(6)), 1000, 2, invoke)
        self.assertEqual((stage["received"], stage["completed"], stage["errors"], stage["dropped"]), (6, 5, 1, 0))
        self.assertIn("timeout", stage["records"][2]["error"])
        self.assertTrue(all(r["queue_ms"] >= 0 for r in stage["records"]))

    async def test_http_workers_honor_requested_concurrency(self):
        concurrency = 16
        barrier = threading.Barrier(concurrency, timeout=10)
        lock = threading.Lock()
        active = peak = 0

        def classify_stub(client, route, fixture, event):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait()
                return {"answer": event}
            finally:
                with lock:
                    active -= 1

        with patch("decision_live.classify", classify_stub):
            stage = await replay_http(None, "decision", {}, list(range(concurrency)), 0, concurrency)
        self.assertEqual(stage["completed"], concurrency)
        self.assertEqual(stage["errors"], 0)
        self.assertEqual(peak, concurrency)


class SystemOneValidatorTest(unittest.TestCase):
    def setUp(self):
        self.request = next(systemone_cases())[1]
        self.response = response_for(self.request)

    def test_all_contract_cases(self):
        for name, request in systemone_cases():
            with self.subTest(name=name):
                validate_systemone(request, response_for(request))

    def test_rejects_malformed_answers(self):
        mutations = [
            lambda r: r["answers"].pop("urgent"),
            lambda r: r["answers"]["urgent"].update(noul=True),
            lambda r: r["answers"]["urgent"].update(noul=float("nan")),
            lambda r: r["answers"]["route"].update(choice="other"),
            lambda r: r["answers"]["route"]["probabilities"].update(other=.5),
            lambda r: r["answers"]["route"].update(confidence=0),
            lambda r: r["answers"]["severity"].update(score=1),
            lambda r: r["answers"]["severity"]["legend"].update({"0": "Wrong"}),
            lambda r: r["usage"].update(output_tokens=1),
        ]
        for mutate in mutations:
            response = copy.deepcopy(self.response)
            mutate(response)
            with self.assertRaises(ValueError):
                validate_systemone(self.request, response)

    def test_fractional_score_and_confidence(self):
        answer = self.response["answers"]["severity"]
        answer.update(probabilities={"0": .1, "1": .8, "2": .1}, score=1, confidence=.7)
        validate_systemone(self.request, self.response)
        answer.update(probabilities={"0": 0, "1": .6, "2": .4}, score=1.4, confidence=.4)
        validate_systemone(self.request, self.response)

    def test_fixture_and_independent_requests(self):
        fixture = json.loads(CHAT.read_text(encoding="utf-8"))
        self.assertEqual(len(fixture["messages"]), 64)
        self.assertEqual(len({m["id"] for m in fixture["messages"]}), 64)
        previous = None
        for message in fixture["messages"]:
            self.assertIn(message["expected"], fixture["criteria"])
            d = chat_request("decision", fixture, message["text"])
            s = chat_request("systemone", fixture, message["text"])
            self.assertEqual(len(d["contexts"]), 1)
            self.assertEqual(len(s["questions"]), 1)
            self.assertEqual(s["state"], {"message": message["text"]})
            if previous is not None:
                self.assertEqual(s["questions"], previous)
            previous = s["questions"]


if __name__ == "__main__":
    unittest.main()
