"""Shared observations extracted from production Serve logs."""

from __future__ import annotations

from typing import Any


def startup_resource_observation(event: dict[str, Any] | None) -> dict[str, Any]:
    """Preserve rank resources and totals, with explicit availability for incomplete logs."""
    if not event:
        return {"status": "unavailable", "reason": "server_start_missing"}
    resources = event.get("resources")
    if resources is None:
        return {"status": "unavailable", "reason": "startup_resources_missing"}
    invalid = {"status": "unavailable", "reason": "startup_resources_invalid"}
    if not isinstance(resources, dict):
        return invalid
    ranks = resources.get("ranks")
    totals = resources.get("total_allocations")
    if not isinstance(ranks, list) or not ranks or not isinstance(totals, dict) or not totals:
        return invalid
    seen = set()
    for rank in ranks:
        if (
            not isinstance(rank, dict)
            or type(rank.get("rank")) is not int
            or rank["rank"] < 0
            or rank["rank"] in seen
            or not isinstance(rank.get("resources"), dict)
            or not rank["resources"]
        ):
            return invalid
        seen.add(rank["rank"])
    return {"status": "available", "ranks": ranks, "total_allocations": totals}
