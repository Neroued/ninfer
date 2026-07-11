from __future__ import annotations

import os
from dataclasses import dataclass

from .config import ConfigError, TargetConfig


@dataclass(frozen=True)
class ResolvedTarget:
    config: TargetConfig
    api_key: str | None


def resolve_target(target: TargetConfig | None) -> ResolvedTarget | None:
    if target is None:
        return None
    api_key = None
    if target.api_key_env:
        api_key = os.environ.get(target.api_key_env)
        if not api_key:
            raise ConfigError(
                f"target {target.name} requires environment variable {target.api_key_env}, but it is unset"
            )
    return ResolvedTarget(config=target, api_key=api_key)


def redact_text(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted
