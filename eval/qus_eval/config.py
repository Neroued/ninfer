from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when the user-facing evaluation configuration is invalid."""


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    return dict(value)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown field(s): {', '.join(unknown)}")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where} must be a positive integer")
    return value


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{where} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RequestConfig:
    timeout_seconds: float = 600.0
    retries: int = 5
    retry_interval_seconds: float = 10.0
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any, where: str) -> "RequestConfig":
        data = _mapping(raw or {}, where)
        _reject_unknown(
            data,
            {"timeout_seconds", "retries", "retry_interval_seconds", "headers"},
            where,
        )
        timeout = data.get("timeout_seconds", 600.0)
        interval = data.get("retry_interval_seconds", 10.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ConfigError(f"{where}.timeout_seconds must be positive")
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or interval < 0
        ):
            raise ConfigError(f"{where}.retry_interval_seconds must be non-negative")
        headers = _mapping(data.get("headers", {}), f"{where}.headers")
        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise ConfigError(f"{where}.headers must contain string keys and values")
        lowered = {k.lower() for k in headers}
        if "authorization" in lowered or "x-api-key" in lowered:
            raise ConfigError(
                f"{where}.headers must not contain credentials; use api_key_env"
            )
        return cls(
            timeout_seconds=float(timeout),
            retries=_nonnegative_int(data.get("retries", 5), f"{where}.retries"),
            retry_interval_seconds=float(interval),
            headers=headers,
        )


@dataclass(frozen=True)
class TargetConfig:
    name: str
    protocol: str
    base_url: str
    model: str
    api_key_env: str | None
    max_concurrency: int
    request: RequestConfig
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, raw: Any) -> "TargetConfig":
        where = f"targets.{name}"
        data = _mapping(raw, where)
        _reject_unknown(
            data,
            {
                "protocol",
                "base_url",
                "model",
                "api_key_env",
                "max_concurrency",
                "request",
                "provenance",
            },
            where,
        )
        for key in ("protocol", "base_url", "model"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ConfigError(f"{where}.{key} must be a non-empty string")
        if not data["base_url"].startswith(("http://", "https://")):
            raise ConfigError(f"{where}.base_url must use http:// or https://")
        key_env = data.get("api_key_env")
        if key_env is not None and (
            not isinstance(key_env, str) or not key_env.strip()
        ):
            raise ConfigError(
                f"{where}.api_key_env must be a non-empty environment variable name"
            )
        return cls(
            name=name,
            protocol=data["protocol"],
            base_url=data["base_url"].rstrip("/"),
            model=data["model"],
            api_key_env=key_env,
            max_concurrency=_positive_int(
                data.get("max_concurrency", 1), f"{where}.max_concurrency"
            ),
            request=RequestConfig.from_dict(
                data.get("request", {}), f"{where}.request"
            ),
            provenance=_mapping(data.get("provenance", {}), f"{where}.provenance"),
        )


@dataclass(frozen=True)
class JobConfig:
    id: str
    backend: str
    dataset: str
    target: str | None
    max_concurrency: int | None
    limit: int | float | None
    repeats: int
    generation: dict[str, Any]
    backend_args: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: Any, where: str) -> "JobConfig":
        data = _mapping(raw, where)
        _reject_unknown(
            data,
            {
                "id",
                "backend",
                "dataset",
                "target",
                "max_concurrency",
                "limit",
                "repeats",
                "generation",
                "backend_args",
            },
            where,
        )
        for key in ("id", "backend", "dataset"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ConfigError(f"{where}.{key} must be a non-empty string")
        target = data.get("target")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ConfigError(f"{where}.target must be a non-empty string when present")
        concurrency = data.get("max_concurrency")
        if concurrency is not None:
            concurrency = _positive_int(concurrency, f"{where}.max_concurrency")
        limit = data.get("limit")
        if limit is not None:
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or limit <= 0
            ):
                raise ConfigError(
                    f"{where}.limit must be a positive integer or fraction"
                )
            if isinstance(limit, float) and limit > 1:
                raise ConfigError(f"{where}.limit as a float must be in (0, 1]")
        generation = _mapping(data.get("generation", {}), f"{where}.generation")
        portable = {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "seed",
            "stream",
            "frequency_penalty",
            "presence_penalty",
            "stop_seqs",
            "extra_body",
        }
        _reject_unknown(generation, portable, f"{where}.generation")
        return cls(
            id=data["id"],
            backend=data["backend"],
            dataset=data["dataset"],
            target=target,
            max_concurrency=concurrency,
            limit=limit,
            repeats=_positive_int(data.get("repeats", 1), f"{where}.repeats"),
            generation=generation,
            backend_args=_mapping(
                data.get("backend_args", {}), f"{where}.backend_args"
            ),
        )


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    jobs: tuple[JobConfig, ...]

    @classmethod
    def from_dict(cls, name: str, raw: Any) -> "SuiteConfig":
        where = f"suites.{name}"
        data = _mapping(raw, where)
        _reject_unknown(data, {"jobs"}, where)
        jobs_raw = data.get("jobs")
        if not isinstance(jobs_raw, list) or not jobs_raw:
            raise ConfigError(f"{where}.jobs must be a non-empty list")
        jobs = tuple(
            JobConfig.from_dict(item, f"{where}.jobs[{i}]")
            for i, item in enumerate(jobs_raw)
        )
        ids = [job.id for job in jobs]
        if len(ids) != len(set(ids)):
            raise ConfigError(f"{where}.jobs contains duplicate job ids")
        return cls(name=name, jobs=jobs)


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = True
    refresh_seconds: float = 1.0
    heartbeat_seconds: float = 30.0

    @classmethod
    def from_dict(cls, raw: Any, where: str) -> "ProgressConfig":
        data = _mapping(raw or {}, where)
        _reject_unknown(
            data, {"enabled", "refresh_seconds", "heartbeat_seconds"}, where
        )
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"{where}.enabled must be boolean")
        refresh = data.get("refresh_seconds", 1.0)
        heartbeat = data.get("heartbeat_seconds", 30.0)
        for key, value in (
            ("refresh_seconds", refresh),
            ("heartbeat_seconds", heartbeat),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ConfigError(f"{where}.{key} must be positive")
        return cls(
            enabled=enabled,
            refresh_seconds=float(refresh),
            heartbeat_seconds=float(heartbeat),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    max_parallel_jobs: int = 1
    runs_dir: str = "eval/runs"
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    sample_retention: str = "all"

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeConfig":
        where = "runtime"
        data = _mapping(raw or {}, where)
        _reject_unknown(
            data, {"max_parallel_jobs", "runs_dir", "progress", "samples"}, where
        )
        runs_dir = data.get("runs_dir", "eval/runs")
        if not isinstance(runs_dir, str) or not runs_dir.strip():
            raise ConfigError("runtime.runs_dir must be a non-empty string")
        samples = _mapping(data.get("samples", {}), "runtime.samples")
        _reject_unknown(samples, {"retention"}, "runtime.samples")
        retention = samples.get("retention", "all")
        if retention not in {"all", "errors", "none"}:
            raise ConfigError("runtime.samples.retention must be all, errors, or none")
        return cls(
            max_parallel_jobs=_positive_int(
                data.get("max_parallel_jobs", 1), "runtime.max_parallel_jobs"
            ),
            runs_dir=runs_dir,
            progress=ProgressConfig.from_dict(
                data.get("progress", {}), "runtime.progress"
            ),
            sample_retention=retention,
        )


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    targets: dict[str, TargetConfig]
    suites: dict[str, SuiteConfig]
    runtime: RuntimeConfig
    source_path: Path

    def suite(self, name: str) -> SuiteConfig:
        try:
            return self.suites[name]
        except KeyError as exc:
            raise ConfigError(f"unknown suite: {name}") from exc

    def target(self, name: str | None) -> TargetConfig | None:
        if name is None:
            return None
        try:
            return self.targets[name]
        except KeyError as exc:
            raise ConfigError(f"unknown target: {name}") from exc

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        data["targets"] = {
            name: _target_to_dict(target) for name, target in self.targets.items()
        }
        data["suites"] = {
            name: {"jobs": [_job_to_dict(job) for job in suite.jobs]}
            for name, suite in self.suites.items()
        }
        data["runtime"] = {
            "max_parallel_jobs": self.runtime.max_parallel_jobs,
            "runs_dir": self.runtime.runs_dir,
            "progress": asdict(self.runtime.progress),
            "samples": {"retention": self.runtime.sample_retention},
        }
        return data

    def fingerprint(self, suite_name: str) -> str:
        suite = self.suite(suite_name)
        used_targets = {job.target for job in suite.jobs if job.target is not None}
        relevant = {
            "schema_version": self.schema_version,
            "targets": {
                name: self.to_dict()["targets"][name] for name in sorted(used_targets)
            },
            "suite": self.to_dict()["suites"][suite_name],
            "runtime": self.to_dict()["runtime"],
        }
        payload = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _target_to_dict(target: TargetConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "protocol": target.protocol,
        "base_url": target.base_url,
        "model": target.model,
        "max_concurrency": target.max_concurrency,
        "request": asdict(target.request),
    }
    if target.api_key_env:
        out["api_key_env"] = target.api_key_env
    if target.provenance:
        out["provenance"] = target.provenance
    return out


def _job_to_dict(job: JobConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": job.id,
        "backend": job.backend,
        "dataset": job.dataset,
        "repeats": job.repeats,
        "generation": job.generation,
        "backend_args": job.backend_args,
    }
    if job.target is not None:
        out["target"] = job.target
    if job.max_concurrency is not None:
        out["max_concurrency"] = job.max_concurrency
    if job.limit is not None:
        out["limit"] = job.limit
    return out


def load_config(path: str | Path) -> AppConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {source}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    data = _mapping(raw, "configuration")
    _reject_unknown(
        data, {"schema_version", "targets", "suites", "runtime"}, "configuration"
    )
    version = data.get("schema_version")
    if version != 1:
        raise ConfigError(f"unsupported schema_version: {version!r}; expected 1")
    targets_raw = _mapping(data.get("targets", {}), "targets")
    suites_raw = _mapping(data.get("suites", {}), "suites")
    if not suites_raw:
        raise ConfigError("suites must not be empty")
    targets = {
        name: TargetConfig.from_dict(name, value) for name, value in targets_raw.items()
    }
    suites = {
        name: SuiteConfig.from_dict(name, value) for name, value in suites_raw.items()
    }
    config = AppConfig(
        schema_version=1,
        targets=targets,
        suites=suites,
        runtime=RuntimeConfig.from_dict(data.get("runtime", {})),
        source_path=source,
    )
    for suite in config.suites.values():
        for job in suite.jobs:
            if job.target is not None and job.target not in targets:
                raise ConfigError(
                    f"suite {suite.name} job {job.id} references unknown target {job.target}"
                )
    return config


def dump_config(config: AppConfig, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
