"""Logical parameters, components, and candidate packing groups used by recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .sources.logical import LogicalSource
from .sources.safetensors import SafetensorsSource

SourceFactory = Callable[[SafetensorsSource, str | None], LogicalSource]


@dataclass(frozen=True, slots=True)
class Parameter:
    name: str
    shape: tuple[int, ...]
    source: LogicalSource
    source_factory: SourceFactory | None = None
    inputs: tuple[str, ...] = ()
    direct_format: str = "bf16"
    residency: str = "text"

    @property
    def projection(self) -> bool:
        return bool(self.inputs)


@dataclass(slots=True)
class Model:
    components: dict
    parameters: dict[str, Parameter] = field(default_factory=dict)
    resources: dict[str, bytes] = field(default_factory=dict)
    packing_groups: list[tuple[str, ...]] = field(default_factory=list)
    token_count: int = 0
    special_token_ids: tuple[int, ...] = ()

    @property
    def config(self) -> dict:
        return self.components["text"]["config"]

    def add(self, parameter: Parameter) -> None:
        if parameter.name in self.parameters:
            raise ValueError(f"duplicate logical parameter {parameter.name}")
        if parameter.source.shape != parameter.shape:
            raise ValueError(
                f"{parameter.name}: source shape differs from logical parameter"
            )
        self.parameters[parameter.name] = parameter

    def source(
        self, parameter: str, store: SafetensorsSource, format: str | None = None
    ) -> LogicalSource:
        item = self.parameters[parameter]
        if item.source_factory is None:
            raise ValueError(f"{parameter}: provide an explicit logical source")
        return item.source_factory(store, format)
