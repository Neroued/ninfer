"""Ordered recipe assignment, explicit sharing, and finite parent packing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from math import prod
from typing import Sequence

from tools.artifact.layouts import encoded_size
from tools.artifact.formats import DirectFormat, Fp8RowFormat, Nvfp4Format, get_format
from tools.artifact.schema import ACTIVATION_POLICIES, TensorSpec
from .methods import (
    AuxiliaryValue,
    MethodInput,
    PrepareRequest,
    PreparedMethod,
    METHODS,
    cast_direct,
    grouped_absmax,
    fp8_row_maxabs,
    import_encoded,
)
from .model import Model
from .sources.logical import LogicalSource, select_rows


@dataclass(frozen=True, slots=True)
class Selection:
    begin: int
    end: int
    format: str
    layout: str | None
    method: object
    source: LogicalSource
    parameters: dict


@dataclass(frozen=True, slots=True)
class WeightJob:
    spec: TensorSpec
    prepared: PreparedMethod
    parameters: tuple[str, ...]
    sources: tuple[str, ...]
    method_name: str
    method_parameters: dict


@dataclass(frozen=True, slots=True)
class PreparedRecipe:
    weights: tuple[WeightJob, ...]
    bindings: dict
    uses: tuple[dict, ...]
    auxiliaries: tuple[tuple[TensorSpec, bytes], ...]


def default_layout(format: str) -> str:
    kind = get_format(format)
    if isinstance(kind, DirectFormat):
        return "contiguous_le_v1"
    if isinstance(kind, Fp8RowFormat):
        return "row_scale_v1"
    if isinstance(kind, Nvfp4Format):
        return "block_scale_k16_m128x4_v1"
    return "row_split_k128_v1"


def _slice_source(source: LogicalSource, begin: int, end: int) -> LogicalSource:
    if begin == 0 and end == prod(source.shape):
        return source
    width = prod(source.shape[1:]) if source.shape else 1
    if begin % width or end % width or not source.shape:
        raise ValueError("recipe subregions must contain complete leading-axis rows")
    if len(source.shape) == 2:
        return select_rows(source, ((begin // width, end // width),))
    shape = ((end - begin) // width, *source.shape[1:])
    return LogicalSource(
        shape, source.label, lambda a, b: source.values(begin + a, begin + b)
    )


class Recipe:
    """Configuration remains ordinary Python; prepared output contains only data and jobs."""

    def __init__(self, model: Model):
        self.model = model
        self.selections: dict[str, list[Selection]] = {}
        self.policies = {}
        self.auxiliary_overrides = {}
        self.aliases: dict[str, str] = {}
        self.separate_parameters: set[str] = set()
        self.explicit_groups: list[tuple[tuple[str, ...], tuple[int, ...] | None]] = []
        for parameter in model.parameters.values():
            self.add_parameter(parameter.name)

    def add_parameter(self, name: str) -> None:
        parameter = self.model.parameters[name]
        if name in self.selections:
            raise ValueError(f"recipe already contains {name}")
        self.selections[name] = [
            Selection(
                0,
                prod(parameter.shape),
                parameter.direct_format,
                None,
                cast_direct,
                parameter.source,
                {},
            )
        ]
        for input_name in parameter.inputs:
            self.policies[(name, input_name)] = "A16Only"

    def match(self, selectors: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(selectors, str):
            selectors = (selectors,)
        if not selectors:
            raise ValueError("recipe selectors must not be empty")
        names = []
        seen = set()
        for selector in selectors:
            matched = (
                [selector]
                if selector in self.selections
                else [name for name in self.selections if fnmatchcase(name, selector)]
            )
            if not matched:
                raise ValueError(
                    f"recipe selector matched no logical parameter: {selector!r}"
                )
            for name in matched:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return tuple(names)

    def assign(
        self,
        selectors,
        *,
        format=None,
        layout=None,
        method=None,
        source=None,
        parameters=None,
        activation_policy=None,
        rows=None,
    ) -> None:
        if isinstance(method, str):
            try:
                method = METHODS[method]
            except KeyError as error:
                raise ValueError(f"unknown conversion method {method!r}") from error
        if method is not None and not callable(method):
            raise TypeError("method must be a conversion function")
        for name in self.match(selectors):
            parameter = self.model.parameters[name]
            if source is not None and tuple(source.shape) != parameter.shape:
                raise ValueError(
                    f"{name}: override source shape differs from logical parameter"
                )
            size = prod(parameter.shape)
            begin, end = 0, size
            if rows is not None:
                if len(rows) != 2 or not parameter.shape:
                    raise ValueError(
                        f"{name}: rows requires two bounds and a nonscalar parameter"
                    )
                first, last = rows
                if (
                    type(first) is not int
                    or type(last) is not int
                    or not 0 <= first < last <= parameter.shape[0]
                ):
                    raise ValueError(f"{name}: row range {rows} is invalid")
                width = prod(parameter.shape[1:])
                begin, end = first * width, last * width
            physical = any(
                value is not None
                for value in (format, layout, method, source, parameters)
            )
            if physical:
                self.aliases.pop(name, None)
                updated = []
                for previous in self.selections[name]:
                    low, high = max(begin, previous.begin), min(end, previous.end)
                    if low >= high:
                        updated.append(previous)
                        continue
                    if previous.begin < low:
                        updated.append(replace(previous, end=low))
                    changes = {"begin": low, "end": high}
                    for key, value in (
                        ("format", format),
                        ("layout", layout),
                        ("method", method),
                        ("source", source),
                        ("parameters", parameters),
                    ):
                        if value is not None:
                            changes[key] = (
                                None
                                if key == "layout" and value == "auto"
                                else deepcopy(value) if key == "parameters" else value
                            )
                    updated.append(replace(previous, **changes))
                    if high < previous.end:
                        updated.append(replace(previous, begin=high))
                self.selections[name] = updated
            if activation_policy is not None:
                for input_name in parameter.inputs:
                    self.use(name, input_name, activation_policy=activation_policy)

    def use(
        self,
        parameter: str,
        input_name: str,
        *,
        activation_policy=None,
        auxiliaries=None,
    ):
        key = (parameter, input_name)
        if key not in self.policies:
            raise ValueError(f"{parameter}@{input_name}: unknown mathematical use")
        if activation_policy is not None:
            if activation_policy not in ACTIVATION_POLICIES:
                raise ValueError(f"unknown activation policy {activation_policy!r}")
            self.policies[key] = activation_policy
        if auxiliaries is not None:
            for role, value in auxiliaries.items():
                if not isinstance(value, AuxiliaryValue):
                    if role != "activation_input_divisor":
                        raise TypeError("provide an AuxiliaryValue for this role")
                    value = AuxiliaryValue.activation_divisor(value)
                self.auxiliary_overrides[(*key, role)] = value

    def separate(self, selectors) -> None:
        self.separate_parameters.update(self.match(selectors))

    def group(self, selectors, *, shape=None) -> None:
        names = self.match(selectors)
        self.explicit_groups.append((names, None if shape is None else tuple(shape)))

    def share(self, parameter: str, target: str) -> None:
        if parameter not in self.selections or target not in self.selections:
            raise ValueError("shared parameters must exist")
        if (
            self.model.parameters[parameter].shape
            != self.model.parameters[target].shape
        ):
            raise ValueError("shared parameters must have equal logical shapes")
        current = target
        visited = {parameter}
        while current in self.aliases:
            if current in visited:
                raise ValueError("cyclic weight sharing")
            visited.add(current)
            current = self.aliases[current]
        if current in visited:
            raise ValueError("cyclic weight sharing")
        self.aliases[parameter] = target
        self.selections[parameter] = list(self.selections[target])

    def prepare(self, *, device="cuda", rows_per_chunk=512) -> PreparedRecipe:
        weights = []
        fragments = {name: [] for name in self.selections}
        auxiliary_values = dict(self.auxiliary_overrides)
        used = set()

        def compatible(items):
            first = items[0][1]
            return all(
                (value.format, value.layout, value.method, value.parameters)
                == (first.format, first.layout, first.method, first.parameters)
                for _, value in items
            )

        def parent_shape(items, chosen):
            sources = [
                _slice_source(value.source, value.begin, value.end)
                for _, value in items
            ]
            size = sum(prod(source.shape) for source in sources)
            if chosen is not None:
                if prod(chosen) != size:
                    raise ValueError("explicit parent shape changes its element count")
                return chosen, sources
            first = sources[0].shape
            direct = isinstance(get_format(items[0][1].format), DirectFormat)
            if len(sources) == 1:
                if direct or len(first) == 2:
                    return first, sources
                return (prod(first[:-1]), first[-1]) if first else (1, 1), sources
            if all(len(source.shape) == 1 for source in sources):
                return (size,) if direct else (1, size), sources
            if all(
                len(source.shape) == 2 and source.shape[1] == first[1]
                for source in sources
            ):
                return (sum(source.shape[0] for source in sources), first[1]), sources
            raise ValueError(
                "grouped inputs need compatible rows or an explicit parent shape"
            )

        def parent_divisors(sources, selection) -> int:
            """How many NVFP4 divisors this parent stores: one per source, or one for all.

            A parent whose sources were quantised apart has to keep each source's divisor, and the
            stored plane addresses them by an equal share of its rows. That is a partition only if
            the sources have equal row counts, so a parent that needs several divisors and cannot be
            divided equally is refused here, where its shape is chosen, rather than part way through
            writing it.
            """
            if selection.format != "nvfp4" or len(sources) < 2:
                return 1
            words = [
                source.weight_divisor() if source.weight_divisor else None
                for source in sources
            ]
            # A stack is only chosen when the sources are known to disagree. A divisor no source
            # states at plan time is unknown, not different, and answering the source count there
            # would mark a plane a stack that no dense route can execute.
            if None in words or len(set(words)) == 1:
                return 1
            if any(len(source.shape) != 2 for source in sources) or (
                len({source.shape[0] for source in sources}) != 1
            ):
                raise ValueError(
                    "NVFP4 sources quantised against different divisors must be complete "
                    "matrices with equal row counts"
                )
            return len(sources)

        def emit(items, chosen=None):
            if len({self.model.parameters[name].residency for name, _ in items}) > 1:
                raise ValueError(
                    "grouping crosses independently selected private weights"
                )
            if not compatible(items):
                raise ValueError(
                    "explicit grouping has incompatible formats, layouts or methods"
                )
            dims, sources = parent_shape(items, chosen)
            selection = items[0][1]
            layout = selection.layout or default_layout(selection.format)
            divisors = parent_divisors(sources, selection)
            encoded_size(layout, selection.format, dims, divisors)
            spec = TensorSpec(
                f"weight/{len(weights):06d}", dims, selection.format, layout, divisors
            )
            inputs = tuple(
                MethodInput(
                    name,
                    source,
                    tuple(
                        (name, value) for value in self.model.parameters[name].inputs
                    ),
                )
                for (name, _), source in zip(items, sources)
            )
            request = PrepareRequest(
                spec,
                inputs,
                self.policies,
                selection.parameters,
                device,
                rows_per_chunk,
                self.auxiliary_overrides,
            )
            try:
                prepared = selection.method(request)
            except Exception as error:
                raise ValueError(
                    f"{','.join(name for name, _ in items)}: {error}"
                ) from error
            if not isinstance(prepared, PreparedMethod):
                raise TypeError("conversion method must return request.job(...)")
            for key, value in prepared.auxiliaries.items():
                if key in auxiliary_values and auxiliary_values[key] != value:
                    raise ValueError(
                        f"{key}: conflicting auxiliary values across physical parts"
                    )
                auxiliary_values[key] = value
            weights.append(
                WeightJob(
                    spec,
                    prepared,
                    tuple(name for name, _ in items),
                    tuple(source.label for source in sources),
                    getattr(
                        selection.method, "__name__", type(selection.method).__name__
                    ),
                    dict(selection.parameters),
                )
            )
            cursor = 0
            for name, value in items:
                count = value.end - value.begin
                fragments[name].append((value.begin, spec.id, cursor, cursor + count))
                cursor += count

        for names, chosen in self.explicit_groups:
            if any(name in used or name in self.aliases for name in names):
                raise ValueError("explicit groups overlap or include a shared alias")
            if any(len(self.selections[name]) != 1 for name in names):
                raise ValueError(
                    "explicit grouping requires unsplit logical selections"
                )
            emit([(name, self.selections[name][0]) for name in names], chosen)
            used.update(names)
        standard = (cast_direct, grouped_absmax, fp8_row_maxabs, import_encoded)
        for names in self.model.packing_groups:
            if any(
                name in used or name in self.aliases or name in self.separate_parameters
                for name in names
            ):
                continue
            if any(len(self.selections[name]) != 1 for name in names):
                continue
            items = [(name, self.selections[name][0]) for name in names]
            if not compatible(items) or items[0][1].method not in standard:
                continue
            if items[0][1].format == "nvfp4" and items[0][1].method is import_encoded:
                divisors = [
                    (
                        value.source.weight_divisor()
                        if value.source.weight_divisor is not None
                        else None
                    )
                    for _, value in items
                ]
                if None in divisors or len(set(divisors)) != 1:
                    continue
            emit(items)
            used.update(names)
        for name, choices in self.selections.items():
            if name not in used and name not in self.aliases:
                for choice in choices:
                    emit([(name, choice)])

        specs = {weight.spec.id: weight.spec for weight in weights}
        bindings = {}
        for name, parts in fragments.items():
            if name in self.aliases:
                continue
            parts.sort()
            count = 0
            for begin, _, first, last in parts:
                if begin != count:
                    raise ValueError(f"{name}: incomplete logical coverage")
                count += last - first
            if count != prod(self.model.parameters[name].shape):
                raise ValueError(f"{name}: logical coverage size is incorrect")
            _, parent, first, last = parts[0]
            if (
                len(parts) == 1
                and first == 0
                and specs[parent].shape == self.model.parameters[name].shape
                and last == prod(specs[parent].shape)
            ):
                bindings[name] = {"object": parent}
            else:
                bindings[name] = {
                    "parts": [
                        {"object": parent, "range": [first, last]}
                        for _, parent, first, last in parts
                    ]
                }
        for name in self.aliases:
            target = self.aliases[name]
            while target in self.aliases:
                target = self.aliases[target]
            bindings[name] = deepcopy(bindings[target])

        uses = []
        auxiliary_outputs = []
        for (name, input_name), policy in self.policies.items():
            # Explicitly shared weights retain independent Use and calibration records.
            key = (name, input_name, "activation_input_divisor")
            target = name
            while target in self.aliases:
                target = self.aliases[target]
            if (
                name in self.aliases
                and policy == "AllowA4"
                and any(s.format == "nvfp4" for s in self.selections[target])
                and key not in auxiliary_values
            ):
                raise ValueError(f"{name}: provide its independent activation divisor")
            use = {"parameter": name, "input": input_name, "activation_policy": policy}
            referenced = {}
            for (parameter, source_input, role), value in auxiliary_values.items():
                if (parameter, source_input) != (name, input_name):
                    continue
                if role == "activation_input_divisor":
                    if value.format != "fp32" or value.shape != ():
                        raise ValueError(
                            f"{name}: activation divisor must be an FP32 scalar"
                        )
                    AuxiliaryValue.activation_divisor(value.data)
                layout = default_layout(value.format)
                if len(value.data) != encoded_size(layout, value.format, value.shape):
                    raise ValueError(f"{name}/{role}: auxiliary data length is invalid")
                spec = TensorSpec(
                    f"auxiliary/{len(auxiliary_outputs):06d}",
                    value.shape,
                    value.format,
                    layout,
                )
                auxiliary_outputs.append((spec, value.data))
                referenced[role] = {"object": spec.id}
            if referenced:
                use["auxiliaries"] = referenced
            uses.append(use)
        known_uses = set(self.policies)
        if any((key[0], key[1]) not in known_uses for key in auxiliary_values):
            raise ValueError("method returned an auxiliary for an unknown Use")
        return PreparedRecipe(
            tuple(weights), bindings, tuple(uses), tuple(auxiliary_outputs)
        )
