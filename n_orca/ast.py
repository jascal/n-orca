"""AST types for N-Orca architecture documents.

Every parsed `.n.orca.md` file becomes a list of `Architecture` objects. The
verifier, compilers and CLI all consume this AST — never the raw Markdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hyperparameter:
    name: str
    type: str            # "int" | "float" | "bool" | "string"
    default: Any | None  # parsed literal of the declared type, or None


@dataclass
class Tensor:
    name: str
    shape: tuple[str, ...]   # each dim is a raw token: "B", "S", "d_model", "3"
    dtype: str               # "float32" | "float16" | "bfloat16" | "int32" | "int64" | "bool"


@dataclass
class OpCall:
    name: str
    args: list[str]          # raw arg tokens, preserved verbatim


@dataclass
class Layer:
    name: str
    is_input: bool = False
    is_output: bool = False
    description: str | None = None
    op: OpCall | None = None
    declared_shape: tuple[str, ...] | None = None
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class FlowEdge:
    source: str
    target: str
    tensor: str


@dataclass
class Invariant:
    """A declarative bound from `## invariants`."""
    kind: str        # "param_count" | "flops" | "depth" | "output_shape"
    op: str          # "<=" | ">=" | "==" | "="
    value: Any       # int, float, or tuple[str, ...] for output_shape


@dataclass
class Architecture:
    name: str
    description: str | None = None
    hyperparameters: list[Hyperparameter] = field(default_factory=list)
    tensors: list[Tensor] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    flow: list[FlowEdge] = field(default_factory=list)
    invariants: list[Invariant] = field(default_factory=list)
    verification_rules: list[str] = field(default_factory=list)
    #: Provenance / runtime hints (e.g. `model_type` stamped by HF `convert`).
    #: Round-trips through Markdown as a `## runtime` section. Used by the
    #: runtime-capability backend; never affects topology/shape verification.
    metadata: dict[str, Any] = field(default_factory=dict)

    def layer(self, name: str) -> Layer | None:
        for ly in self.layers:
            if ly.name == name:
                return ly
        return None

    def hyperparameter(self, name: str) -> Hyperparameter | None:
        for hp in self.hyperparameters:
            if hp.name == name:
                return hp
        return None

    def tensor(self, name: str) -> Tensor | None:
        for t in self.tensors:
            if t.name == name:
                return t
        return None

    def inputs(self) -> list[Layer]:
        return [ly for ly in self.layers if ly.is_input]

    def outputs(self) -> list[Layer]:
        return [ly for ly in self.layers if ly.is_output]
