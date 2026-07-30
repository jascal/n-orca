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
class Group:
    """A named set of layers, declared in `## groups`.

    A group is nothing more than a set of layer names: it is not a subgraph,
    not a module, and not a namespace. It exists so `## properties` can make
    claims at subgraph granularity. Member order is the document's order and
    is used when a diagnostic has to enumerate members.
    """
    name: str
    layers: list[str] = field(default_factory=list)


@dataclass
class Property:
    """A checkable topology claim, declared as one bullet in `## properties`.

    One flat record covers all six kinds; `kind` is the discriminator and the
    Stage 6 checker dispatches on it. Operands not used by a kind stay `None`,
    which is also how the checker detects a malformed bullet.

    | kind             | operands used                     |
    |------------------|-----------------------------------|
    | `reachable`      | source, target                    |
    | `unreachable`    | source, target                    |
    | `passes_through` | through, source, target           |
    | `within`         | source, target, op, bound         |
    | `param_fraction` | subject, op, bound                |
    | `out_dim`        | subject, op, bound                |

    `bound` stays an unresolved string: `out_dim: nm_bus <= n_affect` needs the
    hyperparameter substitution map, which is a verifier concern, not a parser
    one.
    """
    kind: str
    source: str | None = None    # the `from` operand (A)
    target: str | None = None    # the subject of reachability (B)
    through: str | None = None   # the `for A -> B` cut set (C)
    subject: str | None = None   # the bounded operand (G / L)
    op: str = "<="               # comparison operator
    bound: str | None = None     # right-hand side, unresolved
    raw: str = ""                # verbatim bullet, quoted back in messages


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
    groups: list[Group] = field(default_factory=list)
    properties: list[Property] = field(default_factory=list)

    def layer(self, name: str) -> Layer | None:
        for ly in self.layers:
            if ly.name == name:
                return ly
        return None

    def group(self, name: str) -> Group | None:
        for g in self.groups:
            if g.name == name:
                return g
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
