"""Adapter base class + registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from n_orca.ast import Architecture


class HfAdapter(ABC):
    """An adapter knows how to turn an HF `config.json` into an `Architecture`.

    Subclasses declare a `model_types` tuple of strings that match either the
    config's `model_type` field or any entry in `architectures`. The first
    registered adapter whose `matches()` returns True wins.
    """

    #: Strings to match against `config["model_type"]` or `config["architectures"]`.
    model_types: tuple[str, ...] = ()

    def matches(self, config: dict[str, Any]) -> bool:
        mt = (config.get("model_type") or "").lower()
        archs = [a.lower() for a in (config.get("architectures") or [])]
        keys = {k.lower() for k in self.model_types}
        if mt and mt in keys:
            return True
        return any(any(k in a for k in keys) for a in archs)

    @abstractmethod
    def build(self, config: dict[str, Any], *, name: str | None = None) -> Architecture:
        """Build an `Architecture` AST from `config`.

        `name` overrides the architecture name; defaults to a derived value
        (model_type + size hint).
        """
        raise NotImplementedError


_REGISTRY: list[HfAdapter] = []


def register(adapter: HfAdapter) -> HfAdapter:
    """Append an adapter to the registry. Order matters — first match wins."""
    _REGISTRY.append(adapter)
    return adapter


def get_adapter(config: dict[str, Any]) -> HfAdapter | None:
    """Return the first registered adapter that matches `config`, or None."""
    for adapter in _REGISTRY:
        if adapter.matches(config):
            return adapter
    return None


def list_adapters() -> list[HfAdapter]:
    """Read-only view of the registered adapters."""
    return list(_REGISTRY)
