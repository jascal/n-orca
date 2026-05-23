"""Per-architecture adapters that map an HF `config.json` to an n-orca AST.

Following the sae-forge pattern: first-match-wins dispatch on the config's
declared model type / architecture name. Each adapter is a small,
declarative class.
"""
from n_orca.hf.adapters.base import HfAdapter, register, get_adapter, list_adapters
from n_orca.hf.adapters import gpt2, llama_family, bert, esm  # noqa: F401  (side-effect: registers)

__all__ = ["HfAdapter", "register", "get_adapter", "list_adapters"]
