"""Hugging Face integration for n-orca.

Lets you search, inspect, download, and convert HF Hub models into n-orca
architecture documents. Depends only on `huggingface_hub` (not `transformers`)
because we read `config.json` directly — no model code execution required.

Install with `pip install n-orca[hf]`.
"""
from n_orca.hf.client import (
    HfClient,
    HfModelSummary,
    HfModelInfo,
    HfClientError,
)
from n_orca.hf.convert import convert, ConversionResult, UnsupportedModelError

__all__ = [
    "HfClient",
    "HfModelSummary",
    "HfModelInfo",
    "HfClientError",
    "convert",
    "ConversionResult",
    "UnsupportedModelError",
]
