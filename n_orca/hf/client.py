"""Thin wrapper around `huggingface_hub` for search, info, and download.

We deliberately use `huggingface_hub` directly (not `transformers`):
- `config.json` carries everything n-orca needs to build an architecture.
- Avoids the heavy `transformers` import.
- No `trust_remote_code` surface — we never execute model code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import importlib
import json


class HfClientError(Exception):
    """Raised when an HF Hub operation fails or the library is missing."""


@dataclass
class HfModelSummary:
    """Lightweight record returned by `search`."""
    id: str
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0


@dataclass
class HfModelInfo:
    """Detailed record returned by `info`."""
    id: str
    sha: str | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    last_modified: str | None = None
    siblings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


class HfClient:
    """A small, lazy-imported wrapper around `huggingface_hub`.

    Public methods deliberately accept and return plain data structures —
    no `huggingface_hub` types leak across the boundary.
    """

    def __init__(self, token: str | None = None):
        self._token = token
        self._hub = None  # type: ignore[assignment]

    @property
    def hub(self):
        """Lazy-import the huggingface_hub module. Raises if not installed."""
        if self._hub is None:
            try:
                self._hub = importlib.import_module("huggingface_hub")
            except ImportError as ex:
                raise HfClientError(
                    "huggingface_hub is not installed; install with `pip install n-orca[hf]`"
                ) from ex
        return self._hub

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str | None = None,
        *,
        task: str | None = None,
        library: str | None = None,
        limit: int = 20,
        sort: str = "downloads",
    ) -> list[HfModelSummary]:
        """Search the Hub. Returns at most `limit` summaries."""
        api = self.hub.HfApi(token=self._token)
        # `list_models` returns ModelInfo objects with a subset of fields.
        kwargs: dict = {
            "search": query,
            "filter": _build_filter(task=task, library=library),
            "limit": limit,
            "sort": sort,
        }
        # `direction` was removed from list_models in huggingface_hub >= 0.27.
        # Fall back gracefully if the installed version doesn't accept it.
        try:
            try:
                models = api.list_models(direction=-1, **kwargs)
            except TypeError:
                models = api.list_models(**kwargs)
        except Exception as ex:
            raise HfClientError(f"search failed: {ex}") from ex

        out: list[HfModelSummary] = []
        for m in models:
            out.append(HfModelSummary(
                id=getattr(m, "modelId", None) or getattr(m, "id", "") or "",
                pipeline_tag=getattr(m, "pipeline_tag", None),
                library_name=getattr(m, "library_name", None),
                tags=list(getattr(m, "tags", []) or []),
                downloads=int(getattr(m, "downloads", 0) or 0),
                likes=int(getattr(m, "likes", 0) or 0),
            ))
        return out

    # ------------------------------------------------------------------ #
    #  Info
    # ------------------------------------------------------------------ #

    def info(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        include_config: bool = True,
    ) -> HfModelInfo:
        """Fetch model metadata and (optionally) `config.json`."""
        api = self.hub.HfApi(token=self._token)
        try:
            m = api.model_info(model_id, revision=revision)
        except Exception as ex:
            raise HfClientError(f"info({model_id!r}) failed: {ex}") from ex

        siblings = [s.rfilename for s in (m.siblings or [])]
        info = HfModelInfo(
            id=getattr(m, "modelId", None) or getattr(m, "id", "") or model_id,
            sha=getattr(m, "sha", None),
            pipeline_tag=getattr(m, "pipeline_tag", None),
            library_name=getattr(m, "library_name", None),
            tags=list(getattr(m, "tags", []) or []),
            downloads=int(getattr(m, "downloads", 0) or 0),
            likes=int(getattr(m, "likes", 0) or 0),
            last_modified=str(getattr(m, "last_modified", "") or "") or None,
            siblings=siblings,
        )
        if include_config and "config.json" in siblings:
            info.config = self.download_config(model_id, revision=revision)
        return info

    # ------------------------------------------------------------------ #
    #  Download
    # ------------------------------------------------------------------ #

    def download_config(
        self,
        model_id: str,
        *,
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Download and parse just `config.json` — cheap and weights-free."""
        path = self._hub_download(model_id, "config.json", revision=revision)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def download_file(
        self,
        model_id: str,
        filename: str,
        *,
        revision: str | None = None,
    ) -> Path:
        """Download a single file from the repo and return its local path."""
        return Path(self._hub_download(model_id, filename, revision=revision))

    def download_model(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        local_dir: str | Path | None = None,
    ) -> Path:
        """Snapshot-download a model (config + weights + tokenizer)."""
        try:
            snap = self.hub.snapshot_download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=allow_patterns,
                local_dir=str(local_dir) if local_dir else None,
                token=self._token,
            )
        except Exception as ex:
            raise HfClientError(f"download_model({model_id!r}) failed: {ex}") from ex
        return Path(snap)

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _hub_download(self, repo_id: str, filename: str, revision: str | None):
        try:
            return self.hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=self._token,
            )
        except Exception as ex:
            raise HfClientError(
                f"download({repo_id!r}, {filename!r}) failed: {ex}"
            ) from ex


def _build_filter(*, task: str | None, library: str | None):
    """Build the value passed to HfApi.list_models(filter=...)."""
    parts: list[str] = []
    if task:
        parts.append(task)
    if library:
        parts.append(library)
    if not parts:
        return None
    return parts if len(parts) > 1 else parts[0]
