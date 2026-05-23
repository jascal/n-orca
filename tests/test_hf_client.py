"""Tests for the HfClient wrapper.

These tests mock `huggingface_hub` rather than hitting the real Hub.
They confirm that `HfClient` calls the right surface and adapts results
to our plain dataclasses.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from n_orca.hf import HfClient, HfClientError


class _FakeSibling:
    def __init__(self, fn): self.rfilename = fn


class _FakeModelInfo:
    def __init__(self, **kw):
        self.modelId = kw.get("id", "fake/model")
        self.sha = kw.get("sha", "abc123")
        self.pipeline_tag = kw.get("pipeline_tag", "text-generation")
        self.library_name = kw.get("library_name", "transformers")
        self.tags = kw.get("tags", ["pytorch", "gpt2"])
        self.downloads = kw.get("downloads", 100)
        self.likes = kw.get("likes", 10)
        self.last_modified = kw.get("last_modified", "2024-01-01")
        self.siblings = [_FakeSibling(f) for f in kw.get("siblings", ["config.json"])]


class _FakeHfApi:
    last_search: dict = {}

    def __init__(self, token=None):
        self.token = token

    def list_models(self, **kw):
        _FakeHfApi.last_search = dict(kw)
        return [
            _FakeModelInfo(id="fake/gpt2-tiny", downloads=42, likes=3,
                           pipeline_tag="text-generation",
                           tags=["pytorch", "gpt2"]),
            _FakeModelInfo(id="fake/bert-tiny", downloads=10, likes=1,
                           pipeline_tag="fill-mask",
                           tags=["pytorch", "bert"]),
        ]

    def model_info(self, model_id, revision=None):
        return _FakeModelInfo(id=model_id, sha=(revision or "main-sha"))


def _install_fake_hub(monkeypatch, tmp_path):
    """Install a fake `huggingface_hub` module into sys.modules."""
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = _FakeHfApi

    def _hf_hub_download(repo_id, filename, revision=None, token=None):
        # Write a tiny config.json.
        out = tmp_path / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"model_type": "gpt2", "n_layer": 2}', encoding="utf-8")
        return str(out)

    def _snapshot_download(repo_id, revision=None, allow_patterns=None,
                            local_dir=None, token=None):
        return str(tmp_path)

    fake.hf_hub_download = _hf_hub_download
    fake.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def test_lazy_import_raises_when_missing(monkeypatch):
    # Force import to fail.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    def boom(name, *a, **k):
        if name == "huggingface_hub":
            raise ImportError("no module")
        return importlib.__import__(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", boom)
    client = HfClient()
    with pytest.raises(HfClientError):
        _ = client.hub


def test_search_returns_summaries(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, tmp_path)
    results = HfClient().search("gpt", task="text-generation", limit=5)
    assert len(results) == 2
    assert results[0].id == "fake/gpt2-tiny"
    assert results[0].downloads == 42
    assert "text-generation" in (results[0].pipeline_tag or "")


def test_search_passes_filter(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, tmp_path)
    HfClient().search("q", task="text-generation", library="transformers")
    f = _FakeHfApi.last_search.get("filter")
    assert "text-generation" in f and "transformers" in f


def test_info_includes_config(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, tmp_path)
    info = HfClient().info("fake/gpt2-tiny")
    assert info.id == "fake/gpt2-tiny"
    assert info.config.get("model_type") == "gpt2"


def test_info_skips_config_when_requested(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, tmp_path)
    info = HfClient().info("fake/gpt2-tiny", include_config=False)
    assert info.config == {}


def test_download_config_returns_dict(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, tmp_path)
    cfg = HfClient().download_config("fake/gpt2-tiny")
    assert cfg["model_type"] == "gpt2"


def test_download_model_returns_path(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, tmp_path)
    p = HfClient().download_model("fake/gpt2-tiny")
    assert str(p) == str(tmp_path)


def test_convert_via_client_downloads_config(monkeypatch, tmp_path):
    """convert() with a string source uses the client to download config."""
    _install_fake_hub(monkeypatch, tmp_path)
    from n_orca.hf import convert
    # The fake config sets n_layer=2 — enough to verify the wiring without
    # producing a huge architecture.
    result = convert("fake/gpt2-tiny", client=HfClient())
    assert result.model_id == "fake/gpt2-tiny"
    assert result.report.valid
