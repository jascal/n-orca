"""`n-orca` command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from n_orca import __version__
from n_orca.compiler import compile_mermaid, compile_pytorch
from n_orca.parser import parse_file, ParseError
from n_orca.verifier import verify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="n-orca",
        description=(
            "N-Orca — Markdown DSL for neural network architectures."
            " Verify, visualize, and compile `.n.orca.md` files."
        ),
    )
    parser.add_argument("--version", action="version", version=f"n-orca {__version__}")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_verify = subs.add_parser("verify", help="verify a `.n.orca.md` file")
    p_verify.add_argument("file", type=Path)
    p_verify.add_argument("--json", action="store_true", help="JSON output")
    p_verify.add_argument("--strict", action="store_true", help="treat warnings as errors")

    p_compile = subs.add_parser("compile", help="compile to a target backend")
    p_compile.add_argument("target", choices=["mermaid", "pytorch"])
    p_compile.add_argument("file", type=Path)
    p_compile.add_argument("--out", type=Path, help="output path (default: stdout)")

    p_info = subs.add_parser("info", help="summarize an architecture")
    p_info.add_argument("file", type=Path)

    p_hf = subs.add_parser("hf", help="Hugging Face Hub operations")
    hf_subs = p_hf.add_subparsers(dest="hf_cmd", required=True)

    p_search = hf_subs.add_parser("search", help="search Hub models")
    p_search.add_argument("query", nargs="?", default=None)
    p_search.add_argument("--task", help='filter by pipeline tag (e.g. "text-generation")')
    p_search.add_argument("--library", help='filter by library (e.g. "transformers")')
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--sort", default="downloads",
                          choices=["downloads", "likes", "lastModified"])
    p_search.add_argument("--json", action="store_true")

    p_hf_info = hf_subs.add_parser("info", help="show model metadata + config")
    p_hf_info.add_argument("model_id")
    p_hf_info.add_argument("--revision", default=None)
    p_hf_info.add_argument("--no-config", action="store_true", help="skip downloading config.json")
    p_hf_info.add_argument("--json", action="store_true")

    p_dl = hf_subs.add_parser("download", help="download a model snapshot")
    p_dl.add_argument("model_id")
    p_dl.add_argument("--revision", default=None)
    p_dl.add_argument("--config-only", action="store_true",
                      help="download only config.json")
    p_dl.add_argument("--allow", action="append", default=None,
                      help="glob to include (repeatable)")
    p_dl.add_argument("--local-dir", type=Path, default=None,
                      help="where to put the snapshot")

    p_conv = hf_subs.add_parser("convert", help="convert HF model -> n-orca")
    p_conv.add_argument("source",
                        help="HF model id, path to config.json, or '-' for stdin JSON")
    p_conv.add_argument("--revision", default=None)
    p_conv.add_argument("--name", help="override architecture name")
    p_conv.add_argument("--out", type=Path,
                        help="write the markdown here (default: stdout)")
    p_conv.add_argument("--mermaid", type=Path,
                        help="also write the mermaid diagram to this path")
    p_conv.add_argument("--no-verify", action="store_true",
                        help="skip the post-conversion verify summary")

    args = parser.parse_args(argv)

    if args.cmd == "verify":
        return _cmd_verify(args)
    if args.cmd == "compile":
        return _cmd_compile(args)
    if args.cmd == "info":
        return _cmd_info(args)
    if args.cmd == "hf":
        return _cmd_hf(args)
    return 2


def _cmd_verify(args) -> int:
    try:
        archs = parse_file(args.file)
    except ParseError as ex:
        print(f"parse error: {ex}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"file not found: {args.file}", file=sys.stderr)
        return 1

    overall_ok = True
    reports = []
    for arch in archs:
        report = verify(arch, strict=args.strict)
        reports.append(report)
        if not report.valid:
            overall_ok = False

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
        return 0 if overall_ok else 1

    for report in reports:
        _print_human_report(report)
    return 0 if overall_ok else 1


def _print_human_report(report) -> None:
    head = "VALID" if report.valid else "INVALID"
    print(f"Architecture: {report.architecture}")
    print(f"  Result: {head}")
    print(f"  Parameters: {report.param_count:,}")
    print(f"  Depth: {report.depth}")
    for err in report.errors:
        line = f"  [ERR]  {err.code}: {err.message}"
        print(line)
        if err.suggestion:
            print(f"        -> {err.suggestion}")
    for warn in report.warnings:
        line = f"  [WARN] {warn.code}: {warn.message}"
        print(line)
        if warn.suggestion:
            print(f"        -> {warn.suggestion}")
    print()


def _cmd_compile(args) -> int:
    try:
        archs = parse_file(args.file)
    except (ParseError, FileNotFoundError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    if not archs:
        print("error: no architectures found", file=sys.stderr)
        return 1

    parts: list[str] = []
    for arch in archs:
        if args.target == "mermaid":
            parts.append(compile_mermaid(arch))
        elif args.target == "pytorch":
            parts.append(compile_pytorch(arch))

    output = "\n\n".join(parts)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(output)
    return 0


def _cmd_info(args) -> int:
    try:
        archs = parse_file(args.file)
    except (ParseError, FileNotFoundError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    for arch in archs:
        report = verify(arch)
        print(f"Architecture: {arch.name}")
        if arch.description:
            print(f"  Description: {arch.description}")
        print(f"  Hyperparameters: {len(arch.hyperparameters)}")
        for hp in arch.hyperparameters:
            print(f"    - {hp.name}: {hp.type} = {hp.default}")
        print(f"  Layers: {len(arch.layers)}")
        for ly in arch.layers:
            markers = []
            if ly.is_input:
                markers.append("input")
            if ly.is_output:
                markers.append("output")
            m = f" [{','.join(markers)}]" if markers else ""
            op = f" — {ly.op.name}({', '.join(ly.op.args)})" if ly.op else ""
            print(f"    - {ly.name}{m}{op}")
        print(f"  Flow edges: {len(arch.flow)}")
        print(f"  Params: {report.param_count:,}")
        print(f"  Depth: {report.depth}")
        print()
    return 0


def _cmd_hf(args) -> int:
    if args.hf_cmd == "search":
        return _cmd_hf_search(args)
    if args.hf_cmd == "info":
        return _cmd_hf_info(args)
    if args.hf_cmd == "download":
        return _cmd_hf_download(args)
    if args.hf_cmd == "convert":
        return _cmd_hf_convert(args)
    return 2


def _cmd_hf_search(args) -> int:
    from n_orca.hf import HfClient, HfClientError
    try:
        results = HfClient().search(
            query=args.query, task=args.task, library=args.library,
            limit=args.limit, sort=args.sort,
        )
    except HfClientError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
        return 0

    if not results:
        print("(no results)")
        return 0
    width = max(len(r.id) for r in results)
    print(f"{'MODEL':<{width}}  {'DOWNLOADS':>12}  {'LIKES':>6}  TASK / TAGS")
    for r in results:
        tags = (r.pipeline_tag or "") + (" " + " ".join(r.tags[:3]) if r.tags else "")
        print(f"{r.id:<{width}}  {r.downloads:>12,}  {r.likes:>6,}  {tags.strip()}")
    return 0


def _cmd_hf_info(args) -> int:
    from n_orca.hf import HfClient, HfClientError
    try:
        info = HfClient().info(
            args.model_id,
            revision=args.revision,
            include_config=not args.no_config,
        )
    except HfClientError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(info.__dict__, indent=2, default=str))
        return 0

    print(f"Model:        {info.id}")
    if info.sha:
        print(f"Revision:     {info.sha}")
    if info.pipeline_tag:
        print(f"Pipeline:     {info.pipeline_tag}")
    if info.library_name:
        print(f"Library:      {info.library_name}")
    print(f"Downloads:    {info.downloads:,}")
    print(f"Likes:        {info.likes:,}")
    if info.last_modified:
        print(f"Last modified: {info.last_modified}")
    if info.tags:
        print(f"Tags:         {', '.join(info.tags[:10])}")
    if info.config:
        mt = info.config.get("model_type") or "?"
        archs = info.config.get("architectures") or []
        print(f"Model type:   {mt}")
        if archs:
            print(f"Architectures: {', '.join(archs)}")
        from n_orca.hf.adapters import get_adapter
        adapter = get_adapter(info.config)
        if adapter:
            print(f"n-orca:       supported by `{type(adapter).__name__}`")
        else:
            print("n-orca:       no adapter registered (cannot convert)")
    print(f"\nFiles ({len(info.siblings)}):")
    for fn in info.siblings[:15]:
        print(f"  - {fn}")
    if len(info.siblings) > 15:
        print(f"  ... +{len(info.siblings) - 15} more")
    return 0


def _cmd_hf_download(args) -> int:
    from n_orca.hf import HfClient, HfClientError
    try:
        client = HfClient()
        if args.config_only:
            cfg_path = client.download_file(args.model_id, "config.json",
                                            revision=args.revision)
            print(f"wrote {cfg_path}")
            return 0
        snap = client.download_model(
            args.model_id,
            revision=args.revision,
            allow_patterns=args.allow,
            local_dir=args.local_dir,
        )
        print(f"wrote snapshot to {snap}")
        return 0
    except HfClientError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1


def _cmd_hf_convert(args) -> int:
    from n_orca.hf import convert, UnsupportedModelError, HfClientError

    if args.source == "-":
        config = json.load(sys.stdin)
        source = config
    else:
        source = args.source

    try:
        result = convert(source, revision=args.revision, name=args.name)
    except (UnsupportedModelError, HfClientError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    if args.out:
        result.write_markdown(args.out)
        print(f"wrote {args.out}")
    else:
        print(result.markdown)

    if args.mermaid:
        result.write_mermaid(args.mermaid)
        print(f"wrote {args.mermaid}", file=sys.stderr)

    if not args.no_verify:
        head = "VALID" if result.report.valid else "INVALID"
        print(
            f"\n# Verify: {head}  params={result.report.param_count:,}"
            f"  depth={result.report.depth}  layers={len(result.architecture.layers)}",
            file=sys.stderr,
        )
        for e in result.report.errors[:5]:
            print(f"#   [ERR] {e.code}: {e.message}", file=sys.stderr)
        for w in result.report.warnings[:5]:
            print(f"#   [WARN] {w.code}: {w.message}", file=sys.stderr)

    return 0 if result.report.valid else 1


if __name__ == "__main__":
    sys.exit(main())
