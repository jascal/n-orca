"""Markdown parser for N-Orca architecture documents.

Parses `.n.orca.md` source into one or more `Architecture` AST objects.
The grammar is small and line-oriented; we tokenize headings and tables
directly without pulling in a general-purpose markdown library.
"""
from __future__ import annotations

import re
from pathlib import Path

from n_orca.ast import (
    Architecture,
    FlowEdge,
    Hyperparameter,
    Invariant,
    Layer,
    OpCall,
    Tensor,
)


class ParseError(Exception):
    def __init__(self, message: str, line: int | None = None):
        super().__init__(f"line {line}: {message}" if line else message)
        self.line = line
        self.message = message


_H1_RE = re.compile(r"^#\s+architecture\s+(\S+)\s*$")
_H2_RE = re.compile(r"^##\s+(.+?)\s*$")
_LAYER_HEADING_RE = re.compile(r"^##\s+layer\s+(\S+)(?:\s+\[([^\]]+)\])?\s*(?:\[([^\]]+)\])?\s*$")
_BLOCKQUOTE_RE = re.compile(r"^>\s*(.*)$")
_BULLET_RE = re.compile(r"^\s*-\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_DOC_SEPARATOR_RE = re.compile(r"^---+\s*$")
_OP_CALL_RE = re.compile(r"^(\w+)\s*(?:\((.*)\))?\s*$")
_INVARIANT_RE = re.compile(r"^(\w+)\s*(<=|>=|==|=)\s*(.+)$")


def parse_file(path: str | Path) -> list[Architecture]:
    """Parse an `.n.orca.md` file."""
    text = Path(path).read_text(encoding="utf-8")
    return parse(text)


def parse(source: str) -> list[Architecture]:
    """Parse `source` into one or more `Architecture` objects.

    Multiple architectures may appear separated by `---` (markdown horizontal
    rule) at the top level.
    """
    lines = source.splitlines()
    architectures: list[Architecture] = []

    # Split documents on bare `---` lines that are not inside a fence.
    docs: list[list[tuple[int, str]]] = [[]]
    for lineno, raw in enumerate(lines, start=1):
        if _DOC_SEPARATOR_RE.match(raw):
            docs.append([])
        else:
            docs[-1].append((lineno, raw))

    for doc in docs:
        if not any(_H1_RE.match(line) for _, line in doc):
            continue
        arch = _parse_document(doc)
        architectures.append(arch)

    if not architectures:
        raise ParseError("no `# architecture <Name>` heading found")

    return architectures


def _parse_document(doc_lines: list[tuple[int, str]]) -> Architecture:
    """Parse one document (one architecture)."""
    # Locate the H1 line.
    arch_name: str | None = None
    description: str | None = None
    h1_idx = -1
    for i, (_, line) in enumerate(doc_lines):
        m = _H1_RE.match(line)
        if m:
            arch_name = m.group(1)
            h1_idx = i
            break
    if arch_name is None:
        raise ParseError("missing `# architecture <Name>` heading")

    # Capture an optional blockquote description directly after the H1.
    for _, line in doc_lines[h1_idx + 1 :]:
        if not line.strip():
            continue
        m = _BLOCKQUOTE_RE.match(line)
        if m:
            description = m.group(1).strip() or None
        break

    arch = Architecture(name=arch_name, description=description)

    # Split the remaining lines into H2-keyed sections.
    sections: list[tuple[str, str | None, list[tuple[int, str]]]] = []
    # (section_kind, layer_name_if_layer, body_lines)
    current_kind: str | None = None
    current_layer: str | None = None
    current_body: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal current_kind, current_layer, current_body
        if current_kind is not None:
            sections.append((current_kind, current_layer, current_body))
        current_kind = None
        current_layer = None
        current_body = []

    for lineno, line in doc_lines[h1_idx + 1 :]:
        layer_m = _LAYER_HEADING_RE.match(line)
        h2_m = _H2_RE.match(line)
        if layer_m:
            flush()
            current_kind = "layer"
            current_layer = layer_m.group(1)
            # Markers — group 2 and group 3 may carry [input]/[output] etc.
            markers = [
                m for m in (layer_m.group(2), layer_m.group(3)) if m
            ]
            current_body = [(lineno, f"__markers__={','.join(markers)}")]
        elif h2_m:
            flush()
            heading = h2_m.group(1).strip().lower()
            heading = re.sub(r"\s+", " ", heading)
            current_kind = heading
            current_layer = None
            current_body = []
        else:
            if current_kind is not None:
                current_body.append((lineno, line))

    flush()

    # Dispatch sections to type-specific parsers.
    for kind, layer_name, body in sections:
        if kind == "layer":
            arch.layers.append(_parse_layer(layer_name or "", body))
        elif kind == "hyperparameters":
            arch.hyperparameters.extend(_parse_hyperparameters(body))
        elif kind == "tensors":
            arch.tensors.extend(_parse_tensors(body))
        elif kind == "flow":
            arch.flow.extend(_parse_flow(body))
        elif kind == "invariants":
            arch.invariants.extend(_parse_invariants(body))
        elif kind == "verification rules":
            arch.verification_rules.extend(_parse_verification_rules(body))
        else:
            # Unknown section — silently ignore. The verifier can warn later.
            pass

    return arch


# --------------------------------------------------------------------------- #
#  Per-section parsers
# --------------------------------------------------------------------------- #


def _parse_layer(name: str, body: list[tuple[int, str]]) -> Layer:
    layer = Layer(name=name)

    # First line of the body is the synthesized markers token.
    if body and body[0][1].startswith("__markers__="):
        markers_str = body[0][1].split("=", 1)[1]
        markers = [m.strip() for m in markers_str.split(",") if m.strip()]
        if "input" in markers:
            layer.is_input = True
        if "output" in markers:
            layer.is_output = True
        body = body[1:]

    for lineno, line in body:
        if not line.strip():
            continue
        bq = _BLOCKQUOTE_RE.match(line)
        if bq:
            text = bq.group(1).strip()
            if text:
                layer.description = (
                    text if layer.description is None else f"{layer.description} {text}"
                )
            continue
        bul = _BULLET_RE.match(line)
        if bul:
            kv = bul.group(1)
            if ":" in kv:
                key, val = kv.split(":", 1)
                key = key.strip().lower()
                val = val.strip()
                if key == "op":
                    layer.op = _parse_op_call(val, lineno)
                elif key == "shape":
                    layer.declared_shape = _parse_shape(val, lineno)
                elif key == "params":
                    # params: { k: v, ... }
                    layer.params = _parse_inline_dict(val, lineno)
            continue
        # Other content in a layer body is ignored for now.

    return layer


def _parse_op_call(text: str, lineno: int) -> OpCall:
    m = _OP_CALL_RE.match(text)
    if not m:
        raise ParseError(f"invalid op call: {text!r}", lineno)
    name = m.group(1)
    raw_args = m.group(2) or ""
    args = _split_args(raw_args)
    return OpCall(name=name, args=args)


def _split_args(raw: str) -> list[str]:
    """Split op-call arguments by commas, respecting nested parentheses."""
    raw = raw.strip()
    if not raw:
        return []
    out: list[str] = []
    depth = 0
    cur = []
    for ch in raw:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return [a for a in out if a]


def _parse_shape(text: str, lineno: int) -> tuple[str, ...]:
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ParseError(f"shape must be wrapped in parentheses: {text!r}", lineno)
    inner = text[1:-1].strip()
    if not inner:
        return ()
    return tuple(_split_args(inner))


def _parse_inline_dict(text: str, lineno: int) -> dict[str, str]:
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ParseError(f"params must be in {{...}}: {text!r}", lineno)
    inner = text[1:-1].strip()
    out: dict[str, str] = {}
    for part in _split_args(inner):
        if ":" not in part:
            raise ParseError(f"bad params entry: {part!r}", lineno)
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_table(body: list[tuple[int, str]]) -> list[list[str]]:
    """Extract rows from a markdown table; drops header + separator."""
    rows: list[list[str]] = []
    for lineno, line in body:
        m = _TABLE_ROW_RE.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        # Separator row?
        if all(re.match(r"^:?-+:?$", c) for c in cells if c):
            continue
        rows.append(cells)
    if not rows:
        return []
    # Drop the header row.
    return rows[1:]


def _parse_hyperparameters(body: list[tuple[int, str]]) -> list[Hyperparameter]:
    out: list[Hyperparameter] = []
    for cells in _parse_table(body):
        if len(cells) < 2:
            continue
        name = cells[0]
        typ = cells[1]
        default_raw = cells[2] if len(cells) > 2 else ""
        out.append(Hyperparameter(name=name, type=typ, default=_coerce_default(typ, default_raw)))
    return out


def _coerce_default(typ: str, raw: str) -> object | None:
    if not raw:
        return None
    typ = typ.lower()
    try:
        if typ == "int":
            return int(raw)
        if typ == "float":
            return float(raw)
        if typ == "bool":
            return raw.lower() in ("true", "1", "yes")
        return raw
    except (ValueError, TypeError):
        return raw


def _parse_tensors(body: list[tuple[int, str]]) -> list[Tensor]:
    out: list[Tensor] = []
    for cells in _parse_table(body):
        if len(cells) < 3:
            continue
        name, shape_raw, dtype = cells[0], cells[1], cells[2]
        out.append(Tensor(name=name, shape=_parse_shape(shape_raw, 0), dtype=dtype))
    return out


def _parse_flow(body: list[tuple[int, str]]) -> list[FlowEdge]:
    out: list[FlowEdge] = []
    for cells in _parse_table(body):
        if len(cells) < 2:
            continue
        source = cells[0]
        target = cells[1]
        tensor = cells[2] if len(cells) > 2 else ""
        out.append(FlowEdge(source=source, target=target, tensor=tensor))
    return out


def _parse_invariants(body: list[tuple[int, str]]) -> list[Invariant]:
    out: list[Invariant] = []
    for _, line in body:
        bul = _BULLET_RE.match(line)
        if not bul:
            continue
        text = bul.group(1).strip()
        # Either `key: value` form (output_shape: (B, S, d)) or `key op value`.
        if ":" in text and not any(c in text.split(":")[0] for c in "<>="):
            key, val = text.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key == "output_shape":
                out.append(Invariant(kind=key, op="=", value=_parse_shape(val, 0)))
            else:
                out.append(Invariant(kind=key, op="=", value=val))
            continue
        m = _INVARIANT_RE.match(text)
        if m:
            kind, op, value = m.group(1), m.group(2), m.group(3).strip()
            out.append(Invariant(kind=kind, op=op, value=_parse_invariant_value(value)))
    return out


def _parse_invariant_value(raw: str) -> object:
    # Accept suffixes K/M/G (decimal SI).
    suffix_mul = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}
    raw = raw.strip()
    if raw and raw[-1] in suffix_mul:
        try:
            base = float(raw[:-1])
            return int(base * suffix_mul[raw[-1]])
        except ValueError:
            return raw
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_verification_rules(body: list[tuple[int, str]]) -> list[str]:
    out: list[str] = []
    for _, line in body:
        bul = _BULLET_RE.match(line)
        if bul:
            out.append(bul.group(1).strip())
    return out
