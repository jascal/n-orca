"""Op-spec registry for N-Orca's standard layer library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Shape = tuple[str, ...]


class UnknownOpError(Exception):
    pass


class ShapeRuleError(Exception):
    pass


@dataclass(frozen=True)
class OpSpec:
    name: str
    arity: int                              # -1 = variadic, otherwise exact count
    infer: Callable[[list[str], list[Shape]], Shape]
    params: Callable[[list[str], list[Shape]], int]
    pytorch_init: Callable[[list[str]], str]   # rhs of `self.<name> = ...`
    pytorch_call: Callable[[list[str], list[str]], str]  # python expression for forward()
    is_module: bool = True                  # False for functional ops (Add/Mul/Concat...)


# --------------------------------------------------------------------------- #
#  Helpers for shape-rule writing
# --------------------------------------------------------------------------- #


def _require_arity(name: str, expected: int, got: int) -> None:
    if expected != got:
        raise ShapeRuleError(
            f"op {name!r} expects {expected} input(s), got {got}"
        )


def _replace_last(shape: Shape, new_last: str) -> Shape:
    if not shape:
        raise ShapeRuleError("cannot replace last dim of an empty shape")
    return tuple(list(shape[:-1]) + [new_last])


def _all_equal(shapes: list[Shape]) -> bool:
    if not shapes:
        return True
    first = shapes[0]
    return all(s == first for s in shapes[1:])


def _conv_out_dim(dim: str, k: str, s: str, p: str) -> str:
    """Conv output: floor((H + 2p - k) / s) + 1, kept symbolic when possible.

    Algebraic simplifications when the spatial dim is symbolic:
      - 2p = k - 1, s = 1     ->  output = input (the "same padding" case)
      - 2p = k - 1, s > 1     ->  output = input / s (strided same-pad)
      - 2p = 0,    s = 1, k=1 ->  output = input (1x1 conv)
    Anything else falls through to a verbatim formula string.
    """
    try:
        di, ki, si, pi = int(dim), int(k), int(s), int(p)
        return str((di + 2 * pi - ki) // si + 1)
    except (TypeError, ValueError):
        pass
    try:
        ki, si, pi = int(k), int(s), int(p)
        if 2 * pi == ki - 1 and si == 1:
            return dim                         # same padding
        if 2 * pi == ki - 1 and si > 1:
            return f"{dim}/{si}"               # strided same padding
        if pi == 0 and si == 1 and ki == 1:
            return dim                         # 1x1 conv
    except (TypeError, ValueError):
        pass
    return f"({dim}+2*{p}-{k})/{s}+1"


def _pool_out_dim(dim: str, k: str, s: str) -> str:
    """Pool output: same as conv with p=0. Common case: k=s gives dim/s."""
    try:
        di, ki, si = int(dim), int(k), int(s)
        return str((di - ki) // si + 1)
    except (TypeError, ValueError):
        pass
    try:
        ki, si = int(k), int(s)
        if ki == si:
            return f"{dim}/{si}"
    except (TypeError, ValueError):
        pass
    return _conv_out_dim(dim, k, s, "0")


def _try_int(token: str) -> int | None:
    try:
        return int(token)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
#  Shape-rule implementations
# --------------------------------------------------------------------------- #


def _preserve(args, shapes):
    _require_arity("preserve", 1, len(shapes))
    return shapes[0]


def _linear_infer(args, shapes):
    _require_arity("Linear", 1, len(shapes))
    if len(args) < 2:
        raise ShapeRuleError("Linear requires (in, out) args")
    return _replace_last(shapes[0], args[1])


def _linear_params(args, shapes):
    if len(args) < 2:
        return 0
    ai, ao = _try_int(args[0]), _try_int(args[1])
    if ai is None or ao is None:
        return 0
    return ai * ao + ao


def _layernorm_infer(args, shapes):
    _require_arity("LayerNorm", 1, len(shapes))
    return shapes[0]


def _layernorm_params(args, shapes):
    if not args:
        return 0
    d = _try_int(args[0])
    return 2 * d if d is not None else 0


def _bn_params(args, shapes):
    if not args:
        return 0
    c = _try_int(args[0])
    return 2 * c if c is not None else 0


def _conv2d_infer(args, shapes):
    _require_arity("Conv2d", 1, len(shapes))
    if len(args) < 3:
        raise ShapeRuleError("Conv2d requires (in_c, out_c, k, [s], [p]) args")
    in_c, out_c, k = args[0], args[1], args[2]
    s = args[3] if len(args) > 3 else "1"
    p = args[4] if len(args) > 4 else "0"
    shape = shapes[0]
    if len(shape) != 4:
        raise ShapeRuleError(
            f"Conv2d expects (B, C, H, W); got {shape}"
        )
    b, _, h, w = shape
    return (b, out_c, _conv_out_dim(h, k, s, p), _conv_out_dim(w, k, s, p))


def _conv2d_params(args, shapes):
    if len(args) < 3:
        return 0
    in_c, out_c, k = _try_int(args[0]), _try_int(args[1]), _try_int(args[2])
    if in_c is None or out_c is None or k is None:
        return 0
    return in_c * out_c * k * k + out_c


def _pool2d_infer(args, shapes):
    _require_arity("Pool2d", 1, len(shapes))
    if not args:
        raise ShapeRuleError("Pool2d requires kernel arg")
    k = args[0]
    s = args[1] if len(args) > 1 else k
    shape = shapes[0]
    if len(shape) != 4:
        raise ShapeRuleError(f"Pool2d expects (B, C, H, W); got {shape}")
    b, c, h, w = shape
    return (b, c, _pool_out_dim(h, k, s), _pool_out_dim(w, k, s))


def _adaptive_avgpool2d_infer(args, shapes):
    _require_arity("AdaptiveAvgPool2d", 1, len(shapes))
    if not args:
        raise ShapeRuleError("AdaptiveAvgPool2d requires output-size arg")
    out = args[0]
    shape = shapes[0]
    if len(shape) != 4:
        raise ShapeRuleError(f"AdaptiveAvgPool2d expects (B, C, H, W); got {shape}")
    b, c, _, _ = shape
    return (b, c, out, out)


def _embedding_infer(args, shapes):
    _require_arity("Embedding", 1, len(shapes))
    if len(args) < 2:
        raise ShapeRuleError("Embedding requires (num_embeddings, embedding_dim) args")
    return shapes[0] + (args[1],)


def _embedding_params(args, shapes):
    if len(args) < 2:
        return 0
    n, d = _try_int(args[0]), _try_int(args[1])
    if n is None or d is None:
        return 0
    return n * d


def _mha_infer(args, shapes):
    _require_arity("MultiHeadAttention", 1, len(shapes))
    return shapes[0]


def _mha_params(args, shapes):
    if not args:
        return 0
    d = _try_int(args[0])
    if d is None:
        return 0
    return 4 * d * d + 4 * d


def _feedforward_infer(args, shapes):
    _require_arity("FeedForward", 1, len(shapes))
    return shapes[0]


def _feedforward_params(args, shapes):
    if len(args) < 2:
        return 0
    d, df = _try_int(args[0]), _try_int(args[1])
    if d is None or df is None:
        return 0
    return d * df + df + df * d + d


def _add_infer(args, shapes):
    if len(shapes) < 2:
        raise ShapeRuleError(
            f"Add requires at least 2 inputs; got {len(shapes)}"
        )
    if not _all_equal(shapes):
        raise ShapeRuleError(f"Add inputs have mismatched shapes: {shapes}")
    return shapes[0]


def _mul_infer(args, shapes):
    return _add_infer(args, shapes)


def _concat_infer(args, shapes):
    if not args:
        raise ShapeRuleError("Concat requires a `dim` arg")
    if len(shapes) < 2:
        raise ShapeRuleError(f"Concat requires at least 2 inputs; got {len(shapes)}")
    try:
        dim = int(args[0])
    except ValueError:
        raise ShapeRuleError(f"Concat dim must be an integer literal; got {args[0]!r}")
    # Normalize negative dim against the first shape.
    ref = shapes[0]
    if dim < 0:
        dim += len(ref)
    if dim < 0 or dim >= len(ref):
        raise ShapeRuleError(f"Concat dim {args[0]} out of range for shape {ref}")
    # Every non-concat dim must match across inputs.
    for s in shapes[1:]:
        if len(s) != len(ref):
            raise ShapeRuleError(f"Concat inputs differ in rank: {ref} vs {s}")
        for i in range(len(ref)):
            if i == dim:
                continue
            if s[i] != ref[i]:
                raise ShapeRuleError(
                    f"Concat inputs differ on dim {i}: {ref} vs {s}"
                )
    # Try to fold integer literals on the concat dim.
    sizes = [_try_int(s[dim]) for s in shapes]
    if all(x is not None for x in sizes):
        new_dim = str(sum(sizes))  # type: ignore[arg-type]
    else:
        new_dim = "+".join(s[dim] for s in shapes)
    out = list(ref)
    out[dim] = new_dim
    return tuple(out)


def _flatten_infer(args, shapes):
    _require_arity("Flatten", 1, len(shapes))
    start = int(args[0]) if args else 1
    shape = shapes[0]
    if start >= len(shape):
        return shape
    head = list(shape[:start])
    tail = shape[start:]
    if all(_try_int(d) is not None for d in tail):
        prod = 1
        for d in tail:
            prod *= int(d)
        head.append(str(prod))
    else:
        head.append("*".join(tail))
    return tuple(head)


def _reshape_infer(args, shapes):
    _require_arity("Reshape", 1, len(shapes))
    return tuple(args)


def _identity_infer(args, shapes):
    _require_arity("Identity", 1, len(shapes))
    return shapes[0]


# --------------------------------------------------------------------------- #
#  PyTorch emission helpers
# --------------------------------------------------------------------------- #


def _torch_init_simple(module: str):
    def emit(args: list[str]) -> str:
        return f"nn.{module}({', '.join(args)})"
    return emit


def _torch_call_module(args: list[str], inputs: list[str]) -> str:
    return inputs[0] if len(inputs) == 1 else "(" + ", ".join(inputs) + ")"


def _torch_call_mha(args: list[str], inputs: list[str]) -> str:
    x = inputs[0]
    return f"{x}, _ = self.{{NAME}}({x}, {x}, {x})"


def _torch_init_mha(args: list[str]) -> str:
    # nn.MultiheadAttention(embed_dim, num_heads, dropout=..., batch_first=True)
    if len(args) >= 3:
        d, h, drop = args[0], args[1], args[2]
        return f"nn.MultiheadAttention({d}, {h}, dropout={drop}, batch_first=True)"
    if len(args) == 2:
        d, h = args[0], args[1]
        return f"nn.MultiheadAttention({d}, {h}, batch_first=True)"
    return "nn.MultiheadAttention(*placeholders)"


def _torch_init_feedforward(args: list[str]) -> str:
    if len(args) >= 3:
        d, df, drop = args[0], args[1], args[2]
    elif len(args) >= 2:
        d, df = args[0], args[1]
        drop = "0.0"
    else:
        return "nn.Identity()  # FeedForward — missing args"
    return (
        "nn.Sequential("
        f"nn.Linear({d}, {df}), nn.GELU(), nn.Dropout({drop}),"
        f" nn.Linear({df}, {d}), nn.Dropout({drop})"
        ")"
    )


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #


_REGISTRY: dict[str, OpSpec] = {}


def _register(spec: OpSpec) -> None:
    _REGISTRY[spec.name] = spec


_register(OpSpec(
    "Linear", 1,
    infer=_linear_infer, params=_linear_params,
    pytorch_init=_torch_init_simple("Linear"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "LayerNorm", 1,
    infer=_layernorm_infer, params=_layernorm_params,
    pytorch_init=_torch_init_simple("LayerNorm"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "BatchNorm1d", 1,
    infer=_preserve, params=_bn_params,
    pytorch_init=_torch_init_simple("BatchNorm1d"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "BatchNorm2d", 1,
    infer=_preserve, params=_bn_params,
    pytorch_init=_torch_init_simple("BatchNorm2d"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "Conv2d", 1,
    infer=_conv2d_infer, params=_conv2d_params,
    pytorch_init=_torch_init_simple("Conv2d"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "MaxPool2d", 1,
    infer=_pool2d_infer, params=lambda a, s: 0,
    pytorch_init=_torch_init_simple("MaxPool2d"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "AvgPool2d", 1,
    infer=_pool2d_infer, params=lambda a, s: 0,
    pytorch_init=_torch_init_simple("AvgPool2d"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "AdaptiveAvgPool2d", 1,
    infer=_adaptive_avgpool2d_infer, params=lambda a, s: 0,
    pytorch_init=_torch_init_simple("AdaptiveAvgPool2d"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "Dropout", 1,
    infer=_preserve, params=lambda a, s: 0,
    pytorch_init=_torch_init_simple("Dropout"),
    pytorch_call=_torch_call_module,
))
for _act in ("ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"):
    _register(OpSpec(
        _act, 1,
        infer=_preserve, params=lambda a, s: 0,
        pytorch_init=_torch_init_simple(_act),
        pytorch_call=_torch_call_module,
    ))
_register(OpSpec(
    "Softmax", 1,
    infer=_preserve, params=lambda a, s: 0,
    pytorch_init=_torch_init_simple("Softmax"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "Embedding", 1,
    infer=_embedding_infer, params=_embedding_params,
    pytorch_init=_torch_init_simple("Embedding"),
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "MultiHeadAttention", 1,
    infer=_mha_infer, params=_mha_params,
    pytorch_init=_torch_init_mha,
    pytorch_call=_torch_call_mha,
))
_register(OpSpec(
    "FeedForward", 1,
    infer=_feedforward_infer, params=_feedforward_params,
    pytorch_init=_torch_init_feedforward,
    pytorch_call=_torch_call_module,
))
def _positional_embedding_infer(args, shapes):
    _require_arity("PositionalEmbedding", 1, len(shapes))
    if len(args) < 2:
        raise ShapeRuleError("PositionalEmbedding requires (max_pos, d_model) args")
    return shapes[0]


def _positional_embedding_params(args, shapes):
    if len(args) < 2:
        return 0
    mp, d = _try_int(args[0]), _try_int(args[1])
    if mp is None or d is None:
        return 0
    return mp * d


_register(OpSpec(
    "PositionalEmbedding", 1,
    infer=_positional_embedding_infer, params=_positional_embedding_params,
    pytorch_init=lambda a: f"nn.Embedding({a[0]}, {a[1]})" if len(a) >= 2 else "nn.Embedding(*placeholders)",
    pytorch_call=lambda a, inputs: (
        f"{inputs[0]} + self.{{NAME}}(torch.arange({inputs[0]}.shape[1], device={inputs[0]}.device))"
    ),
))


def _tokens_from_spatial(spatial: list[str], kernels: list[str]) -> str:
    """Fold a patch/tubelet token count from spatial dims + per-dim kernels.

    Each spatial dim `d` is downsampled by the matching kernel `k` (kernel ==
    stride for non-overlapping patch embedding), giving `d // k` patches along
    that axis; the token count is their product. When every value is an integer
    literal the count folds to a concrete integer; otherwise it stays symbolic
    as `(d0/k0)*(d1/k1)*...` so the verifier can still propagate a shape.
    """
    per_axis: list[str] = []
    literal = True
    for d, k in zip(spatial, kernels):
        di, ki = _try_int(d), _try_int(k)
        if di is not None and ki is not None:
            per_axis.append(str(di // ki))
        else:
            literal = False
            per_axis.append(f"{d}/{k}")
    if literal:
        prod = 1
        for v in per_axis:
            prod *= int(v)
        return str(prod)
    return "*".join(f"({a})" if "/" in a else a for a in per_axis)


def _tubelet_embed_infer(args, shapes):
    _require_arity("TubeletEmbed", 1, len(shapes))
    if len(args) < 4:
        raise ShapeRuleError(
            "TubeletEmbed requires (in_chans, embed_dim, tubelet_size, patch_size) args"
        )
    shape = shapes[0]
    if len(shape) != 5:
        raise ShapeRuleError(
            f"TubeletEmbed expects a 5D clip (B, C, T, H, W); got {shape}"
        )
    b, _c, t, h, w = shape
    embed_dim, tubelet, patch = args[1], args[2], args[3]
    n_tokens = _tokens_from_spatial([t, h, w], [tubelet, patch, patch])
    return (b, n_tokens, embed_dim)


def _tubelet_embed_params(args, shapes):
    if len(args) < 4:
        return 0
    ic, ed, tub, pat = (_try_int(args[0]), _try_int(args[1]),
                        _try_int(args[2]), _try_int(args[3]))
    if None in (ic, ed, tub, pat):
        return 0
    return ic * ed * tub * pat * pat + ed


def _patch_embed_infer(args, shapes):
    _require_arity("PatchEmbed", 1, len(shapes))
    if len(args) < 3:
        raise ShapeRuleError(
            "PatchEmbed requires (in_chans, embed_dim, patch_size) args"
        )
    shape = shapes[0]
    if len(shape) != 4:
        raise ShapeRuleError(
            f"PatchEmbed expects an image (B, C, H, W); got {shape}"
        )
    b, _c, h, w = shape
    embed_dim, patch = args[1], args[2]
    n_tokens = _tokens_from_spatial([h, w], [patch, patch])
    return (b, n_tokens, embed_dim)


def _patch_embed_params(args, shapes):
    if len(args) < 3:
        return 0
    ic, ed, pat = _try_int(args[0]), _try_int(args[1]), _try_int(args[2])
    if None in (ic, ed, pat):
        return 0
    return ic * ed * pat * pat + ed


def _tubelet_embed_init(args):
    if len(args) < 4:
        return "nn.Identity()  # TubeletEmbed — missing args"
    ic, ed, tub, pat = args[0], args[1], args[2], args[3]
    return (
        f"nn.Conv3d({ic}, {ed}, kernel_size=({tub}, {pat}, {pat}),"
        f" stride=({tub}, {pat}, {pat}))"
    )


def _patch_embed_init(args):
    if len(args) < 3:
        return "nn.Identity()  # PatchEmbed — missing args"
    ic, ed, pat = args[0], args[1], args[2]
    return f"nn.Conv2d({ic}, {ed}, kernel_size={pat}, stride={pat})"


# Conv (3D or 2D) patch projection, then flatten the spatial grid into a token
# sequence and move channels last so attention sees (B, N, embed_dim).
_register(OpSpec(
    "TubeletEmbed", 1,
    infer=_tubelet_embed_infer, params=_tubelet_embed_params,
    pytorch_init=_tubelet_embed_init,
    pytorch_call=lambda a, inputs: (
        f"self.{{NAME}}({inputs[0]}).flatten(2).transpose(1, 2)"
    ),
))
_register(OpSpec(
    "PatchEmbed", 1,
    infer=_patch_embed_infer, params=_patch_embed_params,
    pytorch_init=_patch_embed_init,
    pytorch_call=lambda a, inputs: (
        f"self.{{NAME}}({inputs[0]}).flatten(2).transpose(1, 2)"
    ),
))


def _resolve_in_forward(arg: str) -> str:
    """Resolve an op-arg token for use in a forward() body.

    Integer / float literals pass through verbatim. Anything else is assumed
    to be a hyperparameter name and rewritten as `self.<name>`. This lets a
    functional op-call template like `torch.topk(z, k={a[0]})` work both for
    `TopK(8)` (literal) and `TopK(k)` (hyperparameter reference).
    """
    try:
        float(arg)
        return arg
    except (TypeError, ValueError):
        return f"self.{arg}"


def _topk_infer(args, shapes):
    _require_arity("TopK", 1, len(shapes))
    if not args:
        raise ShapeRuleError("TopK requires a `k` arg")
    return shapes[0]


_register(OpSpec(
    "TopK", 1,
    infer=_topk_infer, params=lambda a, s: 0,
    pytorch_init=lambda a: "nn.Identity()  # TopK (functional)",
    # Sparsify by zeroing all but the k largest along the last dim.
    pytorch_call=lambda a, inputs: (
        f"({inputs[0]} * torch.zeros_like({inputs[0]})"
        f".scatter_(-1, torch.topk({inputs[0]}, k={_resolve_in_forward(a[0])}, dim=-1).indices, 1.0))"
    ),
    is_module=False,
))


def _jumprelu_infer(args, shapes):
    _require_arity("JumpReLU", 1, len(shapes))
    if not args:
        raise ShapeRuleError("JumpReLU requires a `d` (feature dim) arg")
    return shapes[0]


def _jumprelu_params(args, shapes):
    if not args:
        return 0
    d = _try_int(args[0])
    return d if d is not None else 0


def _jumprelu_init(args):
    """Emit init RHS for a JumpReLU buffer-only module.

    Stores a learnable log-threshold parameter of shape (d,) initialised so
    `exp(log_theta) ≈ theta_init` (default 0.1). The straight-through gate
    is applied in the call site, not here.
    """
    d = args[0] if args else "1"
    theta_init = args[1] if len(args) > 1 else "0.1"
    return (
        "nn.ParameterDict({"
        f"'log_theta': nn.Parameter(torch.full(({d},), torch.log(torch.tensor({theta_init}))))"
        "})"
    )


def _jumprelu_call(args, inputs):
    # Straight-through estimator: gate = (x > exp(log_theta)).float() in forward,
    # gradient flows through `x` only. We approximate with the simple gate;
    # the parent project supplies the STE-aware backward in a custom autograd Function.
    x = inputs[0]
    return (
        f"({x} * ({x} > self.{{NAME}}['log_theta'].exp()).to({x}.dtype))"
    )


_register(OpSpec(
    "JumpReLU", 1,
    infer=_jumprelu_infer, params=_jumprelu_params,
    pytorch_init=_jumprelu_init,
    pytorch_call=_jumprelu_call,
))


def _mean_infer(args, shapes):
    _require_arity("Mean", 1, len(shapes))
    if not args:
        raise ShapeRuleError("Mean requires a `dim` arg")
    try:
        dim = int(args[0])
    except ValueError:
        raise ShapeRuleError(f"Mean dim must be an integer; got {args[0]!r}")
    shape = list(shapes[0])
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        raise ShapeRuleError(f"Mean dim {args[0]} out of range for shape {shapes[0]}")
    return tuple(shape[:dim] + shape[dim + 1:])


_register(OpSpec(
    "Mean", 1,
    infer=_mean_infer, params=lambda a, s: 0,
    pytorch_init=lambda a: "nn.Identity()  # Mean (functional)",
    pytorch_call=lambda a, inputs: f"{inputs[0]}.mean(dim={a[0]})",
    is_module=False,
))
_register(OpSpec(
    "Flatten", 1,
    infer=_flatten_infer, params=lambda a, s: 0,
    pytorch_init=lambda a: f"nn.Flatten({a[0] if a else '1'})",
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "Reshape", 1,
    infer=_reshape_infer, params=lambda a, s: 0,
    pytorch_init=lambda a: "nn.Identity()  # reshape applied in forward()",
    pytorch_call=_torch_call_module,
    is_module=False,
))
_register(OpSpec(
    "Identity", 1,
    infer=_identity_infer, params=lambda a, s: 0,
    pytorch_init=lambda a: "nn.Identity()",
    pytorch_call=_torch_call_module,
))
_register(OpSpec(
    "Add", -1,
    infer=_add_infer, params=lambda a, s: 0,
    pytorch_init=lambda a: "nn.Identity()  # Add (functional)",
    pytorch_call=lambda a, inputs: " + ".join(inputs),
    is_module=False,
))
_register(OpSpec(
    "Mul", -1,
    infer=_mul_infer, params=lambda a, s: 0,
    pytorch_init=lambda a: "nn.Identity()  # Mul (functional)",
    pytorch_call=lambda a, inputs: " * ".join(inputs),
    is_module=False,
))
_register(OpSpec(
    "Concat", -1,
    infer=_concat_infer, params=lambda a, s: 0,
    pytorch_init=lambda a: "nn.Identity()  # Concat (functional)",
    pytorch_call=lambda a, inputs: f"torch.cat([{', '.join(inputs)}], dim={a[0]})",
    is_module=False,
))


def get_op(name: str) -> OpSpec:
    if name not in _REGISTRY:
        raise UnknownOpError(f"unknown op: {name!r}")
    return _REGISTRY[name]


def op_names() -> list[str]:
    return sorted(_REGISTRY.keys())
