"""Standard op library.

Each op spec carries:
- expected input arity (1 = unary, 2+ = multi-input like Add/Concat, -1 = variadic)
- a shape inference function `(args, input_shapes) -> output_shape`
- a parameter-count function `(args, input_shapes) -> int`
- a PyTorch emission function `(args) -> (init_code, forward_code_template)`

Shapes are tuples of strings — symbolic, never resolved numerically unless a
hyperparameter resolves to a literal int. That keeps the verifier robust to
symbolic batch / sequence / spatial dims.
"""
from n_orca.ops.spec import (
    OpSpec,
    get_op,
    op_names,
    UnknownOpError,
    ShapeRuleError,
)

__all__ = ["OpSpec", "get_op", "op_names", "UnknownOpError", "ShapeRuleError"]
