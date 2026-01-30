import ast
import operator as op
import math
from typing import Any, Dict, Tuple

# Allowed binary operators mapping
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

# Allowed unary operators mapping
_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
}

# Whitelisted functions from math
_ALLOWED_FUNCS = {
    # basic
    "abs": abs,
    "round": round,
    # trig
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    # exp/log
    "exp": math.exp,
    "log": math.log,   # natural log; log(x, base) also supported if two args supplied
    "ln": math.log,
    "log10": math.log10,
    # power / sqrt
    "sqrt": math.sqrt,
    "pow": pow,
    # misc
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
}

# constants
_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": float("inf"),
    "nan": float("nan"),
}


class CalculatorError(Exception):
    """Custom exception for calculator errors."""
    pass


class _EvalVisitor(ast.NodeVisitor):
    """
    AST visitor that evaluates a math expression with a whitelist.
    """

    def __init__(self, funcs: Dict[str, Any], names: Dict[str, Any]):
        self.funcs = funcs
        self.names = names

    def visit(self, node):
        # Only allow certain node types; otherwise raise
        if isinstance(node, ast.Expression):
            return self.visit(node.body)

        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op_type = type(node.op)
            if op_type in _ALLOWED_BINOPS:
                return _ALLOWED_BINOPS[op_type](left, right)
            raise CalculatorError(f"Unsupported binary operator: {op_type.__name__}")

        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            op_type = type(node.op)
            if op_type in _ALLOWED_UNARYOPS:
                return _ALLOWED_UNARYOPS[op_type](operand)
            raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}")

        if isinstance(node, ast.Call):
            # only allow direct function names (no attribute access)
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in self.funcs:
                    args = [self.visit(arg) for arg in node.args]
                    # support keyword args? not allowed for safety
                    if node.keywords:
                        raise CalculatorError("Keyword arguments are not allowed in function calls.")
                    func = self.funcs[fname]
                    try:
                        return func(*args)
                    except Exception as e:
                        raise CalculatorError(f"Function call error: {fname} -> {e}")
                else:
                    raise CalculatorError(f"Function '{fname}' is not allowed.")
            else:
                raise CalculatorError("Only direct function calls by name are allowed.")

        if isinstance(node, ast.Num):  # Python <=3.7
            return node.n

        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise CalculatorError(f"Constants of type {type(node.value).__name__} are not allowed.")

        if isinstance(node, ast.Name):
            name = node.id
            if name in self.names:
                return self.names[name]
            # prevent access to builtins
            raise CalculatorError(f"Name '{name}' is not allowed.")

        if isinstance(node, ast.Expr):
            return self.visit(node.value)

        # disallow everything else
        raise CalculatorError(f"Unsupported expression: {node.__class__.__name__}")


class CalculatorTool:
    """
    Safe calculator tool. Use .evaluate(expr) -> (value, formatted_str)
    """

    def __init__(self, functions: Dict[str, Any] = None, names: Dict[str, Any] = None):
        funcs = {}
        funcs.update(_ALLOWED_FUNCS)
        if functions:
            funcs.update(functions)
        names = {}
        names.update(_ALLOWED_NAMES)
        if names:
            names.update(names)
        self._visitor = _EvalVisitor(funcs, names)

    def _parse_expr(self, expr: str) -> ast.AST:
        try:
            # parse expression in 'eval' mode
            parsed = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise CalculatorError(f"Syntax error: {e}")
        # Walk AST to ensure no disallowed nodes at top level
        for node in ast.walk(parsed):
            # disallow attribute access, imports, lambda, comprehensions, etc.
            if isinstance(node, (ast.Attribute, ast.Subscript, ast.ListComp, ast.DictComp,
                                 ast.SetComp, ast.GeneratorExp, ast.Lambda, ast.ClassDef,
                                 ast.Import, ast.ImportFrom, ast.Assign, ast.Delete)):
                raise CalculatorError(f"Disallowed syntax: {node.__class__.__name__}")
        return parsed

    def evaluate(self, expr: str) -> Tuple[Any, str]:
        """
        Evaluate an arithmetic/math expression safely.
        Returns (value, human_readable_string)
        """
        if not isinstance(expr, str):
            raise CalculatorError("Expression must be a string.")
        expr = expr.strip()
        if expr == "":
            raise CalculatorError("Empty expression.")
        parsed = self._parse_expr(expr)
        try:
            val = self._visitor.visit(parsed)
        except CalculatorError:
            raise
        except Exception as e:
            raise CalculatorError(f"Evaluation error: {e}")

        # Format result nicely: prefer int if close to integer
        if isinstance(val, float):
            if abs(val - round(val)) < 1e-12:
                val_out = int(round(val))
            else:
                # limit float representation
                val_out = float(val)
        else:
            val_out = val

        # produce string representation
        try:
            s = repr(val_out)
        except Exception:
            s = str(val_out)
        return val_out, s


# --------------------------
# Example integration
# --------------------------
if __name__ == "__main__":
    calc = CalculatorTool()
    tests = [
        "2+3*4",
        " ( 2+3 ) * 4 ",
        "sqrt(16) + pow(2,3)",
        "sin(pi/2)",
        "log(8, 2)",     # log with base -> math.log accepts (x, base)
        "-3 + 4",
        "abs(-5.2)",
        "10 // 3",       # FloorDiv is allowed if parsed as //
        "10 % 3",
    ]
    for t in tests:
        try:
            v, s = calc.evaluate(t)
            print(f"{t} => {s}")
        except Exception as e:
            print(f"{t} => Error: {e}")
