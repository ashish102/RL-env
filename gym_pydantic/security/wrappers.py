"""Safe wrappers for commonly used libraries in transition functions."""

import numpy as np
from typing import Any


class SafeNumpyWrapper:
    """
    Safe wrapper for numpy operations.

    Blocks dangerous operations:
    - File I/O (load, save, loadtxt, savetxt, etc.)
    - Prevents excessively large arrays
    - Whitelists math operations only

    Example:
        ```python
        safe_np = SafeNumpyWrapper(max_array_size=10_000_000)
        x = safe_np.array([1, 2, 3])
        y = safe_np.clip(x, 0, 10)
        ```
    """

    # Whitelist of allowed numpy functions
    ALLOWED_FUNCTIONS = {
        # Array creation
        "array",
        "zeros",
        "ones",
        "empty",
        "arange",
        "linspace",
        # Math operations
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "sqrt",
        "square",
        "exp",
        "log",
        "log10",
        "log2",
        "abs",
        "absolute",
        "sign",
        "clip",
        "minimum",
        "maximum",
        # Trigonometric
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "arctan2",
        "sinh",
        "cosh",
        "tanh",
        # Statistics
        "mean",
        "median",
        "std",
        "var",
        "sum",
        "min",
        "max",
        "argmin",
        "argmax",
        # Linear algebra (basic)
        "dot",
        "cross",
        "norm",
        "transpose",
        # Shape manipulation
        "reshape",
        "flatten",
        "ravel",
        "squeeze",
        # Indexing
        "where",
        "concatenate",
        "stack",
        "vstack",
        "hstack",
        # Random (safe operations)
        "random",
        # Constants
        "pi",
        "e",
        "inf",
        "nan",
        # Data types
        "float32",
        "float64",
        "int32",
        "int64",
        "bool_",
    }

    # Blocked functions (explicit list for clarity)
    BLOCKED_FUNCTIONS = {
        "load",
        "save",
        "loadtxt",
        "savetxt",
        "savez",
        "savez_compressed",
        "fromfile",
        "tofile",
        "genfromtxt",
        "memmap",
        "frombuffer",
    }

    def __init__(self, max_array_size: int = 10_000_000):
        """
        Initialize safe numpy wrapper.

        Args:
            max_array_size: Maximum number of elements allowed in arrays
        """
        self.max_array_size = max_array_size

    def __getattr__(self, name: str) -> Any:
        """Get numpy attribute with safety checks."""
        # Block explicitly forbidden functions
        if name in self.BLOCKED_FUNCTIONS:
            raise SecurityError(f"NumPy function '{name}' is not allowed (file I/O)")

        # Check if function is in whitelist
        if name not in self.ALLOWED_FUNCTIONS:
            raise SecurityError(
                f"NumPy function '{name}' is not in the whitelist. "
                f"Allowed functions: {', '.join(sorted(self.ALLOWED_FUNCTIONS))}"
            )

        # Get the actual numpy attribute
        attr = getattr(np, name)

        # Wrap array creation functions to enforce size limits
        if name in ("array", "zeros", "ones", "empty", "arange", "linspace"):
            return self._wrap_array_creator(attr)

        return attr

    def _wrap_array_creator(self, func):
        """Wrap array creation function to enforce size limits."""

        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                if result.size > self.max_array_size:
                    raise SecurityError(
                        f"Array size {result.size} exceeds maximum "
                        f"allowed size {self.max_array_size}"
                    )
            return result

        return wrapped

    def clip(self, a, a_min, a_max, **kwargs):
        """Safe clip operation with size validation."""
        result = np.clip(a, a_min, a_max, **kwargs)
        if isinstance(result, np.ndarray) and result.size > self.max_array_size:
            raise SecurityError(
                f"Result array size {result.size} exceeds maximum "
                f"allowed size {self.max_array_size}"
            )
        return result


class SecurityError(Exception):
    """Raised when a security constraint is violated."""
    pass


class SafeMathWrapper:
    """
    Safe wrapper for math module operations.

    Provides basic math functions without dangerous operations.
    """

    def __init__(self):
        """Initialize safe math wrapper."""
        import math

        self._math = math

    def __getattr__(self, name: str) -> Any:
        """Get math attribute with safety checks."""
        # Whitelist of safe math functions
        safe_functions = {
            "sqrt",
            "pow",
            "exp",
            "log",
            "log10",
            "log2",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "sinh",
            "cosh",
            "tanh",
            "ceil",
            "floor",
            "abs",
            "fabs",
            "factorial",
            "gcd",
            "degrees",
            "radians",
            "pi",
            "e",
            "tau",
            "inf",
            "nan",
        }

        if name not in safe_functions:
            raise SecurityError(
                f"Math function '{name}' is not in the whitelist. "
                f"Allowed functions: {', '.join(sorted(safe_functions))}"
            )

        return getattr(self._math, name)


class SafeRandomWrapper:
    """
    Safe wrapper for random operations.

    Provides controlled randomness without security risks.
    """

    def __init__(self, seed: int = None):
        """Initialize safe random wrapper."""
        self._rng = np.random.default_rng(seed)

    def random(self, size=None):
        """Generate random floats in [0, 1)."""
        return self._rng.random(size)

    def uniform(self, low=0.0, high=1.0, size=None):
        """Generate random floats from uniform distribution."""
        return self._rng.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Generate random floats from normal distribution."""
        return self._rng.normal(loc, scale, size)

    def integers(self, low, high=None, size=None):
        """Generate random integers."""
        return self._rng.integers(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        """Generate random sample from array."""
        return self._rng.choice(a, size, replace, p)

    def shuffle(self, x):
        """Shuffle array in-place."""
        return self._rng.shuffle(x)

    def seed(self, seed: int):
        """Set random seed."""
        self._rng = np.random.default_rng(seed)
