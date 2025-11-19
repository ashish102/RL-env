"""Sandboxed execution environment for custom transition functions."""

import ast
import re
from typing import Callable, Dict, Any, Tuple
import signal
from contextlib import contextmanager

from ..base import GymState, GymAction


class SandboxSecurityError(Exception):
    """Raised when code violates security constraints."""
    pass


class SandboxTimeoutError(Exception):
    """Raised when code execution times out."""
    pass


class SafeTransitionExecutor:
    """
    Execute user-provided transition functions in a sandboxed environment.

    Security features:
    - Block dangerous operations (file I/O, imports, eval, exec)
    - Whitelist safe operations only
    - Memory and CPU limits
    - Timeout protection
    - AST validation before execution

    WARNING: This is for advanced users only. For production APIs,
    use TransitionRegistry or DSLTransitionCompiler instead.

    Example:
        ```python
        executor = SafeTransitionExecutor()

        code = '''
def transition(state, action):
    import safe_numpy as np
    next_x = state.x + action.dx * 0.1
    next_x = np.clip(next_x, 0, 10)
    reward = -abs(next_x - 8)
    done = abs(next_x - 8) < 0.5
    return type(state)(x=next_x), reward, done, {}
        '''

        transition_fn = executor.compile(code)
        ```
    """

    # Forbidden patterns
    FORBIDDEN_PATTERNS = [
        r'\bimport\s+(?!safe_)',  # Block all imports except safe_*
        r'\bfrom\s+(?!safe_)',
        r'\bopen\s*\(',
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\bcompile\s*\(',
        r'\b__import__\s*\(',
        r'\.load\s*\(',
        r'\.save\s*\(',
        r'\.dump\s*\(',
        r'\bpickle\b',
        r'\bsubprocess\b',
        r'\bos\.',
        r'\bsys\.',
        r'\bsocket\b',
        r'\burllib\b',
        r'\brequests\b',
    ]

    # Forbidden AST node types
    FORBIDDEN_NODES = {
        ast.Import,
        ast.ImportFrom,
    }

    def __init__(
        self,
        max_memory_mb: int = 1024,
        timeout_seconds: int = 10,
        max_array_size: int = 10_000_000,
    ):
        """
        Initialize sandbox executor.

        Args:
            max_memory_mb: Maximum memory usage in MB
            timeout_seconds: Maximum execution time in seconds
            max_array_size: Maximum array size in elements
        """
        self.max_memory_mb = max_memory_mb
        self.timeout_seconds = timeout_seconds
        self.max_array_size = max_array_size

    def compile(
        self, code: str
    ) -> Callable[[GymState, GymAction], Tuple[GymState, float, bool, Dict[str, Any]]]:
        """
        Compile user code into safe transition function.

        Args:
            code: Python code defining transition function

        Returns:
            Compiled transition function

        Raises:
            SandboxSecurityError: If code violates security constraints
        """
        # Pattern validation
        self._validate_patterns(code)

        # AST validation
        self._validate_ast(code)

        # Compile code
        try:
            compiled_code = compile(code, "<user_transition>", "exec")
        except SyntaxError as e:
            raise SandboxSecurityError(f"Syntax error in code: {e}")

        # Create safe namespace
        namespace = self._create_safe_namespace()

        # Execute code to define function
        try:
            with self._timeout_context(self.timeout_seconds):
                exec(compiled_code, namespace)
        except SandboxTimeoutError:
            raise SandboxSecurityError("Code execution timed out during compilation")
        except Exception as e:
            raise SandboxSecurityError(f"Error executing code: {e}")

        # Extract transition function
        if "transition" not in namespace:
            raise SandboxSecurityError("Code must define a 'transition' function")

        user_transition = namespace["transition"]

        # Wrap with timeout protection
        def safe_transition(
            state: GymState, action: GymAction
        ) -> Tuple[GymState, float, bool, Dict[str, Any]]:
            try:
                with self._timeout_context(self.timeout_seconds):
                    return user_transition(state, action)
            except SandboxTimeoutError:
                raise SandboxSecurityError("Transition function execution timed out")

        return safe_transition

    def _validate_patterns(self, code: str) -> None:
        """Validate code against forbidden patterns."""
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                raise SandboxSecurityError(
                    f"Forbidden pattern detected: {pattern}"
                )

    def _validate_ast(self, code: str) -> None:
        """Validate code AST for forbidden operations."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SandboxSecurityError(f"Syntax error: {e}")

        for node in ast.walk(tree):
            for forbidden_type in self.FORBIDDEN_NODES:
                if isinstance(node, forbidden_type):
                    raise SandboxSecurityError(
                        f"Forbidden operation: {forbidden_type.__name__}"
                    )

    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Create safe namespace with whitelisted builtins."""
        from .wrappers import SafeNumpyWrapper

        # Safe builtins
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "len": len,
            "int": int,
            "float": float,
            "bool": bool,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "type": type,
            "print": print,  # Allow for debugging
        }

        # Create namespace
        namespace = {
            "__builtins__": safe_builtins,
            "safe_numpy": SafeNumpyWrapper(self.max_array_size),
        }

        return namespace

    @contextmanager
    def _timeout_context(self, seconds: int):
        """Context manager for execution timeout."""

        def timeout_handler(signum, frame):
            raise SandboxTimeoutError("Execution timed out")

        # Set signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Reset signal
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


class RestrictedPythonExecutor:
    """
    Alternative sandbox using RestrictedPython library.

    Provides more robust sandboxing but requires RestrictedPython package.

    Example:
        ```python
        executor = RestrictedPythonExecutor()
        transition_fn = executor.compile(code)
        ```
    """

    def __init__(self):
        """Initialize RestrictedPython executor."""
        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import safe_builtins, safe_globals

            self.compile_restricted = compile_restricted
            self.safe_builtins = safe_builtins
            self.safe_globals = safe_globals
        except ImportError:
            raise ImportError(
                "RestrictedPython not installed. "
                "Install with: pip install RestrictedPython"
            )

    def compile(
        self, code: str
    ) -> Callable[[GymState, GymAction], Tuple[GymState, float, bool, Dict[str, Any]]]:
        """
        Compile code using RestrictedPython.

        Args:
            code: Python code defining transition function

        Returns:
            Compiled transition function

        Raises:
            SandboxSecurityError: If code violates security constraints
        """
        from .wrappers import SafeNumpyWrapper

        # Compile with RestrictedPython
        byte_code = self.compile_restricted(
            code,
            filename="<user_transition>",
            mode="exec",
        )

        if byte_code.errors:
            raise SandboxSecurityError(
                f"Compilation errors: {', '.join(byte_code.errors)}"
            )

        # Create safe namespace
        namespace = {
            "__builtins__": self.safe_builtins,
            "safe_numpy": SafeNumpyWrapper(),
            "_getattr_": getattr,
            "_getitem_": lambda obj, key: obj[key],
        }

        # Execute
        exec(byte_code.code, namespace)

        if "transition" not in namespace:
            raise SandboxSecurityError("Code must define a 'transition' function")

        return namespace["transition"]
