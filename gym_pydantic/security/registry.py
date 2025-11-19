"""Pre-approved transition function registry for secure environment creation."""

from typing import Callable, Dict, Any, Tuple
from ..base import GymState, GymAction


class TransitionRegistry:
    """
    Registry of pre-approved, vetted transition functions.

    This is the safest security approach - users provide only configuration,
    not code. All transitions are pre-built and audited.

    Example:
        ```python
        registry = TransitionRegistry()

        config = {
            "grid_size": 10,
            "goal": {"x": 9, "y": 9},
            "goal_reward": 10.0,
            "step_penalty": -0.1
        }

        transition_fn = registry.get("grid_world", config)
        ```
    """

    def __init__(self):
        """Initialize the registry with built-in transitions."""
        self._transitions: Dict[str, Callable] = {}
        self._register_builtin_transitions()

    def _register_builtin_transitions(self) -> None:
        """Register all built-in transition functions."""
        # Import built-in transitions
        try:
            from ..transitions.grid_world import create_grid_transition
            from ..transitions.continuous import create_continuous_transition
            from ..transitions.physics import create_physics_transition

            self._transitions["grid_world"] = create_grid_transition
            self._transitions["continuous_control"] = create_continuous_transition
            self._transitions["physics_sim"] = create_physics_transition
        except ImportError:
            # Transitions not yet implemented
            pass

    def register(
        self,
        name: str,
        factory: Callable[[Dict[str, Any]], Callable],
    ) -> None:
        """
        Register a new transition function factory.

        Args:
            name: Unique name for this transition type
            factory: Function that takes config dict and returns transition function

        Raises:
            ValueError: If name already registered
        """
        if name in self._transitions:
            raise ValueError(f"Transition '{name}' already registered")

        self._transitions[name] = factory

    def get(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> Callable[[GymState, GymAction], Tuple[GymState, float, bool, Dict[str, Any]]]:
        """
        Get configured transition function.

        Args:
            name: Name of registered transition type
            config: Configuration dictionary

        Returns:
            Configured transition function

        Raises:
            KeyError: If transition name not found
        """
        if name not in self._transitions:
            raise KeyError(
                f"Transition '{name}' not found. "
                f"Available: {', '.join(self.list_available())}"
            )

        factory = self._transitions[name]
        return factory(config)

    def list_available(self) -> list[str]:
        """
        List all available transition types.

        Returns:
            List of registered transition names
        """
        return list(self._transitions.keys())

    def get_config_schema(self, name: str) -> Dict[str, Any]:
        """
        Get configuration schema for a transition type.

        Args:
            name: Name of registered transition type

        Returns:
            JSON schema for configuration

        Raises:
            KeyError: If transition name not found
        """
        if name not in self._transitions:
            raise KeyError(f"Transition '{name}' not found")

        # This would be implemented by each transition factory
        # For now, return a basic schema
        return {
            "type": "object",
            "description": f"Configuration for {name} transition",
        }


# Global registry instance
_global_registry = TransitionRegistry()


def register_transition(name: str) -> Callable:
    """
    Decorator to register a transition factory with the global registry.

    Example:
        ```python
        @register_transition("my_transition")
        def create_my_transition(config: Dict[str, Any]):
            def transition(state, action):
                # Implementation
                ...
            return transition
        ```
    """

    def decorator(factory: Callable) -> Callable:
        _global_registry.register(name, factory)
        return factory

    return decorator


def get_global_registry() -> TransitionRegistry:
    """Get the global transition registry instance."""
    return _global_registry
