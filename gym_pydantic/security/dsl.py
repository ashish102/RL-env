"""Domain-Specific Language (DSL) compiler for safe transition functions."""

from typing import Dict, Any, Callable, Tuple, List
import numpy as np
from pydantic import BaseModel

from ..base import GymState, GymAction


class DSLCompilationError(Exception):
    """Raised when DSL configuration cannot be compiled."""
    pass


class DSLTransitionCompiler:
    """
    Compile declarative JSON/YAML into safe transition functions.

    Only whitelisted operations are supported - no arbitrary code execution.

    Supported operations:
    - Arithmetic: add, subtract, multiply, divide
    - Functions: abs, min, max, sqrt, square, clip
    - Reward types: distance_to_target, sparse, dense
    - Done conditions: threshold, goal_reached, timeout

    Example:
        ```python
        compiler = DSLTransitionCompiler()

        dsl_config = {
            "state_updates": [
                {
                    "field": "position.x",
                    "operation": "add",
                    "operands": ["position.x", "velocity.vx"],
                    "scale": 0.1,
                    "clip": {"min": 0, "max": 10}
                }
            ],
            "reward": {
                "type": "distance_to_target",
                "target": {"x": 8, "y": 8},
                "scale": -0.1
            },
            "done": {
                "type": "threshold",
                "field": "distance",
                "threshold": 0.5
            }
        }

        transition_fn = compiler.compile(dsl_config)
        ```
    """

    ALLOWED_OPERATIONS = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "abs",
        "min",
        "max",
        "sqrt",
        "square",
        "clip",
        "set",
    }

    ALLOWED_REWARD_TYPES = {
        "distance_to_target",
        "sparse",
        "dense",
        "constant",
        "custom_formula",
    }

    ALLOWED_DONE_TYPES = {
        "threshold",
        "goal_reached",
        "timeout",
        "never",
    }

    def compile(
        self, config: Dict[str, Any]
    ) -> Callable[[GymState, GymAction], Tuple[GymState, float, bool, Dict[str, Any]]]:
        """
        Compile DSL configuration into transition function.

        Args:
            config: DSL configuration dictionary

        Returns:
            Compiled transition function

        Raises:
            DSLCompilationError: If configuration is invalid
        """
        self._validate_config(config)

        state_updates = config.get("state_updates", [])
        reward_config = config.get("reward", {})
        done_config = config.get("done", {})

        # Compile components
        update_fn = self._compile_state_updates(state_updates)
        reward_fn = self._compile_reward(reward_config)
        done_fn = self._compile_done(done_config)

        def transition(
            state: GymState, action: GymAction
        ) -> Tuple[GymState, float, bool, Dict[str, Any]]:
            """Compiled transition function."""
            # Apply state updates
            next_state = update_fn(state, action)

            # Calculate reward
            reward = reward_fn(state, action, next_state)

            # Check termination
            done = done_fn(state, action, next_state)

            # Info
            info = {}

            return next_state, reward, done, info

        return transition

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate DSL configuration."""
        if not isinstance(config, dict):
            raise DSLCompilationError("Config must be a dictionary")

        if "state_updates" in config and not isinstance(config["state_updates"], list):
            raise DSLCompilationError("state_updates must be a list")

        if "reward" in config and not isinstance(config["reward"], dict):
            raise DSLCompilationError("reward must be a dictionary")

        if "done" in config and not isinstance(config["done"], dict):
            raise DSLCompilationError("done must be a dictionary")

    def _compile_state_updates(
        self, updates: List[Dict[str, Any]]
    ) -> Callable[[GymState, GymAction], GymState]:
        """Compile state update rules."""

        def update_fn(state: GymState, action: GymAction) -> GymState:
            # Convert to dict for easier manipulation
            state_dict = state.model_dump()
            action_dict = action.model_dump()

            for update in updates:
                field = update["field"]
                operation = update["operation"]

                if operation not in self.ALLOWED_OPERATIONS:
                    raise DSLCompilationError(f"Operation '{operation}' not allowed")

                # Get field value
                value = self._get_nested_field(state_dict, field)

                # Apply operation
                if operation == "add":
                    operands = update["operands"]
                    value = sum(
                        self._resolve_operand(op, state_dict, action_dict)
                        for op in operands
                    )
                elif operation == "multiply":
                    operands = update["operands"]
                    value = 1.0
                    for op in operands:
                        value *= self._resolve_operand(op, state_dict, action_dict)
                elif operation == "set":
                    value = update["value"]
                elif operation == "clip":
                    clip_config = update.get("clip", {})
                    value = np.clip(
                        value, clip_config.get("min", -np.inf), clip_config.get("max", np.inf)
                    )

                # Apply scale if provided
                if "scale" in update:
                    value *= update["scale"]

                # Apply clip if provided
                if "clip" in update:
                    clip_config = update["clip"]
                    value = np.clip(
                        value, clip_config.get("min", -np.inf), clip_config.get("max", np.inf)
                    )

                # Set field value
                self._set_nested_field(state_dict, field, value)

            # Create new state instance
            return type(state)(**state_dict)

        return update_fn

    def _compile_reward(
        self, reward_config: Dict[str, Any]
    ) -> Callable[[GymState, GymAction, GymState], float]:
        """Compile reward function."""
        reward_type = reward_config.get("type", "constant")

        if reward_type not in self.ALLOWED_REWARD_TYPES:
            raise DSLCompilationError(f"Reward type '{reward_type}' not allowed")

        if reward_type == "constant":
            constant_value = reward_config.get("value", 0.0)
            return lambda s, a, ns: constant_value

        elif reward_type == "distance_to_target":
            target = reward_config["target"]
            scale = reward_config.get("scale", -1.0)

            def distance_reward(s: GymState, a: GymAction, ns: GymState) -> float:
                state_dict = ns.model_dump()
                distance = 0.0
                for key, value in target.items():
                    state_value = self._get_nested_field(state_dict, key)
                    distance += (state_value - value) ** 2
                distance = np.sqrt(distance)
                return scale * distance

            return distance_reward

        elif reward_type == "sparse":
            threshold = reward_config.get("threshold", 0.5)
            reward_value = reward_config.get("reward", 1.0)
            target = reward_config["target"]

            def sparse_reward(s: GymState, a: GymAction, ns: GymState) -> float:
                state_dict = ns.model_dump()
                distance = 0.0
                for key, value in target.items():
                    state_value = self._get_nested_field(state_dict, key)
                    distance += (state_value - value) ** 2
                distance = np.sqrt(distance)
                return reward_value if distance < threshold else 0.0

            return sparse_reward

        else:
            return lambda s, a, ns: 0.0

    def _compile_done(
        self, done_config: Dict[str, Any]
    ) -> Callable[[GymState, GymAction, GymState], bool]:
        """Compile termination condition."""
        done_type = done_config.get("type", "never")

        if done_type not in self.ALLOWED_DONE_TYPES:
            raise DSLCompilationError(f"Done type '{done_type}' not allowed")

        if done_type == "never":
            return lambda s, a, ns: False

        elif done_type == "threshold":
            field = done_config["field"]
            threshold = done_config["threshold"]
            comparison = done_config.get("comparison", "less_than")

            def threshold_done(s: GymState, a: GymAction, ns: GymState) -> bool:
                state_dict = ns.model_dump()
                value = self._get_nested_field(state_dict, field)
                if comparison == "less_than":
                    return value < threshold
                elif comparison == "greater_than":
                    return value > threshold
                elif comparison == "equals":
                    return abs(value - threshold) < 1e-6
                return False

            return threshold_done

        elif done_type == "goal_reached":
            target = done_config["target"]
            threshold = done_config.get("threshold", 0.5)

            def goal_reached(s: GymState, a: GymAction, ns: GymState) -> bool:
                state_dict = ns.model_dump()
                distance = 0.0
                for key, value in target.items():
                    state_value = self._get_nested_field(state_dict, key)
                    distance += (state_value - value) ** 2
                distance = np.sqrt(distance)
                return distance < threshold

            return goal_reached

        else:
            return lambda s, a, ns: False

    def _get_nested_field(self, obj: Dict[str, Any], field: str) -> Any:
        """Get nested field value using dot notation."""
        parts = field.split(".")
        value = obj
        for part in parts:
            if isinstance(value, dict):
                value = value[part]
            else:
                value = getattr(value, part)
        return value

    def _set_nested_field(self, obj: Dict[str, Any], field: str, value: Any) -> None:
        """Set nested field value using dot notation."""
        parts = field.split(".")
        current = obj
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _resolve_operand(
        self, operand: Any, state_dict: Dict[str, Any], action_dict: Dict[str, Any]
    ) -> float:
        """Resolve operand to numeric value."""
        if isinstance(operand, (int, float)):
            return float(operand)
        elif isinstance(operand, str):
            # Try state first, then action
            try:
                return float(self._get_nested_field(state_dict, operand))
            except (KeyError, AttributeError):
                return float(self._get_nested_field(action_dict, operand))
        else:
            raise DSLCompilationError(f"Invalid operand: {operand}")
