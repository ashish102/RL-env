"""Generic environment factory for creating Gymnasium environments from Pydantic models."""

from typing import Callable, Dict, Optional, Tuple, Type, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import GymState, GymAction


def create_gym_env(
    state_class: Type[GymState],
    action_class: Type[GymAction],
    transition_fn: Callable[
        [GymState, GymAction], Tuple[GymState, float, bool, Dict[str, Any]]
    ],
    initial_state_fn: Callable[[Optional[np.random.Generator]], GymState],
    max_steps: int = 1000,
    render_fn: Optional[Callable[[GymState, int], None]] = None,
    env_name: str = "CustomEnv",
) -> Type[gym.Env]:
    """
    Create a Gymnasium-compatible environment from Pydantic models and transition function.

    Args:
        state_class: Pydantic model class defining the state space
        action_class: Pydantic model class defining the action space
        transition_fn: Function (state, action) -> (next_state, reward, done, info)
        initial_state_fn: Function (rng) -> initial_state
        max_steps: Maximum steps before truncation (default: 1000)
        render_fn: Optional rendering function (state, step) -> None
        env_name: Name for the environment class (default: "CustomEnv")

    Returns:
        Gymnasium environment class ready to instantiate

    Example:
        ```python
        from pydantic import Field
        from enum import Enum

        class Direction(str, Enum):
            UP = "up"
            DOWN = "down"

        class GridState(GymState):
            x: int = Field(ge=0, le=9)
            y: int = Field(ge=0, le=9)

        class GridAction(GymAction):
            direction: Direction

        def transition(state, action):
            # Implement transition logic
            ...
            return next_state, reward, done, info

        GridEnv = create_gym_env(
            state_class=GridState,
            action_class=GridAction,
            transition_fn=transition,
            initial_state_fn=lambda rng: GridState(x=0, y=0),
            max_steps=100
        )

        env = GridEnv()
        model = DQN("MultiInputPolicy", env)
        model.learn(50000)
        ```
    """

    class CustomGymEnv(gym.Env):
        """Dynamically created Gymnasium environment."""

        metadata = {"render_modes": ["human"], "render_fps": 30}

        def __init__(self, render_mode: Optional[str] = None):
            """Initialize the environment."""
            super().__init__()

            # Store configuration
            self._state_class = state_class
            self._action_class = action_class
            self._transition_fn = transition_fn
            self._initial_state_fn = initial_state_fn
            self._max_steps = max_steps
            self._render_fn = render_fn
            self.render_mode = render_mode

            # Automatically derive spaces
            self.observation_space = state_class.get_space()
            self.action_space = action_class.get_space()

            # Internal state
            self._current_state: Optional[GymState] = None
            self._step_count: int = 0
            self._rng: np.random.Generator = np.random.default_rng()

        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Tuple[Any, Dict[str, Any]]:
            """
            Reset the environment to initial state.

            Args:
                seed: Random seed
                options: Additional options

            Returns:
                (observation, info) tuple
            """
            super().reset(seed=seed)

            if seed is not None:
                self._rng = np.random.default_rng(seed)

            # Get initial state
            self._current_state = self._initial_state_fn(self._rng)
            self._step_count = 0

            # Convert to gym format
            observation = self._current_state.to_gym()

            info = {"step": self._step_count}

            if self.render_mode == "human" and self._render_fn is not None:
                self._render_fn(self._current_state, self._step_count)

            return observation, info

        def step(
            self, action: Any
        ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
            """
            Execute one step in the environment.

            Args:
                action: Action in gymnasium format

            Returns:
                (observation, reward, terminated, truncated, info) tuple
            """
            if self._current_state is None:
                raise RuntimeError("Environment not initialized. Call reset() first.")

            # Convert action from gym format to Pydantic
            action_obj = self._action_class.from_gym(action)

            # Execute transition function
            next_state, reward, done, info = self._transition_fn(
                self._current_state, action_obj
            )

            # Update state
            self._current_state = next_state
            self._step_count += 1

            # Check truncation
            truncated = self._step_count >= self._max_steps
            terminated = done

            # Convert observation to gym format
            observation = self._current_state.to_gym()

            # Add step count to info
            info["step"] = self._step_count

            if self.render_mode == "human" and self._render_fn is not None:
                self._render_fn(self._current_state, self._step_count)

            return observation, float(reward), terminated, truncated, info

        def render(self) -> Optional[np.ndarray]:
            """
            Render the environment.

            Returns:
                Rendered frame if applicable
            """
            if self._render_fn is not None and self._current_state is not None:
                self._render_fn(self._current_state, self._step_count)
            return None

        def close(self) -> None:
            """Clean up resources."""
            pass

    # Set the class name
    CustomGymEnv.__name__ = env_name
    CustomGymEnv.__qualname__ = env_name

    return CustomGymEnv
