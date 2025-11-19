"""Pre-built transition function for discrete grid world environments."""

from typing import Dict, Any, Tuple, Callable
from enum import Enum
import numpy as np

from ..base import GymState, GymAction


def create_grid_transition(config: Dict[str, Any]) -> Callable:
    """
    Create a grid world transition function from configuration.

    Configuration:
    - grid_size: int - Size of the square grid (default: 10)
    - goal: dict - Goal position {"x": int, "y": int}
    - goal_reward: float - Reward for reaching goal (default: 10.0)
    - step_penalty: float - Penalty per step (default: -0.1)
    - walls: list - List of wall positions [{"x": int, "y": int}, ...]
    - wall_penalty: float - Penalty for hitting wall (default: -1.0)

    Example config:
        {
            "grid_size": 10,
            "goal": {"x": 9, "y": 9},
            "goal_reward": 10.0,
            "step_penalty": -0.1,
            "walls": [{"x": 5, "y": 5}, {"x": 5, "y": 6}],
            "wall_penalty": -1.0
        }

    Expected state fields: x: int, y: int
    Expected action field: direction: Enum (with UP, DOWN, LEFT, RIGHT values)

    Args:
        config: Configuration dictionary

    Returns:
        Transition function
    """
    grid_size = config.get("grid_size", 10)
    goal = config["goal"]
    goal_reward = config.get("goal_reward", 10.0)
    step_penalty = config.get("step_penalty", -0.1)
    walls = config.get("walls", [])
    wall_penalty = config.get("wall_penalty", -1.0)

    # Convert walls to set for fast lookup
    wall_set = {(w["x"], w["y"]) for w in walls}

    def transition(
        state: GymState, action: GymAction
    ) -> Tuple[GymState, float, bool, Dict[str, Any]]:
        """Grid world transition function."""
        # Get current position
        current_x = state.x
        current_y = state.y

        # Get action direction
        direction = action.direction

        # Calculate next position based on direction
        next_x = current_x
        next_y = current_y

        # Convert enum to string for comparison
        dir_str = direction.value if isinstance(direction, Enum) else str(direction).lower()

        if dir_str == "up":
            next_y = current_y + 1
        elif dir_str == "down":
            next_y = current_y - 1
        elif dir_str == "left":
            next_x = current_x - 1
        elif dir_str == "right":
            next_x = current_x + 1

        # Check boundaries
        next_x = np.clip(next_x, 0, grid_size - 1)
        next_y = np.clip(next_y, 0, grid_size - 1)

        # Check walls
        hit_wall = (next_x, next_y) in wall_set
        if hit_wall:
            # Stay in place if hit wall
            next_x = current_x
            next_y = current_y

        # Create next state (preserve other fields if they exist)
        state_dict = state.model_dump()
        state_dict["x"] = int(next_x)
        state_dict["y"] = int(next_y)
        next_state = type(state)(**state_dict)

        # Calculate reward
        reward = step_penalty

        if hit_wall:
            reward += wall_penalty

        # Check if reached goal
        done = False
        if next_x == goal["x"] and next_y == goal["y"]:
            reward = goal_reward
            done = True

        # Info
        info = {
            "distance_to_goal": abs(next_x - goal["x"]) + abs(next_y - goal["y"]),
            "hit_wall": hit_wall,
        }

        return next_state, reward, done, info

    return transition
