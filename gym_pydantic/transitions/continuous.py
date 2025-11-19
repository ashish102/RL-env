"""Pre-built transition function for continuous control environments."""

from typing import Dict, Any, Tuple, Callable
import numpy as np

from ..base import GymState, GymAction


def create_continuous_transition(config: Dict[str, Any]) -> Callable:
    """
    Create a continuous control transition function from configuration.

    Configuration:
    - dt: float - Time step for integration (default: 0.1)
    - damping: float - Velocity damping factor (default: 0.95)
    - max_velocity: float - Maximum velocity magnitude (default: 2.0)
    - bounds: dict - Position bounds {"x_min": float, "x_max": float, ...}
    - target: dict - Target position {"x": float, "y": float, ...}
    - goal_threshold: float - Distance threshold for goal (default: 0.5)
    - goal_reward: float - Reward for reaching goal (default: 10.0)
    - distance_scale: float - Scale for distance reward (default: -0.1)
    - action_penalty: float - Penalty for large actions (default: -0.01)

    Example config:
        {
            "dt": 0.1,
            "damping": 0.95,
            "max_velocity": 2.0,
            "bounds": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10},
            "target": {"x": 8, "y": 8},
            "goal_threshold": 0.5,
            "goal_reward": 10.0,
            "distance_scale": -0.1,
            "action_penalty": -0.01
        }

    Expected state fields:
    - position: nested model with x, y fields (or direct x, y fields)
    - velocity: nested model with vx, vy fields (or direct vx, vy fields)

    Expected action fields:
    - force_x: float (or fx, or nested force.x)
    - force_y: float (or fy, or nested force.y)

    Args:
        config: Configuration dictionary

    Returns:
        Transition function
    """
    dt = config.get("dt", 0.1)
    damping = config.get("damping", 0.95)
    max_velocity = config.get("max_velocity", 2.0)
    bounds = config.get("bounds", {})
    target = config["target"]
    goal_threshold = config.get("goal_threshold", 0.5)
    goal_reward = config.get("goal_reward", 10.0)
    distance_scale = config.get("distance_scale", -0.1)
    action_penalty = config.get("action_penalty", -0.01)

    def transition(
        state: GymState, action: GymAction
    ) -> Tuple[GymState, float, bool, Dict[str, Any]]:
        """Continuous control transition function."""
        state_dict = state.model_dump()
        action_dict = action.model_dump()

        # Extract position and velocity (handle nested models)
        if "position" in state_dict:
            pos_x = state_dict["position"]["x"]
            pos_y = state_dict["position"]["y"]
        else:
            pos_x = state_dict.get("x", 0.0)
            pos_y = state_dict.get("y", 0.0)

        if "velocity" in state_dict:
            vel_x = state_dict["velocity"]["vx"]
            vel_y = state_dict["velocity"]["vy"]
        else:
            vel_x = state_dict.get("vx", 0.0)
            vel_y = state_dict.get("vy", 0.0)

        # Extract forces from action (handle nested models)
        if "force" in action_dict:
            force_x = action_dict["force"]["x"]
            force_y = action_dict["force"]["y"]
        elif "fx" in action_dict:
            force_x = action_dict["fx"]
            force_y = action_dict["fy"]
        else:
            force_x = action_dict.get("force_x", 0.0)
            force_y = action_dict.get("force_y", 0.0)

        # Physics update (simple Euler integration)
        # v_new = damping * v_old + force * dt
        vel_x = damping * vel_x + force_x * dt
        vel_y = damping * vel_y + force_y * dt

        # Clip velocity
        velocity_mag = np.sqrt(vel_x**2 + vel_y**2)
        if velocity_mag > max_velocity:
            vel_x = vel_x / velocity_mag * max_velocity
            vel_y = vel_y / velocity_mag * max_velocity

        # p_new = p_old + v_new * dt
        pos_x = pos_x + vel_x * dt
        pos_y = pos_y + vel_y * dt

        # Apply bounds
        if "x_min" in bounds:
            pos_x = np.clip(pos_x, bounds["x_min"], bounds["x_max"])
        if "y_min" in bounds:
            pos_y = np.clip(pos_y, bounds["y_min"], bounds["y_max"])

        # Update state dictionary
        if "position" in state_dict:
            state_dict["position"]["x"] = float(pos_x)
            state_dict["position"]["y"] = float(pos_y)
        else:
            state_dict["x"] = float(pos_x)
            state_dict["y"] = float(pos_y)

        if "velocity" in state_dict:
            state_dict["velocity"]["vx"] = float(vel_x)
            state_dict["velocity"]["vy"] = float(vel_y)
        else:
            state_dict["vx"] = float(vel_x)
            state_dict["vy"] = float(vel_y)

        # Create next state
        next_state = type(state)(**state_dict)

        # Calculate distance to target
        distance = np.sqrt(
            (pos_x - target["x"]) ** 2 + (pos_y - target["y"]) ** 2
        )

        # Calculate reward
        reward = distance_scale * distance

        # Add action penalty (encourage smooth control)
        action_magnitude = np.sqrt(force_x**2 + force_y**2)
        reward += action_penalty * action_magnitude

        # Check if goal reached
        done = False
        if distance < goal_threshold:
            reward = goal_reward
            done = True

        # Info
        info = {
            "distance_to_target": float(distance),
            "velocity_magnitude": float(velocity_mag),
            "action_magnitude": float(action_magnitude),
        }

        return next_state, reward, done, info

    return transition
