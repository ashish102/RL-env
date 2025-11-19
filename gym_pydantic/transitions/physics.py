"""Pre-built transition function for simple physics simulation environments."""

from typing import Dict, Any, Tuple, Callable
import numpy as np

from ..base import GymState, GymAction


def create_physics_transition(config: Dict[str, Any]) -> Callable:
    """
    Create a physics simulation transition function from configuration.

    Simulates basic physics with gravity, friction, and collisions.

    Configuration:
    - dt: float - Time step for integration (default: 0.1)
    - gravity: float - Gravitational acceleration (default: 9.81)
    - friction: float - Friction coefficient (default: 0.1)
    - mass: float - Object mass (default: 1.0)
    - bounds: dict - Position bounds {"x_min": float, "x_max": float, "y_min": float, "y_max": float}
    - restitution: float - Bounce coefficient (default: 0.8)
    - target: dict - Optional target position for reward
    - goal_threshold: float - Distance threshold for goal (default: 0.5)
    - goal_reward: float - Reward for reaching goal (default: 10.0)
    - survival_reward: float - Reward per timestep (default: 0.1)

    Example config:
        {
            "dt": 0.05,
            "gravity": 9.81,
            "friction": 0.1,
            "mass": 1.0,
            "bounds": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10},
            "restitution": 0.8,
            "target": {"x": 8, "y": 8},
            "goal_threshold": 0.5,
            "goal_reward": 10.0,
            "survival_reward": 0.1
        }

    Expected state fields:
    - x, y: float - Position
    - vx, vy: float - Velocity

    Expected action fields:
    - force_x, force_y: float - Applied forces

    Args:
        config: Configuration dictionary

    Returns:
        Transition function
    """
    dt = config.get("dt", 0.1)
    gravity = config.get("gravity", 9.81)
    friction = config.get("friction", 0.1)
    mass = config.get("mass", 1.0)
    bounds = config.get("bounds", {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10})
    restitution = config.get("restitution", 0.8)
    target = config.get("target")
    goal_threshold = config.get("goal_threshold", 0.5)
    goal_reward = config.get("goal_reward", 10.0)
    survival_reward = config.get("survival_reward", 0.1)

    def transition(
        state: GymState, action: GymAction
    ) -> Tuple[GymState, float, bool, Dict[str, Any]]:
        """Physics simulation transition function."""
        state_dict = state.model_dump()
        action_dict = action.model_dump()

        # Extract state variables
        x = state_dict.get("x", 0.0)
        y = state_dict.get("y", 0.0)
        vx = state_dict.get("vx", 0.0)
        vy = state_dict.get("vy", 0.0)

        # Extract action forces
        force_x = action_dict.get("force_x", 0.0)
        force_y = action_dict.get("force_y", 0.0)

        # Calculate acceleration: F = ma => a = F/m
        ax = force_x / mass
        ay = force_y / mass - gravity  # Add gravity in y direction

        # Apply friction (proportional to velocity)
        friction_x = -friction * vx
        friction_y = -friction * vy
        ax += friction_x
        ay += friction_y

        # Update velocity: v_new = v_old + a * dt
        vx = vx + ax * dt
        vy = vy + ay * dt

        # Update position: p_new = p_old + v * dt
        x = x + vx * dt
        y = y + vy * dt

        # Handle boundary collisions
        collided = False

        if x < bounds["x_min"]:
            x = bounds["x_min"]
            vx = -vx * restitution
            collided = True
        elif x > bounds["x_max"]:
            x = bounds["x_max"]
            vx = -vx * restitution
            collided = True

        if y < bounds["y_min"]:
            y = bounds["y_min"]
            vy = -vy * restitution
            collided = True
        elif y > bounds["y_max"]:
            y = bounds["y_max"]
            vy = -vy * restitution
            collided = True

        # Update state dictionary
        state_dict["x"] = float(x)
        state_dict["y"] = float(y)
        state_dict["vx"] = float(vx)
        state_dict["vy"] = float(vy)

        # Create next state
        next_state = type(state)(**state_dict)

        # Calculate reward
        reward = survival_reward

        done = False

        # If target specified, calculate distance-based reward
        if target is not None:
            distance = np.sqrt((x - target["x"]) ** 2 + (y - target["y"]) ** 2)

            if distance < goal_threshold:
                reward = goal_reward
                done = True

        # Calculate kinetic energy and potential energy
        kinetic_energy = 0.5 * mass * (vx**2 + vy**2)
        potential_energy = mass * gravity * y

        # Info
        info = {
            "collided": collided,
            "kinetic_energy": float(kinetic_energy),
            "potential_energy": float(potential_energy),
            "total_energy": float(kinetic_energy + potential_energy),
        }

        if target is not None:
            info["distance_to_target"] = float(distance)

        return next_state, reward, done, info

    return transition
