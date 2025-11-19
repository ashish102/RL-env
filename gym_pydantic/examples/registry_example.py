"""
Transition Registry Example

Demonstrates using pre-approved transitions from the registry for security.
"""

from enum import Enum
from pydantic import Field
import numpy as np

from gym_pydantic import GymState, GymAction, create_gym_env
from gym_pydantic.security import get_global_registry


# Define Direction enum
class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


# Define State
class GridState(GymState):
    """Grid world state."""

    x: int = Field(ge=0, le=9)
    y: int = Field(ge=0, le=9)


# Define Action
class GridAction(GymAction):
    """Grid world action."""

    direction: Direction


def main():
    """Run example."""
    print("=== Transition Registry Example ===\n")

    # Get the global registry
    registry = get_global_registry()

    # List available transitions
    print("Available transitions:")
    for name in registry.list_available():
        print(f"  - {name}")

    # Configure grid world transition
    config = {
        "grid_size": 10,
        "goal": {"x": 9, "y": 9},
        "goal_reward": 10.0,
        "step_penalty": -0.1,
        "walls": [
            {"x": 5, "y": 5},
            {"x": 5, "y": 6},
            {"x": 5, "y": 7},
        ],
        "wall_penalty": -1.0,
    }

    print(f"\nConfiguration: {config}")

    # Get transition function from registry
    transition_fn = registry.get("grid_world", config)

    print("\n✓ Transition function retrieved from registry (no custom code needed!)")

    # Initial state function
    def initial_state(rng: np.random.Generator):
        return GridState(x=0, y=0)

    # Create environment
    GridEnv = create_gym_env(
        state_class=GridState,
        action_class=GridAction,
        transition_fn=transition_fn,  # From registry!
        initial_state_fn=initial_state,
        max_steps=100,
    )

    # Test the environment
    print("\n=== Testing Environment ===")
    env = GridEnv()
    obs, info = env.reset()
    print(f"Initial state: {obs}")

    # Try to move into a wall
    print("\nTrying to move right (into wall at x=5)...")
    for _ in range(5):
        action = 3  # RIGHT
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Position: ({obs['x']}, {obs['y']}), reward: {reward:.2f}, hit_wall: {info['hit_wall']}")

    env.close()
    print("\n✓ Registry example completed!")
    print("\nSecurity benefit: Users provide only configuration, not code!")


if __name__ == "__main__":
    main()
