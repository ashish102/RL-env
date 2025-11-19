"""
Simple Grid World Example

Demonstrates basic usage of gym-pydantic with a simple grid world environment.
"""

from enum import Enum
from pydantic import Field
import numpy as np

from gym_pydantic import GymState, GymAction, create_gym_env


# Define Direction enum
class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


# Define State using Pydantic
class GridState(GymState):
    """Grid world state with position."""

    x: int = Field(ge=0, le=9, description="X coordinate")
    y: int = Field(ge=0, le=9, description="Y coordinate")


# Define Action using Pydantic
class GridAction(GymAction):
    """Grid world action - move in a direction."""

    direction: Direction


# Define transition function
def transition(state: GridState, action: GridAction):
    """
    Transition function for grid world.

    Goal: Reach position (9, 9)
    Reward: -0.1 per step, +10 for reaching goal
    """
    # Get current position
    x, y = state.x, state.y

    # Apply action
    if action.direction == Direction.UP:
        y = min(y + 1, 9)
    elif action.direction == Direction.DOWN:
        y = max(y - 1, 0)
    elif action.direction == Direction.LEFT:
        x = max(x - 1, 0)
    elif action.direction == Direction.RIGHT:
        x = min(x + 1, 9)

    # Create next state
    next_state = GridState(x=x, y=y)

    # Calculate reward
    if x == 9 and y == 9:
        reward = 10.0
        done = True
    else:
        reward = -0.1
        done = False

    # Info
    info = {"distance_to_goal": abs(9 - x) + abs(9 - y)}

    return next_state, reward, done, info


# Initial state function
def initial_state(rng: np.random.Generator):
    """Start at (0, 0)."""
    return GridState(x=0, y=0)


# Create environment
GridEnv = create_gym_env(
    state_class=GridState,
    action_class=GridAction,
    transition_fn=transition,
    initial_state_fn=initial_state,
    max_steps=100,
    env_name="SimpleGridEnv",
)


def main():
    """Run example."""
    print("Creating Simple Grid World Environment...")

    # Create environment instance
    env = GridEnv()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run a few episodes
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        obs, info = env.reset()
        print(f"Initial state: {obs}")

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 20:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            print(f"Step {steps}: action={action}, obs={obs}, reward={reward:.2f}")

        print(f"Episode finished. Total reward: {total_reward:.2f}")

    env.close()
    print("\n✓ Simple Grid World example completed!")


if __name__ == "__main__":
    main()
