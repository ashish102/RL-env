"""
Continuous Control Example

Demonstrates continuous state and action spaces with physics-based control.
"""

from pydantic import Field
import numpy as np

from gym_pydantic import GymState, GymAction, create_gym_env


# Define State
class RobotState(GymState):
    """2D robot state with position and velocity."""

    x: float = Field(ge=0.0, le=10.0, description="X position")
    y: float = Field(ge=0.0, le=10.0, description="Y position")
    vx: float = Field(ge=-2.0, le=2.0, description="X velocity")
    vy: float = Field(ge=-2.0, le=2.0, description="Y velocity")


# Define Action
class RobotAction(GymAction):
    """2D force applied to robot."""

    force_x: float = Field(ge=-1.0, le=1.0, description="X force")
    force_y: float = Field(ge=-1.0, le=1.0, description="Y force")


# Define transition function
def transition(state: RobotState, action: RobotAction):
    """
    Physics-based transition function.

    Goal: Navigate to (8, 8) with minimal energy expenditure.
    """
    dt = 0.1  # Time step
    damping = 0.95  # Velocity damping

    # Update velocity (F = ma, assume m = 1)
    vx = damping * state.vx + action.force_x * dt
    vy = damping * state.vy + action.force_y * dt

    # Clip velocity
    vx = np.clip(vx, -2.0, 2.0)
    vy = np.clip(vy, -2.0, 2.0)

    # Update position
    x = state.x + vx * dt
    y = state.y + vy * dt

    # Clip position to bounds
    x = np.clip(x, 0.0, 10.0)
    y = np.clip(y, 0.0, 10.0)

    # Create next state
    next_state = RobotState(x=x, y=y, vx=vx, vy=vy)

    # Calculate reward
    distance = np.sqrt((x - 8.0) ** 2 + (y - 8.0) ** 2)
    reward = -distance * 0.1  # Distance penalty

    # Action penalty (encourage energy efficiency)
    action_magnitude = np.sqrt(action.force_x**2 + action.force_y**2)
    reward -= 0.01 * action_magnitude

    # Check if goal reached
    done = distance < 0.5

    if done:
        reward = 10.0  # Goal bonus

    # Info
    info = {
        "distance": float(distance),
        "velocity": float(np.sqrt(vx**2 + vy**2)),
    }

    return next_state, reward, done, info


# Initial state function
def initial_state(rng: np.random.Generator):
    """Start at random position near (1, 1)."""
    x = rng.uniform(0.5, 1.5)
    y = rng.uniform(0.5, 1.5)
    return RobotState(x=x, y=y, vx=0.0, vy=0.0)


# Create environment
ContinuousEnv = create_gym_env(
    state_class=RobotState,
    action_class=RobotAction,
    transition_fn=transition,
    initial_state_fn=initial_state,
    max_steps=200,
    env_name="ContinuousRobotEnv",
)


def main():
    """Run example."""
    print("Creating Continuous Control Environment...")

    # Create environment instance
    env = ContinuousEnv()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run an episode with a simple controller
    print("\n=== Running Episode with Simple Controller ===")
    obs, info = env.reset()
    print(f"Initial state: {obs}")

    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:
        # Simple proportional controller towards goal
        state_dict = obs
        dx = 8.0 - state_dict["x"][0]
        dy = 8.0 - state_dict["y"][0]

        # Normalize and scale
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            force_x = np.clip(dx / magnitude * 0.5, -1.0, 1.0)
            force_y = np.clip(dy / magnitude * 0.5, -1.0, 1.0)
        else:
            force_x = 0.0
            force_y = 0.0

        action = {"force_x": np.array([force_x]), "force_y": np.array([force_y])}

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        if steps % 10 == 0:
            print(
                f"Step {steps}: pos=({state_dict['x'][0]:.2f}, {state_dict['y'][0]:.2f}), "
                f"distance={info['distance']:.2f}, reward={reward:.3f}"
            )

    print(f"\nEpisode finished in {steps} steps. Total reward: {total_reward:.2f}")

    env.close()
    print("\n✓ Continuous Control example completed!")


if __name__ == "__main__":
    main()
