"""
Nested State Example

Demonstrates automatic space derivation for nested Pydantic models.
"""

from pydantic import BaseModel, Field
import numpy as np

from gym_pydantic import GymState, GymAction, create_gym_env


# Define nested models
class Position(BaseModel):
    """Position submodel."""

    x: float = Field(ge=0.0, le=10.0)
    y: float = Field(ge=0.0, le=10.0)


class Velocity(BaseModel):
    """Velocity submodel."""

    vx: float = Field(ge=-2.0, le=2.0)
    vy: float = Field(ge=-2.0, le=2.0)


# Define State with nested models
class ComplexRobotState(GymState):
    """Robot state with nested position and velocity."""

    position: Position
    velocity: Velocity
    energy: float = Field(ge=0.0, le=100.0, description="Remaining energy")


# Define Action
class ComplexRobotAction(GymAction):
    """Robot action with force."""

    force_x: float = Field(ge=-1.0, le=1.0)
    force_y: float = Field(ge=-1.0, le=1.0)


# Define transition function
def transition(state: ComplexRobotState, action: ComplexRobotAction):
    """
    Transition function with energy management.

    Goal: Reach (8, 8) while managing energy.
    """
    dt = 0.1
    damping = 0.95

    # Extract nested values
    x = state.position.x
    y = state.position.y
    vx = state.velocity.vx
    vy = state.velocity.vy
    energy = state.energy

    # Update velocity
    vx = damping * vx + action.force_x * dt
    vy = damping * vy + action.force_y * dt

    # Clip velocity
    vx = float(np.clip(vx, -2.0, 2.0))
    vy = float(np.clip(vy, -2.0, 2.0))

    # Update position
    x = float(np.clip(x + vx * dt, 0.0, 10.0))
    y = float(np.clip(y + vy * dt, 0.0, 10.0))

    # Energy consumption (proportional to action magnitude)
    action_magnitude = np.sqrt(action.force_x**2 + action.force_y**2)
    energy_cost = action_magnitude * 0.5
    energy = float(max(0.0, energy - energy_cost))

    # Create next state with nested models
    next_state = ComplexRobotState(
        position=Position(x=x, y=y),
        velocity=Velocity(vx=vx, vy=vy),
        energy=energy,
    )

    # Calculate reward
    distance = np.sqrt((x - 8.0) ** 2 + (y - 8.0) ** 2)
    reward = -distance * 0.1

    # Penalize energy usage
    reward -= energy_cost * 0.1

    # Check termination
    done = False

    if distance < 0.5:
        reward = 10.0
        done = True
    elif energy <= 0:
        reward = -5.0  # Ran out of energy
        done = True

    # Info
    info = {
        "distance": float(distance),
        "energy": energy,
        "energy_cost": float(energy_cost),
    }

    return next_state, reward, done, info


# Initial state function
def initial_state(rng: np.random.Generator):
    """Start at (1, 1) with full energy."""
    return ComplexRobotState(
        position=Position(x=1.0, y=1.0),
        velocity=Velocity(vx=0.0, vy=0.0),
        energy=100.0,
    )


# Create environment
NestedEnv = create_gym_env(
    state_class=ComplexRobotState,
    action_class=ComplexRobotAction,
    transition_fn=transition,
    initial_state_fn=initial_state,
    max_steps=200,
    env_name="NestedStateEnv",
)


def main():
    """Run example."""
    print("Creating Nested State Environment...")

    # Create environment instance
    env = NestedEnv()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\nNote: Observation space is automatically nested (Dict of Dicts)!")

    # Run an episode
    print("\n=== Running Episode ===")
    obs, info = env.reset()
    print(f"Initial state: {obs}")

    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:
        # Simple proportional controller
        position_obs = obs["position"]
        dx = 8.0 - position_obs["x"][0]
        dy = 8.0 - position_obs["y"][0]

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

        if steps % 20 == 0:
            pos = obs["position"]
            print(
                f"Step {steps}: pos=({pos['x'][0]:.2f}, {pos['y'][0]:.2f}), "
                f"energy={obs['energy'][0]:.1f}, distance={info['distance']:.2f}"
            )

    print(f"\nEpisode finished in {steps} steps. Total reward: {total_reward:.2f}")

    env.close()
    print("\n✓ Nested State example completed!")


if __name__ == "__main__":
    main()
