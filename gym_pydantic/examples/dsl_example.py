"""
DSL Compiler Example

Demonstrates creating safe transition functions using declarative DSL.
"""

from pydantic import Field
import numpy as np

from gym_pydantic import GymState, GymAction, create_gym_env
from gym_pydantic.security import DSLTransitionCompiler


# Define State
class PointState(GymState):
    """2D point state."""

    x: float = Field(ge=0.0, le=10.0)
    y: float = Field(ge=0.0, le=10.0)
    vx: float = Field(ge=-1.0, le=1.0)
    vy: float = Field(ge=-1.0, le=1.0)


# Define Action
class PointAction(GymAction):
    """2D acceleration."""

    ax: float = Field(ge=-0.5, le=0.5)
    ay: float = Field(ge=-0.5, le=0.5)


def main():
    """Run example."""
    print("=== DSL Compiler Example ===\n")

    # Define transition using DSL (declarative configuration)
    dsl_config = {
        "state_updates": [
            {
                "field": "vx",
                "operation": "add",
                "operands": ["vx", "ax"],
                "scale": 1.0,
                "clip": {"min": -1.0, "max": 1.0},
            },
            {
                "field": "vy",
                "operation": "add",
                "operands": ["vy", "ay"],
                "scale": 1.0,
                "clip": {"min": -1.0, "max": 1.0},
            },
            {
                "field": "x",
                "operation": "add",
                "operands": ["x", "vx"],
                "scale": 0.1,  # dt = 0.1
                "clip": {"min": 0.0, "max": 10.0},
            },
            {
                "field": "y",
                "operation": "add",
                "operands": ["y", "vy"],
                "scale": 0.1,
                "clip": {"min": 0.0, "max": 10.0},
            },
        ],
        "reward": {
            "type": "distance_to_target",
            "target": {"x": 8.0, "y": 8.0},
            "scale": -0.1,
        },
        "done": {
            "type": "goal_reached",
            "target": {"x": 8.0, "y": 8.0},
            "threshold": 0.5,
        },
    }

    print("DSL Configuration:")
    import json
    print(json.dumps(dsl_config, indent=2))

    # Compile DSL to transition function
    compiler = DSLTransitionCompiler()
    transition_fn = compiler.compile(dsl_config)

    print("\n✓ DSL compiled to transition function (no code execution!)")

    # Initial state function
    def initial_state(rng: np.random.Generator):
        return PointState(x=1.0, y=1.0, vx=0.0, vy=0.0)

    # Create environment
    DSLEnv = create_gym_env(
        state_class=PointState,
        action_class=PointAction,
        transition_fn=transition_fn,  # From DSL!
        initial_state_fn=initial_state,
        max_steps=200,
    )

    # Test the environment
    print("\n=== Testing DSL-based Environment ===")
    env = DSLEnv()
    obs, info = env.reset()
    print(f"Initial state: {obs}")

    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:
        # Simple controller
        dx = 8.0 - obs["x"][0]
        dy = 8.0 - obs["y"][0]

        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            ax = np.clip(dx / magnitude * 0.3, -0.5, 0.5)
            ay = np.clip(dy / magnitude * 0.3, -0.5, 0.5)
        else:
            ax = 0.0
            ay = 0.0

        action = {"ax": np.array([ax]), "ay": np.array([ay])}

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        if steps % 20 == 0:
            print(
                f"Step {steps}: pos=({obs['x'][0]:.2f}, {obs['y'][0]:.2f}), "
                f"vel=({obs['vx'][0]:.2f}, {obs['vy'][0]:.2f}), reward={reward:.3f}"
            )

    print(f"\nEpisode finished in {steps} steps. Total reward: {total_reward:.2f}")

    env.close()
    print("\n✓ DSL example completed!")
    print("\nSecurity benefit: Declarative configuration only, no arbitrary code!")


if __name__ == "__main__":
    main()
