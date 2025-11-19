"""Tests for environment factory."""

import pytest
import numpy as np
from enum import Enum
from pydantic import Field
import gymnasium as gym

from gym_pydantic import GymState, GymAction, create_gym_env


# Test models
class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


class TestState(GymState):
    x: int = Field(ge=0, le=10)
    y: int = Field(ge=0, le=10)


class TestAction(GymAction):
    direction: Direction


def simple_transition(state, action):
    """Simple transition for testing."""
    x = state.x
    y = state.y

    if action.direction == Direction.UP:
        y = min(y + 1, 10)
    else:
        y = max(y - 1, 0)

    next_state = TestState(x=x, y=y)
    reward = -0.1
    done = (x == 10 and y == 10)

    return next_state, reward, done, {}


def initial_state(rng):
    """Initial state function."""
    return TestState(x=0, y=0)


class TestEnvironmentFactory:
    """Test environment creation."""

    def test_create_env(self):
        """Test basic environment creation."""
        EnvClass = create_gym_env(
            state_class=TestState,
            action_class=TestAction,
            transition_fn=simple_transition,
            initial_state_fn=initial_state,
            max_steps=100,
        )

        assert issubclass(EnvClass, gym.Env)

    def test_env_spaces(self):
        """Test that spaces are correctly set."""
        EnvClass = create_gym_env(
            state_class=TestState,
            action_class=TestAction,
            transition_fn=simple_transition,
            initial_state_fn=initial_state,
        )

        env = EnvClass()

        assert env.observation_space is not None
        assert env.action_space is not None

    def test_env_reset(self):
        """Test environment reset."""
        EnvClass = create_gym_env(
            state_class=TestState,
            action_class=TestAction,
            transition_fn=simple_transition,
            initial_state_fn=initial_state,
        )

        env = EnvClass()
        obs, info = env.reset()

        assert obs is not None
        assert isinstance(info, dict)

    def test_env_step(self):
        """Test environment step."""
        EnvClass = create_gym_env(
            state_class=TestState,
            action_class=TestAction,
            transition_fn=simple_transition,
            initial_state_fn=initial_state,
        )

        env = EnvClass()
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_env_truncation(self):
        """Test that environment truncates at max_steps."""
        EnvClass = create_gym_env(
            state_class=TestState,
            action_class=TestAction,
            transition_fn=simple_transition,
            initial_state_fn=initial_state,
            max_steps=10,
        )

        env = EnvClass()
        env.reset()

        truncated = False
        for i in range(15):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)

            if truncated:
                assert i >= 9  # Should truncate at step 10
                break

        assert truncated

    def test_env_episode(self):
        """Test complete episode."""
        EnvClass = create_gym_env(
            state_class=TestState,
            action_class=TestAction,
            transition_fn=simple_transition,
            initial_state_fn=initial_state,
            max_steps=100,
        )

        env = EnvClass()
        obs, info = env.reset()

        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        assert steps > 0
        assert isinstance(total_reward, (int, float))

    def test_env_custom_name(self):
        """Test custom environment name."""
        EnvClass = create_gym_env(
            state_class=TestState,
            action_class=TestAction,
            transition_fn=simple_transition,
            initial_state_fn=initial_state,
            env_name="CustomTestEnv",
        )

        assert EnvClass.__name__ == "CustomTestEnv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
