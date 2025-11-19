"""Tests for automatic space derivation."""

import pytest
import numpy as np
from enum import Enum
from pydantic import BaseModel, Field
from gymnasium import spaces

from gym_pydantic import GymState, GymAction
from gym_pydantic.space_derivation import (
    derive_space_from_model,
    pydantic_to_gym,
    gym_to_pydantic,
    SpaceDerivationError,
)


# Test models
class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class SimpleState(GymState):
    """Simple state for testing."""

    x: int = Field(ge=0, le=10)
    y: int = Field(ge=0, le=10)


class FloatState(GymState):
    """State with float fields."""

    position: float = Field(ge=0.0, le=1.0)
    velocity: float = Field(ge=-1.0, le=1.0)


class BoolState(GymState):
    """State with bool field."""

    active: bool


class EnumAction(GymAction):
    """Action with enum."""

    direction: Direction


class NestedPosition(BaseModel):
    """Nested position model."""

    x: float = Field(ge=0, le=10)
    y: float = Field(ge=0, le=10)


class NestedState(GymState):
    """State with nested model."""

    position: NestedPosition
    energy: float = Field(ge=0, le=100)


class TestSpaceDerivation:
    """Test space derivation from Pydantic models."""

    def test_integer_fields(self):
        """Test derivation of integer fields."""
        space = derive_space_from_model(SimpleState)

        assert isinstance(space, spaces.Dict)
        assert "x" in space.spaces
        assert "y" in space.spaces

        assert isinstance(space.spaces["x"], spaces.Box)
        assert isinstance(space.spaces["y"], spaces.Box)

    def test_float_fields(self):
        """Test derivation of float fields."""
        space = derive_space_from_model(FloatState)

        assert isinstance(space, spaces.Dict)
        assert "position" in space.spaces
        assert "velocity" in space.spaces

    def test_bool_field(self):
        """Test derivation of bool fields."""
        space = derive_space_from_model(BoolState)

        # Single field should unwrap
        assert isinstance(space, spaces.Discrete)
        assert space.n == 2

    def test_enum_field(self):
        """Test derivation of enum fields."""
        space = derive_space_from_model(EnumAction)

        assert isinstance(space, spaces.Discrete)
        assert space.n == 4  # 4 directions

    def test_nested_model(self):
        """Test derivation of nested models."""
        space = derive_space_from_model(NestedState)

        assert isinstance(space, spaces.Dict)
        assert "position" in space.spaces
        assert "energy" in space.spaces

        # Position should be nested Dict
        assert isinstance(space.spaces["position"], spaces.Dict)
        assert "x" in space.spaces["position"].spaces
        assert "y" in space.spaces["position"].spaces

    def test_pydantic_to_gym_conversion(self):
        """Test conversion from Pydantic to Gym format."""
        state = SimpleState(x=5, y=7)
        space = state.get_space()

        gym_obj = pydantic_to_gym(state, space)

        assert isinstance(gym_obj, dict)
        assert "x" in gym_obj
        assert "y" in gym_obj
        assert np.array_equal(gym_obj["x"], np.array([5]))
        assert np.array_equal(gym_obj["y"], np.array([7]))

    def test_gym_to_pydantic_conversion(self):
        """Test conversion from Gym to Pydantic format."""
        gym_obj = {"x": np.array([5]), "y": np.array([7])}
        space = SimpleState.get_space()

        state = gym_to_pydantic(gym_obj, SimpleState, space)

        assert isinstance(state, SimpleState)
        assert state.x == 5
        assert state.y == 7

    def test_round_trip_conversion(self):
        """Test round-trip conversion (Pydantic -> Gym -> Pydantic)."""
        original = SimpleState(x=3, y=8)
        space = original.get_space()

        gym_obj = pydantic_to_gym(original, space)
        converted = gym_to_pydantic(gym_obj, SimpleState, space)

        assert converted.x == original.x
        assert converted.y == original.y

    def test_enum_conversion(self):
        """Test enum conversion."""
        action = EnumAction(direction=Direction.UP)
        space = action.get_space()

        gym_obj = pydantic_to_gym(action, space)
        assert gym_obj == 0  # UP should be index 0

        converted = gym_to_pydantic(gym_obj, EnumAction, space)
        assert converted.direction == Direction.UP

    def test_missing_bounds_error(self):
        """Test that missing bounds raises error."""

        with pytest.raises(SpaceDerivationError):

            class InvalidState(GymState):
                x: float  # Missing bounds!

            derive_space_from_model(InvalidState)


class TestGymStateAction:
    """Test GymState and GymAction base classes."""

    def test_get_space(self):
        """Test get_space class method."""
        space = SimpleState.get_space()
        assert isinstance(space, spaces.Space)

    def test_to_gym(self):
        """Test to_gym instance method."""
        state = SimpleState(x=5, y=7)
        gym_obj = state.to_gym()

        assert isinstance(gym_obj, dict)
        assert "x" in gym_obj
        assert "y" in gym_obj

    def test_from_gym(self):
        """Test from_gym class method."""
        gym_obj = {"x": np.array([5]), "y": np.array([7])}
        state = SimpleState.from_gym(gym_obj)

        assert isinstance(state, SimpleState)
        assert state.x == 5
        assert state.y == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
