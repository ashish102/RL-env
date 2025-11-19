"""Base classes for Pydantic-based Gymnasium states and actions."""

from abc import ABC
from typing import Any, Type, TypeVar
from pydantic import BaseModel
from gymnasium import spaces

from .space_derivation import (
    derive_space_from_model,
    pydantic_to_gym,
    gym_to_pydantic,
)


T = TypeVar('T', bound='GymState')
A = TypeVar('A', bound='GymAction')


class GymState(BaseModel, ABC):
    """
    Base class for Gymnasium-compatible state models.

    Automatically derives observation space from Pydantic field specifications.

    Example:
        ```python
        class MyState(GymState):
            x: float = Field(ge=0, le=10)
            y: float = Field(ge=0, le=10)
            health: int = Field(ge=0, le=100)

        # Automatically creates spaces.Dict({
        #     'x': spaces.Box(0, 10, shape=(1,)),
        #     'y': spaces.Box(0, 10, shape=(1,)),
        #     'health': spaces.Box(0, 100, shape=(1,), dtype=int64)
        # })
        ```
    """

    @classmethod
    def get_space(cls) -> spaces.Space:
        """
        Automatically derive Gymnasium observation space from model fields.

        Returns:
            Gymnasium Space object

        Raises:
            SpaceDerivationError: If space cannot be derived from fields
        """
        return derive_space_from_model(cls)

    def to_gym(self) -> Any:
        """
        Convert this state instance to Gymnasium format.

        Returns:
            Data in format expected by the observation space
        """
        space = self.get_space()
        return pydantic_to_gym(self, space)

    @classmethod
    def from_gym(cls: Type[T], gym_obj: Any) -> T:
        """
        Convert Gymnasium format data to state instance.

        Args:
            gym_obj: Data from Gymnasium environment

        Returns:
            State instance
        """
        space = cls.get_space()
        return gym_to_pydantic(gym_obj, cls, space)

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True


class GymAction(BaseModel, ABC):
    """
    Base class for Gymnasium-compatible action models.

    Automatically derives action space from Pydantic field specifications.

    Example:
        ```python
        from enum import Enum

        class Direction(str, Enum):
            UP = "up"
            DOWN = "down"
            LEFT = "left"
            RIGHT = "right"

        class MyAction(GymAction):
            direction: Direction

        # Automatically creates spaces.Discrete(4)
        ```
    """

    @classmethod
    def get_space(cls) -> spaces.Space:
        """
        Automatically derive Gymnasium action space from model fields.

        Returns:
            Gymnasium Space object

        Raises:
            SpaceDerivationError: If space cannot be derived from fields
        """
        return derive_space_from_model(cls)

    def to_gym(self) -> Any:
        """
        Convert this action instance to Gymnasium format.

        Returns:
            Data in format expected by the action space
        """
        space = self.get_space()
        return pydantic_to_gym(self, space)

    @classmethod
    def from_gym(cls: Type[A], gym_obj: Any) -> A:
        """
        Convert Gymnasium format data to action instance.

        Args:
            gym_obj: Data from Gymnasium environment

        Returns:
            Action instance
        """
        space = cls.get_space()
        return gym_to_pydantic(gym_obj, cls, space)

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
