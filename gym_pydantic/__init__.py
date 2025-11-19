"""
Gym-Pydantic: Production-ready library for converting Pydantic models into Gymnasium environments.

Features:
- Automatic space derivation from Pydantic models
- Zero-boilerplate environment creation
- Multiple security layers (Registry, DSL, Sandbox)
- REST API for remote environment management
- Full Stable-Baselines3 compatibility
"""

__version__ = "0.1.0"

from .base import GymState, GymAction
from .factory import create_gym_env
from .space_derivation import (
    derive_space_from_model,
    derive_space_from_field,
    pydantic_to_gym,
    gym_to_pydantic,
    SpaceDerivationError,
)
from .security import (
    TransitionRegistry,
    register_transition,
    DSLTransitionCompiler,
    SafeTransitionExecutor,
    SafeNumpyWrapper,
)

__all__ = [
    # Version
    "__version__",
    # Base classes
    "GymState",
    "GymAction",
    # Factory
    "create_gym_env",
    # Space derivation
    "derive_space_from_model",
    "derive_space_from_field",
    "pydantic_to_gym",
    "gym_to_pydantic",
    "SpaceDerivationError",
    # Security
    "TransitionRegistry",
    "register_transition",
    "DSLTransitionCompiler",
    "SafeTransitionExecutor",
    "SafeNumpyWrapper",
]
