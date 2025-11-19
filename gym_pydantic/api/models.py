"""API request and response models."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class EnvironmentRequest(BaseModel):
    """Request model for creating an environment."""

    state_schema: Dict[str, Any] = Field(
        ...,
        description="Pydantic model schema for state as dictionary",
        example={
            "x": {"type": "integer", "ge": 0, "le": 9},
            "y": {"type": "integer", "ge": 0, "le": 9},
        },
    )

    action_schema: Dict[str, Any] = Field(
        ...,
        description="Pydantic model schema for action as dictionary",
        example={"direction": {"type": "enum", "values": ["up", "down", "left", "right"]}},
    )

    transition_type: str = Field(
        ...,
        description="Type of pre-approved transition (e.g., 'grid_world')",
        example="grid_world",
    )

    transition_config: Dict[str, Any] = Field(
        ...,
        description="Configuration for the transition function",
        example={
            "grid_size": 10,
            "goal": {"x": 9, "y": 9},
            "goal_reward": 10.0,
            "step_penalty": -0.1,
        },
    )

    max_steps: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum steps before truncation",
    )

    initial_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional initial state values",
    )


class EnvironmentResponse(BaseModel):
    """Response model for environment creation."""

    env_id: str = Field(..., description="Unique environment ID")
    observation_space: Dict[str, Any] = Field(..., description="Observation space description")
    action_space: Dict[str, Any] = Field(..., description="Action space description")
    max_steps: int = Field(..., description="Maximum steps before truncation")


class ResetResponse(BaseModel):
    """Response model for environment reset."""

    observation: Dict[str, Any] = Field(..., description="Initial observation")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional information")


class StepRequest(BaseModel):
    """Request model for taking a step."""

    action: Any = Field(..., description="Action in appropriate format")


class StepResponse(BaseModel):
    """Response model for step execution."""

    observation: Any = Field(..., description="Observation after step")
    reward: float = Field(..., description="Reward from this step")
    terminated: bool = Field(..., description="Whether episode is terminated")
    truncated: bool = Field(..., description="Whether episode is truncated")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional information")


class TransitionInfo(BaseModel):
    """Information about available transition type."""

    name: str = Field(..., description="Transition type name")
    description: str = Field(..., description="Transition type description")
    config_schema: Dict[str, Any] = Field(..., description="Configuration schema")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    environments_active: int = Field(..., description="Number of active environments")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
