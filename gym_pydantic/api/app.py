"""FastAPI application for secure environment management."""

import uuid
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import gymnasium as gym
from pydantic import create_model
import numpy as np

from ..base import GymState, GymAction
from ..factory import create_gym_env
from ..security import TransitionRegistry, DSLTransitionCompiler
from .models import (
    EnvironmentRequest,
    EnvironmentResponse,
    ResetResponse,
    StepRequest,
    StepResponse,
    TransitionInfo,
    HealthResponse,
    ErrorResponse,
)
from .auth import verify_api_key, get_api_key_manager


def create_app(
    title: str = "Gym-Pydantic API",
    version: str = "0.1.0",
    enable_cors: bool = True,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        title: API title
        version: API version
        enable_cors: Whether to enable CORS

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Production-ready REST API for creating Gymnasium environments from Pydantic models",
    )

    # Enable CORS if requested
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Environment storage (in production, use a database or cache)
    environments: Dict[str, Dict[str, Any]] = {}

    # Initialize transition registry
    transition_registry = TransitionRegistry()
    dsl_compiler = DSLTransitionCompiler()

    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "message": "Gym-Pydantic API",
            "version": version,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["General"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version=version,
            environments_active=len(environments),
        )

    @app.get(
        "/transitions/available",
        response_model=list[TransitionInfo],
        tags=["Transitions"],
    )
    async def list_transitions(api_key: str = Depends(verify_api_key)):
        """
        List all available pre-approved transition types.

        Requires API key authentication.
        """
        available = transition_registry.list_available()

        transitions_info = []
        for name in available:
            try:
                schema = transition_registry.get_config_schema(name)
                transitions_info.append(
                    TransitionInfo(
                        name=name,
                        description=f"Pre-built {name} transition",
                        config_schema=schema,
                    )
                )
            except Exception:
                pass

        return transitions_info

    @app.post(
        "/environments/create",
        response_model=EnvironmentResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Environments"],
    )
    async def create_environment(
        request: EnvironmentRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Create a new Gymnasium environment from Pydantic schemas.

        Security: Only pre-approved transitions or DSL-based transitions allowed.
        No arbitrary code execution permitted.

        Args:
            request: Environment creation request

        Returns:
            Environment creation response with env_id and space information
        """
        try:
            # Create dynamic Pydantic models from schemas
            state_class = _schema_to_pydantic(request.state_schema, "State", GymState)
            action_class = _schema_to_pydantic(request.action_schema, "Action", GymAction)

            # Get transition function from registry
            try:
                transition_fn = transition_registry.get(
                    request.transition_type, request.transition_config
                )
            except KeyError:
                # Try DSL compilation as fallback
                try:
                    transition_fn = dsl_compiler.compile(request.transition_config)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid transition type or configuration: {str(e)}",
                    )

            # Create initial state function
            def initial_state_fn(rng):
                if request.initial_state:
                    return state_class(**request.initial_state)
                else:
                    # Create default initial state (use field defaults or zeros)
                    defaults = {}
                    for field_name, field_info in state_class.model_fields.items():
                        if field_info.default is not None:
                            defaults[field_name] = field_info.default
                        else:
                            # Use zero/false for numeric/bool fields
                            annotation = field_info.annotation
                            if annotation == int:
                                defaults[field_name] = 0
                            elif annotation == float:
                                defaults[field_name] = 0.0
                            elif annotation == bool:
                                defaults[field_name] = False
                    return state_class(**defaults)

            # Create environment class
            env_class = create_gym_env(
                state_class=state_class,
                action_class=action_class,
                transition_fn=transition_fn,
                initial_state_fn=initial_state_fn,
                max_steps=request.max_steps,
            )

            # Instantiate environment
            env = env_class()

            # Generate unique ID
            env_id = str(uuid.uuid4())

            # Store environment
            environments[env_id] = {
                "env": env,
                "state_class": state_class,
                "action_class": action_class,
                "max_steps": request.max_steps,
                "api_key": api_key,
            }

            # Get space information
            obs_space = env.observation_space
            action_space = env.action_space

            return EnvironmentResponse(
                env_id=env_id,
                observation_space=_space_to_dict(obs_space),
                action_space=_space_to_dict(action_space),
                max_steps=request.max_steps,
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error creating environment: {str(e)}",
            )

    @app.get(
        "/environments/{env_id}/reset",
        response_model=ResetResponse,
        tags=["Environments"],
    )
    async def reset_environment(
        env_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Reset environment to initial state.

        Args:
            env_id: Environment ID

        Returns:
            Initial observation and info
        """
        if env_id not in environments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found",
            )

        env_data = environments[env_id]

        # Verify ownership
        if env_data["api_key"] != api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this environment",
            )

        env = env_data["env"]
        observation, info = env.reset()

        return ResetResponse(
            observation=_convert_to_json_serializable(observation),
            info=info,
        )

    @app.post(
        "/environments/{env_id}/step",
        response_model=StepResponse,
        tags=["Environments"],
    )
    async def step_environment(
        env_id: str,
        request: StepRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Execute one step in the environment.

        Args:
            env_id: Environment ID
            request: Step request with action

        Returns:
            Step result (observation, reward, terminated, truncated, info)
        """
        if env_id not in environments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found",
            )

        env_data = environments[env_id]

        # Verify ownership
        if env_data["api_key"] != api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this environment",
            )

        env = env_data["env"]

        try:
            observation, reward, terminated, truncated, info = env.step(request.action)

            return StepResponse(
                observation=_convert_to_json_serializable(observation),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error executing step: {str(e)}",
            )

    @app.delete(
        "/environments/{env_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Environments"],
    )
    async def delete_environment(
        env_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Delete an environment and free resources.

        Args:
            env_id: Environment ID
        """
        if env_id not in environments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found",
            )

        env_data = environments[env_id]

        # Verify ownership
        if env_data["api_key"] != api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this environment",
            )

        # Clean up
        env = env_data["env"]
        env.close()
        del environments[env_id]

    return app


def _schema_to_pydantic(schema: Dict[str, Any], name: str, base_class: type) -> type:
    """
    Convert JSON schema to Pydantic model class.

    Args:
        schema: Field schema dictionary
        name: Model class name
        base_class: Base class (GymState or GymAction)

    Returns:
        Pydantic model class
    """
    from pydantic import Field
    from enum import Enum

    fields = {}

    for field_name, field_spec in schema.items():
        field_type = field_spec.get("type")
        constraints = {}

        if "ge" in field_spec:
            constraints["ge"] = field_spec["ge"]
        if "le" in field_spec:
            constraints["le"] = field_spec["le"]
        if "gt" in field_spec:
            constraints["gt"] = field_spec["gt"]
        if "lt" in field_spec:
            constraints["lt"] = field_spec["lt"]

        if field_type == "integer":
            fields[field_name] = (int, Field(**constraints) if constraints else ...)
        elif field_type == "float":
            fields[field_name] = (float, Field(**constraints) if constraints else ...)
        elif field_type == "boolean":
            fields[field_name] = (bool, ...)
        elif field_type == "enum":
            values = field_spec.get("values", [])
            enum_class = Enum(f"{name}{field_name.capitalize()}Enum", {v.upper(): v for v in values})
            fields[field_name] = (enum_class, ...)
        else:
            raise ValueError(f"Unsupported field type: {field_type}")

    return create_model(name, **fields, __base__=base_class)


def _space_to_dict(space: gym.Space) -> Dict[str, Any]:
    """Convert Gymnasium space to JSON-serializable dict."""
    from gymnasium import spaces

    if isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {key: _space_to_dict(subspace) for key, subspace in space.spaces.items()},
        }
    elif isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "shape": list(space.shape),
            "dtype": str(space.dtype),
        }
    elif isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
        }
    else:
        return {"type": str(type(space).__name__)}


def _convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-serializable types to JSON-compatible types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj
