"""Automatic space derivation from Pydantic model specifications."""

from typing import Any, Dict, Type, get_args, get_origin, Union
from enum import Enum
import numpy as np
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from gymnasium import spaces


class SpaceDerivationError(Exception):
    """Raised when space cannot be derived from Pydantic field."""
    pass


def derive_space_from_field(
    field_name: str,
    field_info: FieldInfo,
    annotation: Type,
) -> spaces.Space:
    """
    Derive Gymnasium space from a single Pydantic field.

    Args:
        field_name: Name of the field
        field_info: Pydantic FieldInfo object
        annotation: Type annotation of the field

    Returns:
        Gymnasium Space object

    Raises:
        SpaceDerivationError: If space cannot be derived
    """
    # Handle Optional types (Union with None)
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            annotation = non_none_args[0]
            origin = get_origin(annotation)
        else:
            raise SpaceDerivationError(
                f"Field '{field_name}': Union types (except Optional) are not supported"
            )

    # Handle List types
    if origin is list:
        args = get_args(annotation)
        if not args:
            raise SpaceDerivationError(
                f"Field '{field_name}': List must have type parameter (e.g., List[float])"
            )

        element_type = args[0]

        # Get constraints from field_info.metadata
        constraints = {}
        if field_info.metadata:
            for metadata in field_info.metadata:
                if hasattr(metadata, '__dict__'):
                    constraints.update(metadata.__dict__)
                else:
                    # Handle annotated_types (Ge, Le, etc.) which use __slots__
                    for attr in ['ge', 'le', 'gt', 'lt', 'min_length', 'max_length']:
                        if hasattr(metadata, attr):
                            constraints[attr] = getattr(metadata, attr)

        max_items = constraints.get('max_length') or field_info.json_schema_extra
        min_items = constraints.get('min_length', 0)

        if max_items is None:
            raise SpaceDerivationError(
                f"Field '{field_name}': List fields must specify max_items constraint"
            )

        # Determine bounds for elements
        if element_type in (float, int):
            ge = constraints.get('ge', -np.inf)
            le = constraints.get('le', np.inf)
            gt = constraints.get('gt')
            lt = constraints.get('lt')

            low = gt + np.finfo(np.float32).eps if gt is not None else ge
            high = lt - np.finfo(np.float32).eps if lt is not None else le

            dtype = np.float32 if element_type == float else np.int64
            return spaces.Box(
                low=low,
                high=high,
                shape=(max_items,),
                dtype=dtype
            )
        else:
            raise SpaceDerivationError(
                f"Field '{field_name}': List element type {element_type} not supported"
            )

    # Handle Enum types
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        n_values = len(annotation)
        return spaces.Discrete(n_values)

    # Handle bool
    if annotation is bool:
        return spaces.Discrete(2)

    # Handle int and float with constraints
    if annotation in (int, float):
        # Get constraints from field_info.metadata
        constraints = {}
        if field_info.metadata:
            for metadata in field_info.metadata:
                if hasattr(metadata, '__dict__'):
                    constraints.update(metadata.__dict__)
                else:
                    # Handle annotated_types (Ge, Le, etc.) which use __slots__
                    for attr in ['ge', 'le', 'gt', 'lt', 'min_length', 'max_length']:
                        if hasattr(metadata, attr):
                            constraints[attr] = getattr(metadata, attr)

        ge = constraints.get('ge', -np.inf)
        le = constraints.get('le', np.inf)
        gt = constraints.get('gt')
        lt = constraints.get('lt')

        low = gt + np.finfo(np.float32).eps if gt is not None else ge
        high = lt - np.finfo(np.float32).eps if lt is not None else le

        if low == -np.inf or high == np.inf:
            raise SpaceDerivationError(
                f"Field '{field_name}': Numeric fields must have finite bounds (use Field(ge=..., le=...))"
            )

        dtype = np.float32 if annotation == float else np.int64
        return spaces.Box(
            low=low,
            high=high,
            shape=(1,),
            dtype=dtype
        )

    # Handle nested Pydantic models
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return derive_space_from_model(annotation)

    raise SpaceDerivationError(
        f"Field '{field_name}': Type {annotation} is not supported for space derivation"
    )


def derive_space_from_model(model_class: Type[BaseModel]) -> spaces.Space:
    """
    Derive Gymnasium space from Pydantic model.

    Supports:
    - float/int with Field(ge=..., le=...)
    - bool
    - Enum
    - Nested Pydantic models
    - List[T] with max_items
    - Optional types

    Args:
        model_class: Pydantic model class

    Returns:
        Gymnasium Space (typically spaces.Dict for multi-field models)

    Raises:
        SpaceDerivationError: If space cannot be derived
    """
    if not issubclass(model_class, BaseModel):
        raise SpaceDerivationError(f"{model_class} is not a Pydantic BaseModel")

    space_dict = {}

    for field_name, field_info in model_class.model_fields.items():
        annotation = field_info.annotation
        space_dict[field_name] = derive_space_from_field(field_name, field_info, annotation)

    # If single field, return the space directly (unwrapped)
    if len(space_dict) == 1:
        return list(space_dict.values())[0]

    return spaces.Dict(space_dict)


def pydantic_to_gym(instance: BaseModel, space: spaces.Space) -> Any:
    """
    Convert Pydantic model instance to Gymnasium format.

    Args:
        instance: Pydantic model instance
        space: Corresponding Gymnasium space

    Returns:
        Data in format expected by the space
    """
    if isinstance(space, spaces.Dict):
        result = {}
        for key in space.spaces.keys():
            value = getattr(instance, key)
            result[key] = pydantic_to_gym(value, space.spaces[key])
        return result

    elif isinstance(space, spaces.Box):
        # For single-field models or direct numeric values
        if isinstance(instance, BaseModel):
            # Get the single field
            fields = list(instance.model_fields.keys())
            if len(fields) == 1:
                value = getattr(instance, fields[0])
            else:
                raise ValueError("Box space with BaseModel must have single field")
        else:
            value = instance

        if isinstance(value, (list, tuple)):
            return np.array(value, dtype=space.dtype)
        else:
            return np.array([value], dtype=space.dtype)

    elif isinstance(space, spaces.Discrete):
        if isinstance(instance, BaseModel):
            fields = list(instance.model_fields.keys())
            value = getattr(instance, fields[0])
        else:
            value = instance

        # Handle Enum
        if isinstance(value, Enum):
            # Map enum to index
            enum_class = type(value)
            return list(enum_class).index(value)
        # Handle bool
        elif isinstance(value, bool):
            return int(value)
        else:
            return int(value)

    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


def gym_to_pydantic(gym_data: Any, model_class: Type[BaseModel], space: spaces.Space) -> BaseModel:
    """
    Convert Gymnasium format data to Pydantic model instance.

    Args:
        gym_data: Data in Gymnasium format
        model_class: Target Pydantic model class
        space: Corresponding Gymnasium space

    Returns:
        Pydantic model instance
    """
    if isinstance(space, spaces.Dict):
        kwargs = {}
        for key in space.spaces.keys():
            field_info = model_class.model_fields[key]
            field_type = field_info.annotation

            # Handle Optional
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    field_type = non_none_args[0]

            # Recursively convert nested models
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                kwargs[key] = gym_to_pydantic(gym_data[key], field_type, space.spaces[key])
            else:
                kwargs[key] = _convert_single_value(gym_data[key], field_type, space.spaces[key])

        return model_class(**kwargs)

    else:
        # Single-field model
        fields = list(model_class.model_fields.keys())
        if len(fields) != 1:
            raise ValueError("Non-Dict space requires single-field model")

        field_name = fields[0]
        field_info = model_class.model_fields[field_name]
        field_type = field_info.annotation

        value = _convert_single_value(gym_data, field_type, space)
        return model_class(**{field_name: value})


def _convert_single_value(gym_data: Any, field_type: Type, space: spaces.Space) -> Any:
    """Convert single value from gym format to Python type."""
    origin = get_origin(field_type)

    # Handle Optional
    if origin is Union:
        args = get_args(field_type)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            field_type = non_none_args[0]
            origin = get_origin(field_type)

    # Handle List
    if origin is list:
        if isinstance(gym_data, np.ndarray):
            return gym_data.tolist()
        return list(gym_data)

    # Handle Enum
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        enum_values = list(field_type)
        return enum_values[int(gym_data)]

    # Handle bool
    if field_type is bool:
        return bool(gym_data)

    # Handle numeric types
    if isinstance(space, spaces.Box):
        if isinstance(gym_data, np.ndarray):
            if gym_data.shape == (1,):
                value = gym_data[0]
            else:
                return gym_data.tolist()
        else:
            value = gym_data

        if field_type is int:
            return int(value)
        else:
            return float(value)

    # Handle Discrete
    if isinstance(space, spaces.Discrete):
        return int(gym_data)

    return gym_data
