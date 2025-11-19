# Gym-Pydantic 🚀

**Production-ready library for converting Pydantic models into Gymnasium environments with automatic space derivation and multiple security layers.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **Zero-Boilerplate**: Define state and action as Pydantic models, get Gymnasium environments automatically
- **Automatic Space Derivation**: Infer observation/action spaces from Pydantic field constraints
- **Nested Models**: Full support for arbitrary nesting depth
- **Security-First**: Multiple security layers for safe transition functions
- **REST API**: Production-ready FastAPI server with authentication and rate limiting
- **SB3 Compatible**: Works seamlessly with Stable-Baselines3 algorithms
- **Type-Safe**: Full type hints and Pydantic validation

## 📦 Installation

```bash
# Core library
pip install -r requirements.txt

# Or install specific components
pip install "gym-pydantic[api]"     # Include REST API
pip install "gym-pydantic[security]" # Include sandboxing
pip install "gym-pydantic[all]"      # Everything
```

## 🚀 Quick Start

### Simple Grid World (5 minutes)

```python
from enum import Enum
from pydantic import Field
import numpy as np
from gym_pydantic import GymState, GymAction, create_gym_env

# 1. Define state as Pydantic model
class GridState(GymState):
    x: int = Field(ge=0, le=9)
    y: int = Field(ge=0, le=9)

# 2. Define action as Pydantic model
class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

class GridAction(GymAction):
    direction: Direction

# 3. Define transition function
def transition(state: GridState, action: GridAction):
    # Your logic here
    next_state = GridState(x=new_x, y=new_y)
    reward = -0.1
    done = (new_x == 9 and new_y == 9)
    return next_state, reward, done, {}

# 4. Create environment (spaces automatically derived!)
GridEnv = create_gym_env(
    state_class=GridState,
    action_class=GridAction,
    transition_fn=transition,
    initial_state_fn=lambda rng: GridState(x=0, y=0),
    max_steps=100,
)

# 5. Use with Stable-Baselines3
from stable_baselines3 import DQN

env = GridEnv()
model = DQN("MultiInputPolicy", env)
model.learn(50000)
```

**That's it!** No manual space definitions, no boilerplate.

## 🏗️ Core Concepts

### Automatic Space Derivation

Gym-Pydantic automatically converts Pydantic field types to Gymnasium spaces:

| Pydantic Field | Gymnasium Space |
|----------------|----------------|
| `int` with `Field(ge=0, le=10)` | `spaces.Box(0, 10, dtype=int)` |
| `float` with `Field(ge=0.0, le=1.0)` | `spaces.Box(0.0, 1.0, dtype=float32)` |
| `bool` | `spaces.Discrete(2)` |
| `Enum` with N values | `spaces.Discrete(N)` |
| Nested `BaseModel` | `spaces.Dict({...})` (recursive) |
| `List[float]` with `max_items=5` | `spaces.Box(shape=(5,))` |

### Nested Models Example

```python
from pydantic import BaseModel

class Position(BaseModel):
    x: float = Field(ge=0, le=10)
    y: float = Field(ge=0, le=10)

class Velocity(BaseModel):
    vx: float = Field(ge=-2, le=2)
    vy: float = Field(ge=-2, le=2)

class RobotState(GymState):
    position: Position  # Nested!
    velocity: Velocity  # Nested!
    energy: float = Field(ge=0, le=100)

# Automatically creates:
# spaces.Dict({
#     'position': spaces.Dict({'x': Box, 'y': Box}),
#     'velocity': spaces.Dict({'vx': Box, 'vy': Box}),
#     'energy': Box
# })
```

## 🔒 Security Layers

**Critical**: Transition functions from users are a security risk. Gym-Pydantic provides multiple approaches:

### 1. TransitionRegistry (Safest) ✅

Users provide only **configuration**, not code.

```python
from gym_pydantic.security import get_global_registry

registry = get_global_registry()

# User provides configuration only
config = {
    "grid_size": 10,
    "goal": {"x": 9, "y": 9},
    "goal_reward": 10.0,
    "step_penalty": -0.1
}

transition_fn = registry.get("grid_world", config)
```

**Built-in transitions**:
- `grid_world`: Discrete grid navigation
- `continuous_control`: Physics-based continuous control
- `physics_sim`: Simple physics simulation

### 2. DSL Compiler (Safe) ✅

Declarative JSON/YAML compiled to safe transitions.

```python
from gym_pydantic.security import DSLTransitionCompiler

compiler = DSLTransitionCompiler()

dsl_config = {
    "state_updates": [
        {
            "field": "x",
            "operation": "add",
            "operands": ["x", "vx"],
            "scale": 0.1,
            "clip": {"min": 0, "max": 10}
        }
    ],
    "reward": {
        "type": "distance_to_target",
        "target": {"x": 8, "y": 8},
        "scale": -0.1
    },
    "done": {
        "type": "goal_reached",
        "target": {"x": 8, "y": 8},
        "threshold": 0.5
    }
}

transition_fn = compiler.compile(dsl_config)
```

**Supported operations**: add, subtract, multiply, divide, clip, abs, min, max, sqrt

### 3. Sandboxed Execution (Advanced Users) ⚠️

For users who need custom code but with security restrictions.

```python
from gym_pydantic.security import SafeTransitionExecutor

executor = SafeTransitionExecutor()

code = """
def transition(state, action):
    import safe_numpy as np
    # Only safe operations allowed
    next_x = np.clip(state.x + action.dx * 0.1, 0, 10)
    return type(state)(x=next_x), reward, done, {}
"""

transition_fn = executor.compile(code)
```

**Security features**:
- Blocks file I/O, imports, eval, exec
- Memory limits (1GB)
- CPU timeout (10 seconds)
- Safe numpy wrapper (no load/save)

## 🌐 REST API

Production-ready FastAPI server with authentication and rate limiting.

### Start Server

```python
from gym_pydantic.api import create_app
from gym_pydantic.api.auth import get_api_key_manager
import uvicorn

# Create API key
manager = get_api_key_manager()
api_key = manager.create_key("my-key", rate_limit=1000)

# Start server
app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Or use the example server:

```bash
python -m gym_pydantic.examples.api_server
```

### API Endpoints

```
POST /environments/create      - Create environment
GET  /environments/{id}/reset  - Reset environment
POST /environments/{id}/step   - Take action
DELETE /environments/{id}      - Delete environment
GET  /transitions/available    - List available transitions
GET  /health                   - Health check
```

### Client Example

```python
import requests

# Create environment
response = requests.post(
    "http://localhost:8000/environments/create",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "state_schema": {
            "x": {"type": "integer", "ge": 0, "le": 9},
            "y": {"type": "integer", "ge": 0, "le": 9}
        },
        "action_schema": {
            "direction": {"type": "enum", "values": ["up", "down", "left", "right"]}
        },
        "transition_type": "grid_world",
        "transition_config": {
            "grid_size": 10,
            "goal": {"x": 9, "y": 9}
        },
        "max_steps": 100
    }
)

env_id = response.json()["env_id"]

# Reset
obs = requests.get(f"http://localhost:8000/environments/{env_id}/reset",
                   headers={"Authorization": f"Bearer {api_key}"}).json()

# Step
result = requests.post(
    f"http://localhost:8000/environments/{env_id}/step",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"action": 0}  # UP
).json()
```

## 📚 Examples

See the `gym_pydantic/examples/` directory:

- `simple_grid.py` - Basic grid world
- `continuous_control.py` - Continuous state/action spaces
- `nested_state.py` - Nested Pydantic models
- `registry_example.py` - Using TransitionRegistry
- `dsl_example.py` - Using DSL compiler
- `api_client_example.py` - REST API client
- `api_server.py` - REST API server

Run examples:

```bash
python -m gym_pydantic.examples.simple_grid
python -m gym_pydantic.examples.continuous_control
```

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=gym_pydantic --cov-report=html
```

## 📖 API Reference

### Core Classes

#### `GymState`

Base class for state models. Provides:
- `get_space()`: Get Gymnasium observation space
- `to_gym()`: Convert instance to Gym format
- `from_gym(gym_obj)`: Convert from Gym format

#### `GymAction`

Base class for action models. Provides same methods as `GymState`.

#### `create_gym_env()`

Factory function to create Gymnasium environments.

**Parameters**:
- `state_class`: Pydantic model class for state
- `action_class`: Pydantic model class for action
- `transition_fn`: Function `(state, action) -> (next_state, reward, done, info)`
- `initial_state_fn`: Function `(rng) -> initial_state`
- `max_steps`: Maximum steps before truncation
- `render_fn`: Optional rendering function
- `env_name`: Environment class name

**Returns**: Gymnasium environment class

### Security Classes

#### `TransitionRegistry`

Registry of pre-approved transitions.

- `get(name, config)`: Get configured transition
- `list_available()`: List available transitions
- `register(name, factory)`: Register new transition

#### `DSLTransitionCompiler`

Compile declarative DSL to transitions.

- `compile(config)`: Compile DSL config to function

#### `SafeTransitionExecutor`

Execute user code in sandbox.

- `compile(code)`: Compile and sandbox code

## 🔧 Advanced Usage

### Custom Transition Registration

```python
from gym_pydantic.security import register_transition

@register_transition("my_custom_transition")
def create_my_transition(config):
    def transition(state, action):
        # Your logic
        return next_state, reward, done, info
    return transition
```

### Multi-Agent Environments

```python
class MultiAgentState(GymState):
    agent1_x: int = Field(ge=0, le=10)
    agent1_y: int = Field(ge=0, le=10)
    agent2_x: int = Field(ge=0, le=10)
    agent2_y: int = Field(ge=0, le=10)

# Spaces automatically handle multi-agent scenarios
```

## 🛡️ Security Best Practices

### For Python Library Users

- ✅ Use TransitionRegistry for known patterns
- ✅ Use DSL for simple custom logic
- ⚠️ Use Sandbox only if absolutely necessary
- ❌ Never execute untrusted code without sandboxing

### For REST API Deployment

- ✅ ONLY use TransitionRegistry or DSL
- ✅ Enable rate limiting
- ✅ Use HTTPS in production
- ✅ Rotate API keys regularly
- ✅ Monitor for abuse
- ❌ NEVER allow arbitrary code execution

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on [Gymnasium](https://gymnasium.farama.org/)
- Uses [Pydantic](https://pydantic.dev/) for validation
- Compatible with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- API built with [FastAPI](https://fastapi.tiangolo.com/)

## 📞 Support

- 📧 Issues: [GitHub Issues](https://github.com/yourusername/gym-pydantic/issues)
- 📖 Documentation: [Read the Docs](https://gym-pydantic.readthedocs.io)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/gym-pydantic/discussions)

---

**Made with ❤️ for the RL community**
