"""
REST API Client Example

Demonstrates using the Gym-Pydantic REST API for remote environment management.

Note: This requires the API server to be running.
Start the server with:
    python -m gym_pydantic.examples.api_server
"""

import requests
import time


class GymPydanticClient:
    """Simple client for Gym-Pydantic REST API."""

    def __init__(self, base_url: str, api_key: str):
        """
        Initialize client.

        Args:
            base_url: API base URL (e.g., "http://localhost:8000")
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def create_environment(
        self,
        state_schema: dict,
        action_schema: dict,
        transition_type: str,
        transition_config: dict,
        max_steps: int = 1000,
    ) -> dict:
        """Create a new environment."""
        response = requests.post(
            f"{self.base_url}/environments/create",
            headers=self.headers,
            json={
                "state_schema": state_schema,
                "action_schema": action_schema,
                "transition_type": transition_type,
                "transition_config": transition_config,
                "max_steps": max_steps,
            },
        )
        response.raise_for_status()
        return response.json()

    def reset(self, env_id: str) -> dict:
        """Reset environment."""
        response = requests.get(
            f"{self.base_url}/environments/{env_id}/reset",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def step(self, env_id: str, action: dict) -> dict:
        """Take a step in environment."""
        response = requests.post(
            f"{self.base_url}/environments/{env_id}/step",
            headers=self.headers,
            json={"action": action},
        )
        response.raise_for_status()
        return response.json()

    def delete(self, env_id: str) -> None:
        """Delete environment."""
        response = requests.delete(
            f"{self.base_url}/environments/{env_id}",
            headers=self.headers,
        )
        response.raise_for_status()

    def list_transitions(self) -> list:
        """List available transition types."""
        response = requests.get(
            f"{self.base_url}/transitions/available",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def main():
    """Run API client example."""
    print("=== Gym-Pydantic REST API Client Example ===\n")

    # Configuration
    BASE_URL = "http://localhost:8000"
    API_KEY = "your-api-key-here"  # Replace with actual API key

    # Create client
    client = GymPydanticClient(BASE_URL, API_KEY)

    try:
        # Check API health
        print("Checking API health...")
        health = client.health()
        print(f"API Status: {health}")

        # List available transitions
        print("\nAvailable transitions:")
        transitions = client.list_transitions()
        for t in transitions:
            print(f"  - {t['name']}: {t['description']}")

        # Create environment
        print("\n=== Creating Grid World Environment ===")
        env_response = client.create_environment(
            state_schema={
                "x": {"type": "integer", "ge": 0, "le": 9},
                "y": {"type": "integer", "ge": 0, "le": 9},
            },
            action_schema={
                "direction": {
                    "type": "enum",
                    "values": ["up", "down", "left", "right"],
                }
            },
            transition_type="grid_world",
            transition_config={
                "grid_size": 10,
                "goal": {"x": 9, "y": 9},
                "goal_reward": 10.0,
                "step_penalty": -0.1,
            },
            max_steps=100,
        )

        env_id = env_response["env_id"]
        print(f"Environment created: {env_id}")
        print(f"Observation space: {env_response['observation_space']}")
        print(f"Action space: {env_response['action_space']}")

        # Reset environment
        print("\n=== Running Episode ===")
        reset_response = client.reset(env_id)
        obs = reset_response["observation"]
        print(f"Initial observation: {obs}")

        # Run episode
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 20:
            # Take random action
            import random

            action = random.randint(0, 3)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

            step_response = client.step(env_id, action)

            obs = step_response["observation"]
            reward = step_response["reward"]
            terminated = step_response["terminated"]
            truncated = step_response["truncated"]
            info = step_response["info"]

            done = terminated or truncated
            total_reward += reward
            steps += 1

            print(
                f"Step {steps}: action={action}, obs={obs}, "
                f"reward={reward:.2f}, done={done}"
            )

        print(f"\nEpisode finished. Total reward: {total_reward:.2f}")

        # Clean up
        print("\n=== Cleaning Up ===")
        client.delete(env_id)
        print(f"Environment {env_id} deleted")

        print("\n✓ API client example completed!")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Please start the server first:")
        print("    python -m gym_pydantic.examples.api_server")
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ API Error: {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")


if __name__ == "__main__":
    main()
