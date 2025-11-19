"""
API Server Example

Simple script to run the Gym-Pydantic REST API server.

Usage:
    python -m gym_pydantic.examples.api_server

Then access the API at http://localhost:8000
API documentation available at http://localhost:8000/docs
"""

import uvicorn
from gym_pydantic.api import create_app
from gym_pydantic.api.auth import get_api_key_manager


def main():
    """Run the API server."""
    print("=== Gym-Pydantic API Server ===\n")

    # Create API key for testing
    api_key_manager = get_api_key_manager()
    api_key = api_key_manager.create_key(
        name="test-key",
        rate_limit=1000,
        expiration_days=30,
    )

    print(f"Test API Key created: {api_key}")
    print("\nSave this key - you'll need it for API requests!")
    print("\nStarting server...")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health\n")

    # Create and run app
    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
