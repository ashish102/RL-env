"""API key authentication and rate limiting."""

import secrets
from typing import Dict, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from collections import defaultdict
import threading


class APIKeyManager:
    """
    Manage API keys and rate limiting.

    Features:
    - API key generation and validation
    - Per-key rate limiting
    - Request tracking
    - Key expiration
    """

    def __init__(
        self,
        default_rate_limit: int = 1000,
        rate_limit_window: int = 86400,  # 24 hours in seconds
    ):
        """
        Initialize API key manager.

        Args:
            default_rate_limit: Default requests per window
            rate_limit_window: Window size in seconds
        """
        self.default_rate_limit = default_rate_limit
        self.rate_limit_window = rate_limit_window

        # Storage (in production, use a database)
        self._keys: Dict[str, Dict] = {}
        self._request_counts: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

        # Security scheme
        self.security_scheme = HTTPBearer()

    def create_key(
        self,
        name: str,
        rate_limit: Optional[int] = None,
        expiration_days: Optional[int] = None,
    ) -> str:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            rate_limit: Custom rate limit (requests per window)
            expiration_days: Days until expiration (None = never)

        Returns:
            Generated API key
        """
        api_key = secrets.token_urlsafe(32)

        with self._lock:
            self._keys[api_key] = {
                "name": name,
                "created": datetime.now(),
                "rate_limit": rate_limit or self.default_rate_limit,
                "expiration": (
                    datetime.now() + timedelta(days=expiration_days)
                    if expiration_days
                    else None
                ),
                "active": True,
            }

        return api_key

    def validate_key(self, api_key: str) -> Dict:
        """
        Validate API key and return key info.

        Args:
            api_key: API key to validate

        Returns:
            Key information dictionary

        Raises:
            HTTPException: If key is invalid or expired
        """
        with self._lock:
            if api_key not in self._keys:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                )

            key_info = self._keys[api_key]

            # Check if key is active
            if not key_info["active"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has been deactivated",
                )

            # Check expiration
            if key_info["expiration"] and datetime.now() > key_info["expiration"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired",
                )

            return key_info

    def check_rate_limit(self, api_key: str) -> None:
        """
        Check if request is within rate limit.

        Args:
            api_key: API key to check

        Raises:
            HTTPException: If rate limit exceeded
        """
        key_info = self.validate_key(api_key)
        rate_limit = key_info["rate_limit"]

        with self._lock:
            # Clean old requests outside the window
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=self.rate_limit_window)

            self._request_counts[api_key] = [
                req_time
                for req_time in self._request_counts[api_key]
                if req_time > window_start
            ]

            # Check rate limit
            request_count = len(self._request_counts[api_key])
            if request_count >= rate_limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Limit: {rate_limit} requests per {self.rate_limit_window} seconds",
                )

            # Record this request
            self._request_counts[api_key].append(current_time)

    def revoke_key(self, api_key: str) -> None:
        """
        Revoke an API key.

        Args:
            api_key: API key to revoke
        """
        with self._lock:
            if api_key in self._keys:
                self._keys[api_key]["active"] = False

    def get_usage_stats(self, api_key: str) -> Dict:
        """
        Get usage statistics for an API key.

        Args:
            api_key: API key to check

        Returns:
            Usage statistics
        """
        key_info = self.validate_key(api_key)

        with self._lock:
            # Clean old requests
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=self.rate_limit_window)

            self._request_counts[api_key] = [
                req_time
                for req_time in self._request_counts[api_key]
                if req_time > window_start
            ]

            request_count = len(self._request_counts[api_key])

            return {
                "name": key_info["name"],
                "requests_in_window": request_count,
                "rate_limit": key_info["rate_limit"],
                "remaining": key_info["rate_limit"] - request_count,
                "window_seconds": self.rate_limit_window,
            }


# Global API key manager instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
) -> str:
    """
    Verify API key from request.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Validated API key

    Raises:
        HTTPException: If authentication fails
    """
    api_key = credentials.credentials
    manager = get_api_key_manager()

    # Validate key
    manager.validate_key(api_key)

    # Check rate limit
    manager.check_rate_limit(api_key)

    return api_key
