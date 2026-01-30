"""
JWT token handler for authentication.

Provides functions to create and verify JWT tokens for user authentication.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# JWT Configuration from environment variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))


def create_access_token(user_id: int, openid: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token for a user.

    Args:
        user_id: User's database ID
        openid: User's WeChat OpenID
        expires_delta: Optional custom expiration time

    Returns:
        str: Encoded JWT token

    Example:
        token = create_access_token(user_id=123, openid="wx_abc123")
    """
    if expires_delta is None:
        expires_delta = timedelta(days=JWT_EXPIRE_DAYS)

    expire = datetime.utcnow() + expires_delta

    to_encode = {
        "user_id": user_id,
        "openid": openid,
        "exp": expire,
        "iat": datetime.utcnow(),
    }

    try:
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        logger.info(f"Created JWT token for user_id={user_id}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating JWT token: {e}")
        raise


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string

    Returns:
        Dict containing user_id and openid if valid, None if invalid

    Example:
        payload = verify_token(token)
        if payload:
            user_id = payload["user_id"]
            openid = payload["openid"]
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: int = payload.get("user_id")
        openid: str = payload.get("openid")

        if user_id is None or openid is None:
            logger.warning("Token missing user_id or openid")
            return None

        return {"user_id": user_id, "openid": openid}
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        return None


def decode_token_without_verification(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode a JWT token without verification (for debugging only).

    Args:
        token: JWT token string

    Returns:
        Dict containing token payload if decodable, None otherwise

    Warning:
        This function does NOT verify the token signature.
        Use only for debugging purposes.
    """
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except Exception as e:
        logger.error(f"Error decoding token: {e}")
        return None
