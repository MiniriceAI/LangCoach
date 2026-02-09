"""
Authentication middleware for FastAPI.

Provides dependency functions for protecting routes with JWT authentication.
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from src.auth.jwt_handler import verify_token
from src.database import get_db, User

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency to get the current authenticated user.

    Extracts JWT token from Authorization header, verifies it,
    and returns the corresponding User object from database.

    Args:
        credentials: HTTP Bearer credentials from request header
        db: Database session

    Returns:
        User: Authenticated user object

    Raises:
        HTTPException: 401 if token is invalid or user not found

    Usage:
        @app.get("/protected")
        def protected_route(current_user: User = Depends(get_current_user)):
            return {"user_id": current_user.id}
    """
    token = credentials.credentials

    # Verify token
    payload = verify_token(token)
    if payload is None:
        logger.warning("Invalid or expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("user_id")
    openid = payload.get("openid")

    # Fetch user from database
    user = db.query(User).filter(User.id == user_id, User.wechat_openid == openid).first()
    if user is None:
        logger.warning(f"User not found for user_id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(f"Authenticated user: {user.id} ({user.nickname})")
    return user


async def get_optional_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    FastAPI dependency to get the current user if authenticated, None otherwise.

    Useful for routes that work both with and without authentication.

    Args:
        authorization: Authorization header value
        db: Database session

    Returns:
        User if authenticated, None otherwise

    Usage:
        @app.get("/optional-auth")
        def optional_route(current_user: Optional[User] = Depends(get_optional_user)):
            if current_user:
                return {"message": f"Hello, {current_user.nickname}"}
            return {"message": "Hello, guest"}
    """
    if authorization is None or not authorization.startswith("Bearer "):
        return None

    token = authorization.replace("Bearer ", "")

    # Verify token
    payload = verify_token(token)
    if payload is None:
        return None

    user_id = payload.get("user_id")
    openid = payload.get("openid")

    # Fetch user from database
    user = db.query(User).filter(User.id == user_id, User.wechat_openid == openid).first()
    return user


def verify_user_owns_conversation(user: User, conversation) -> None:
    """
    Verify that a user owns a specific conversation.

    Args:
        user: User object
        conversation: Conversation object

    Raises:
        HTTPException: 403 if user doesn't own the conversation
    """
    if conversation.user_id != user.id:
        logger.warning(f"User {user.id} attempted to access conversation {conversation.id} owned by user {conversation.user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this conversation"
        )
