"""
WeChat Mini Program authentication helper.

Handles WeChat OAuth flow and user management.
"""

import os
import logging
import httpx
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from src.database import User
from src.auth import create_access_token

logger = logging.getLogger(__name__)

# WeChat API configuration
WECHAT_APP_ID = os.getenv("WECHAT_APP_ID", "")
WECHAT_APP_SECRET = os.getenv("WECHAT_APP_SECRET", "")
WECHAT_API_URL = "https://api.weixin.qq.com/sns/jscode2session"


async def exchange_code_for_openid(code: str) -> Optional[Dict[str, Any]]:
    """
    Exchange WeChat authorization code for openid and session_key.

    Args:
        code: Authorization code from wx.login()

    Returns:
        Dict with openid and session_key if successful, None otherwise

    WeChat API Response:
        {
            "openid": "user_openid",
            "session_key": "session_key",
            "unionid": "unionid" (optional)
        }
    """
    if not WECHAT_APP_ID or not WECHAT_APP_SECRET:
        logger.warning("WeChat APP_ID or APP_SECRET not configured")
        # For development/testing, return mock data
        return {
            "openid": f"mock_openid_{code}",
            "session_key": "mock_session_key"
        }

    params = {
        "appid": WECHAT_APP_ID,
        "secret": WECHAT_APP_SECRET,
        "js_code": code,
        "grant_type": "authorization_code"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(WECHAT_API_URL, params=params, timeout=10.0)
            data = response.json()

            if "errcode" in data and data["errcode"] != 0:
                logger.error(f"WeChat API error: {data.get('errmsg')}")
                return None

            if "openid" not in data:
                logger.error("WeChat API response missing openid")
                return None

            return data
    except Exception as e:
        logger.error(f"Error calling WeChat API: {e}")
        return None


def get_or_create_user(db: Session, openid: str, nickname: Optional[str] = None, avatar_url: Optional[str] = None) -> User:
    """
    Get existing user by openid or create a new one.

    Args:
        db: Database session
        openid: WeChat openid
        nickname: User's nickname (optional)
        avatar_url: User's avatar URL (optional)

    Returns:
        User: User object (existing or newly created)
    """
    # Try to find existing user
    user = db.query(User).filter(User.wechat_openid == openid).first()

    if user:
        # Update last login time
        user.last_login = datetime.utcnow()
        # Update nickname and avatar if provided
        if nickname:
            user.nickname = nickname
        if avatar_url:
            user.avatar_url = avatar_url
        db.commit()
        db.refresh(user)
        logger.info(f"Existing user logged in: {user.id}")
    else:
        # Create new user
        user = User(
            wechat_openid=openid,
            nickname=nickname or "学习者",
            avatar_url=avatar_url,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"New user created: {user.id}")

    return user


async def wechat_login(code: str, nickname: Optional[str], avatar_url: Optional[str], db: Session) -> Dict[str, Any]:
    """
    Complete WeChat login flow.

    Args:
        code: WeChat authorization code
        nickname: User's nickname (optional)
        avatar_url: User's avatar URL (optional)
        db: Database session

    Returns:
        Dict with token and user info

    Raises:
        ValueError: If login fails
    """
    # Exchange code for openid
    wechat_data = await exchange_code_for_openid(code)
    if not wechat_data:
        raise ValueError("Failed to exchange code for openid")

    openid = wechat_data["openid"]

    # Get or create user
    user = get_or_create_user(db, openid, nickname, avatar_url)

    # Generate JWT token
    token = create_access_token(user.id, user.wechat_openid)

    return {
        "token": token,
        "user": {
            "id": user.id,
            "nickname": user.nickname,
            "avatar_url": user.avatar_url,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }
    }
