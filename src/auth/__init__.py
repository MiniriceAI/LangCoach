"""
Authentication package for LangCoach.

Exports JWT token handling and authentication middleware.
"""

from src.auth.jwt_handler import create_access_token, verify_token, decode_token_without_verification
from src.auth.middleware import get_current_user, get_optional_user, verify_user_owns_conversation

__all__ = [
    "create_access_token",
    "verify_token",
    "decode_token_without_verification",
    "get_current_user",
    "get_optional_user",
    "verify_user_owns_conversation",
]
