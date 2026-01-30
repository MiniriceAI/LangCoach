"""
Database package for LangCoach.

Exports database models, connection utilities, and session management.
"""

from src.database.models import Base, User, Conversation, Message, CustomScenario
from src.database.connection import (
    engine,
    SessionLocal,
    init_db,
    get_db,
    get_db_session,
    close_db,
)

__all__ = [
    "Base",
    "User",
    "Conversation",
    "Message",
    "CustomScenario",
    "engine",
    "SessionLocal",
    "init_db",
    "get_db",
    "get_db_session",
    "close_db",
]
