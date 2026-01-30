"""
Database models for LangCoach application.

Defines SQLAlchemy ORM models for:
- User: WeChat user information
- Conversation: Conversation sessions
- Message: Individual messages in conversations
- CustomScenario: Custom scenario extraction details
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class User(Base):
    """User model for WeChat Mini Program users."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    wechat_openid = Column(String(100), unique=True, nullable=False, index=True)
    nickname = Column(String(100))
    avatar_url = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_user_openid', 'wechat_openid'),
        Index('idx_user_created', 'created_at'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, nickname='{self.nickname}', openid='{self.wechat_openid[:10]}...')>"


class Conversation(Base):
    """Conversation model for storing conversation sessions."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    scenario = Column(String(100), nullable=False, index=True)
    difficulty = Column(String(20), nullable=False)
    max_turns = Column(Integer, nullable=False)
    current_turn = Column(Integer, default=0, nullable=False)
    status = Column(String(20), default="active", nullable=False, index=True)  # active, ended
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ended_at = Column(DateTime, index=True)
    rating = Column(Integer)  # 1-5 stars
    overall_score = Column(Integer)  # 0-100

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    custom_scenario = relationship("CustomScenario", back_populates="conversation", uselist=False, cascade="all, delete-orphan")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_conv_user_created', 'user_id', 'created_at'),
        Index('idx_conv_user_status', 'user_id', 'status'),
        Index('idx_conv_session', 'session_id'),
    )

    def __repr__(self):
        return f"<Conversation(id={self.id}, session_id='{self.session_id}', scenario='{self.scenario}', status='{self.status}')>"

    def to_dict(self):
        """Convert conversation to dictionary for API responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "max_turns": self.max_turns,
            "current_turn": self.current_turn,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "rating": self.rating,
            "overall_score": self.overall_score,
        }


class Message(Base):
    """Message model for storing individual messages in conversations."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    audio_url = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    # Indexes
    __table_args__ = (
        Index('idx_msg_conv_time', 'conversation_id', 'timestamp'),
    )

    def __repr__(self):
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, role='{self.role}')>"

    def to_dict(self):
        """Convert message to dictionary for API responses."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "audio_url": self.audio_url,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class CustomScenario(Base):
    """Custom scenario model for storing extracted scenario details."""

    __tablename__ = "custom_scenarios"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), unique=True, nullable=False, index=True)
    ai_role = Column(String(200))
    ai_role_cn = Column(String(200))
    user_role = Column(String(200))
    user_role_cn = Column(String(200))
    goal = Column(Text)
    goal_cn = Column(Text)
    challenge = Column(Text)
    challenge_cn = Column(Text)
    greeting = Column(Text)
    difficulty_level = Column(String(20))
    speaking_speed = Column(String(20))
    vocabulary = Column(String(20))
    scenario_summary = Column(Text)
    scenario_summary_cn = Column(Text)

    # Relationships
    conversation = relationship("Conversation", back_populates="custom_scenario")

    def __repr__(self):
        return f"<CustomScenario(id={self.id}, conversation_id={self.conversation_id}, ai_role='{self.ai_role}')>"

    def to_dict(self):
        """Convert custom scenario to dictionary for API responses."""
        return {
            "id": self.id,
            "ai_role": self.ai_role,
            "ai_role_cn": self.ai_role_cn,
            "user_role": self.user_role,
            "user_role_cn": self.user_role_cn,
            "goal": self.goal,
            "goal_cn": self.goal_cn,
            "challenge": self.challenge,
            "challenge_cn": self.challenge_cn,
            "greeting": self.greeting,
            "difficulty_level": self.difficulty_level,
            "speaking_speed": self.speaking_speed,
            "vocabulary": self.vocabulary,
            "scenario_summary": self.scenario_summary,
            "scenario_summary_cn": self.scenario_summary_cn,
        }
