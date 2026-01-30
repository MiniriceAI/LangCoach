"""
Conversation history management helpers.

Provides functions for querying and managing conversation history.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.database import User, Conversation, Message, CustomScenario

logger = logging.getLogger(__name__)


def get_user_conversations(
    db: Session,
    user: User,
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get paginated list of user's conversations.

    Args:
        db: Database session
        user: User object
        limit: Maximum number of conversations to return
        offset: Number of conversations to skip

    Returns:
        Dict with conversations list and total count
    """
    # Query conversations with pagination
    query = db.query(Conversation).filter(
        Conversation.user_id == user.id
    ).order_by(desc(Conversation.created_at))

    total = query.count()
    conversations = query.offset(offset).limit(limit).all()

    # Format conversations for response
    conversations_data = []
    for conv in conversations:
        # Calculate duration if conversation ended
        duration_minutes = None
        if conv.ended_at and conv.created_at:
            duration_seconds = (conv.ended_at - conv.created_at).total_seconds()
            duration_minutes = int(duration_seconds / 60)

        conv_data = {
            "id": conv.id,
            "session_id": conv.session_id,
            "scenario": conv.scenario,
            "difficulty": conv.difficulty,
            "date": conv.created_at.isoformat() if conv.created_at else None,
            "duration": duration_minutes,
            "turns": conv.current_turn,
            "max_turns": conv.max_turns,
            "status": conv.status,
            "rating": conv.rating,
            "overall_score": conv.overall_score,
        }

        # Include custom scenario info if available
        if conv.custom_scenario:
            conv_data["is_custom"] = True
            conv_data["scenario_summary"] = conv.custom_scenario.scenario_summary
        else:
            conv_data["is_custom"] = False

        conversations_data.append(conv_data)

    return {
        "conversations": conversations_data,
        "total": total,
        "limit": limit,
        "offset": offset
    }


def get_conversation_detail(
    db: Session,
    user: User,
    conversation_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific conversation.

    Args:
        db: Database session
        user: User object
        conversation_id: Conversation ID

    Returns:
        Dict with conversation details, or None if not found
    """
    # Query conversation
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()

    if not conversation:
        return None

    # Calculate duration
    duration_minutes = None
    if conversation.ended_at and conversation.created_at:
        duration_seconds = (conversation.ended_at - conversation.created_at).total_seconds()
        duration_minutes = int(duration_seconds / 60)

    # Get messages
    messages = db.query(Message).filter(
        Message.conversation_id == conversation.id
    ).order_by(Message.timestamp).all()

    messages_data = [msg.to_dict() for msg in messages]

    # Build response
    detail = {
        "id": conversation.id,
        "session_id": conversation.session_id,
        "scenario": conversation.scenario,
        "difficulty": conversation.difficulty,
        "max_turns": conversation.max_turns,
        "current_turn": conversation.current_turn,
        "status": conversation.status,
        "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
        "ended_at": conversation.ended_at.isoformat() if conversation.ended_at else None,
        "duration": duration_minutes,
        "rating": conversation.rating,
        "overall_score": conversation.overall_score,
        "messages": messages_data,
    }

    # Include custom scenario details if available
    if conversation.custom_scenario:
        detail["custom_scenario"] = conversation.custom_scenario.to_dict()

    return detail


def get_recent_conversation_topics(
    db: Session,
    user: User,
    limit: int = 3
) -> str:
    """
    Get user's recent conversation topics for personalized greetings.

    Args:
        db: Database session
        user: User object
        limit: Number of recent conversations to include

    Returns:
        Formatted string of recent topics (e.g., "job interview (Jan 28), hotel check-in (Jan 27)")
    """
    # Query recent conversations
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user.id,
        Conversation.status == "ended"
    ).order_by(desc(Conversation.created_at)).limit(limit).all()

    if not conversations:
        return ""

    # Format topics
    topics = []
    for conv in conversations:
        # Format date as "Jan 28"
        date_str = conv.created_at.strftime("%b %d") if conv.created_at else ""

        # Use custom scenario summary if available, otherwise use scenario name
        if conv.custom_scenario and conv.custom_scenario.scenario_summary:
            topic_name = conv.custom_scenario.scenario_summary
        else:
            # Convert scenario name to readable format
            topic_name = conv.scenario.replace("_", " ").title()

        topics.append(f"{topic_name} ({date_str})")

    return ", ".join(topics)


def save_conversation_to_db(
    db: Session,
    user: User,
    session_id: str,
    scenario: str,
    difficulty: str,
    max_turns: int,
    custom_scenario_data: Optional[Dict[str, Any]] = None
) -> Conversation:
    """
    Create a new conversation record in the database.

    Args:
        db: Database session
        user: User object
        session_id: Unique session identifier
        scenario: Scenario name
        difficulty: Difficulty level
        max_turns: Maximum number of turns
        custom_scenario_data: Optional custom scenario extraction data

    Returns:
        Conversation: Created conversation object
    """
    # Create conversation
    conversation = Conversation(
        user_id=user.id,
        session_id=session_id,
        scenario=scenario,
        difficulty=difficulty,
        max_turns=max_turns,
        current_turn=0,
        status="active",
        created_at=datetime.utcnow()
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    # Create custom scenario if provided
    if custom_scenario_data:
        custom_scenario = CustomScenario(
            conversation_id=conversation.id,
            **custom_scenario_data
        )
        db.add(custom_scenario)
        db.commit()

    logger.info(f"Created conversation {conversation.id} for user {user.id}")
    return conversation


def save_message_to_db(
    db: Session,
    conversation: Conversation,
    role: str,
    content: str,
    audio_url: Optional[str] = None
) -> Message:
    """
    Save a message to the database.

    Args:
        db: Database session
        conversation: Conversation object
        role: Message role ('user' or 'assistant')
        content: Message content
        audio_url: Optional audio URL

    Returns:
        Message: Created message object
    """
    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        audio_url=audio_url,
        timestamp=datetime.utcnow()
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    logger.debug(f"Saved {role} message to conversation {conversation.id}")
    return message


def update_conversation_turn(
    db: Session,
    conversation: Conversation,
    increment: int = 1
) -> None:
    """
    Update conversation turn count.

    Args:
        db: Database session
        conversation: Conversation object
        increment: Number to increment turn count by
    """
    conversation.current_turn += increment
    db.commit()


def end_conversation(
    db: Session,
    conversation: Conversation,
    rating: Optional[int] = None,
    overall_score: Optional[int] = None
) -> None:
    """
    Mark a conversation as ended.

    Args:
        db: Database session
        conversation: Conversation object
        rating: Optional rating (1-5 stars)
        overall_score: Optional overall score (0-100)
    """
    conversation.status = "ended"
    conversation.ended_at = datetime.utcnow()
    if rating is not None:
        conversation.rating = rating
    if overall_score is not None:
        conversation.overall_score = overall_score
    db.commit()

    logger.info(f"Ended conversation {conversation.id}")
