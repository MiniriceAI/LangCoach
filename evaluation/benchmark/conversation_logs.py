"""
Conversation Log Management for Quality Evaluation

Provides utilities for:
- Storing conversation logs
- Sampling conversations for evaluation
- Loading conversation history
- Exporting logs for analysis
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ConversationLog:
    """A complete conversation session log."""
    session_id: str
    user_id: str
    scenario: str
    difficulty: str
    turn_limit: int
    timestamp: str
    turns: List[Dict[str, Any]]
    system_prompt: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationLog':
        """Create from dictionary."""
        return cls(**data)


class ConversationLogManager:
    """
    Manager for conversation logs.

    Handles storage, retrieval, and sampling of conversation logs
    for quality evaluation.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize log manager.

        Args:
            log_dir: Directory for storing logs. Defaults to evaluation/logs
        """
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Organize logs by date
        self.daily_dir = self.log_dir / "daily"
        self.daily_dir.mkdir(exist_ok=True)

    def save_conversation(self, conversation: ConversationLog) -> str:
        """
        Save a conversation log.

        Args:
            conversation: ConversationLog object

        Returns:
            Path to saved log file
        """
        # Organize by date
        date_str = datetime.fromisoformat(conversation.timestamp).strftime("%Y-%m-%d")
        date_dir = self.daily_dir / date_str
        date_dir.mkdir(exist_ok=True)

        # Save with session ID as filename
        filename = f"{conversation.session_id}.json"
        filepath = date_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)

        return str(filepath)

    def load_conversation(self, session_id: str, date: Optional[str] = None) -> Optional[ConversationLog]:
        """
        Load a conversation log by session ID.

        Args:
            session_id: Session ID
            date: Date string (YYYY-MM-DD). If None, searches all dates.

        Returns:
            ConversationLog or None if not found
        """
        if date:
            filepath = self.daily_dir / date / f"{session_id}.json"
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return ConversationLog.from_dict(data)
        else:
            # Search all date directories
            for date_dir in self.daily_dir.iterdir():
                if date_dir.is_dir():
                    filepath = date_dir / f"{session_id}.json"
                    if filepath.exists():
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        return ConversationLog.from_dict(data)

        return None

    def get_conversations_by_date(self, date: str) -> List[ConversationLog]:
        """
        Get all conversations for a specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            List of ConversationLog objects
        """
        date_dir = self.daily_dir / date
        if not date_dir.exists():
            return []

        conversations = []
        for filepath in date_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                conversations.append(ConversationLog.from_dict(data))
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

        return conversations

    def get_conversations_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[ConversationLog]:
        """
        Get conversations within a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of ConversationLog objects
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        conversations = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            conversations.extend(self.get_conversations_by_date(date_str))
            current += timedelta(days=1)

        return conversations

    def sample_conversations(
        self,
        date: str,
        n: int = 50,
        random_seed: Optional[int] = None
    ) -> List[ConversationLog]:
        """
        Randomly sample conversations from a specific date.

        This implements the "Daily Evals" sampling strategy from EVALUATION_PLAN.md:
        - Randomly select 50 conversation sessions from the previous day's logs

        Args:
            date: Date string (YYYY-MM-DD)
            n: Number of samples (default: 50)
            random_seed: Random seed for reproducibility

        Returns:
            List of sampled ConversationLog objects
        """
        conversations = self.get_conversations_by_date(date)

        if not conversations:
            return []

        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # Sample up to n conversations
        sample_size = min(n, len(conversations))
        return random.sample(conversations, sample_size)

    def sample_previous_day(
        self,
        n: int = 50,
        random_seed: Optional[int] = None
    ) -> List[ConversationLog]:
        """
        Sample conversations from the previous day.

        Args:
            n: Number of samples (default: 50)
            random_seed: Random seed for reproducibility

        Returns:
            List of sampled ConversationLog objects
        """
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return self.sample_conversations(yesterday, n, random_seed)

    def extract_turns_for_evaluation(
        self,
        conversations: List[ConversationLog]
    ) -> List[Dict[str, Any]]:
        """
        Extract individual turns from conversations for quality evaluation.

        Each turn becomes a separate evaluation sample.

        Args:
            conversations: List of ConversationLog objects

        Returns:
            List of turn dictionaries ready for quality evaluation
        """
        turns = []

        for conv in conversations:
            for i, turn in enumerate(conv.turns):
                turn_data = {
                    "session_id": conv.session_id,
                    "turn_number": i + 1,
                    "user_input": turn.get("user_input", ""),
                    "ai_response": turn.get("ai_response", ""),
                    "scenario": conv.scenario,
                    "difficulty": conv.difficulty,
                    "system_prompt": conv.system_prompt,
                    "turn_limit": conv.turn_limit,
                    "correction_enabled": turn.get("correction_enabled", False),
                    "timestamp": turn.get("timestamp", conv.timestamp),
                }
                turns.append(turn_data)

        return turns

    def get_statistics(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored conversations.

        Args:
            date: Date string (YYYY-MM-DD). If None, returns overall stats.

        Returns:
            Dictionary with statistics
        """
        if date:
            conversations = self.get_conversations_by_date(date)
        else:
            # Get all conversations
            conversations = []
            for date_dir in self.daily_dir.iterdir():
                if date_dir.is_dir():
                    conversations.extend(
                        self.get_conversations_by_date(date_dir.name)
                    )

        if not conversations:
            return {
                "total_conversations": 0,
                "total_turns": 0,
                "scenarios": {},
                "difficulties": {},
            }

        # Calculate statistics
        total_turns = sum(len(conv.turns) for conv in conversations)
        scenarios = {}
        difficulties = {}

        for conv in conversations:
            scenarios[conv.scenario] = scenarios.get(conv.scenario, 0) + 1
            difficulties[conv.difficulty] = difficulties.get(conv.difficulty, 0) + 1

        return {
            "total_conversations": len(conversations),
            "total_turns": total_turns,
            "avg_turns_per_conversation": total_turns / len(conversations) if conversations else 0,
            "scenarios": scenarios,
            "difficulties": difficulties,
            "date": date or "all",
        }

    def export_for_analysis(
        self,
        conversations: List[ConversationLog],
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export conversations for external analysis.

        Args:
            conversations: List of ConversationLog objects
            output_path: Output file path
            format: Export format (json, csv)

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [conv.to_dict() for conv in conversations]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            import csv

            # Flatten conversations to turns
            turns = self.extract_turns_for_evaluation(conversations)

            if turns:
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=turns[0].keys())
                    writer.writeheader()
                    writer.writerows(turns)

        return str(output_path)


def create_mock_conversation(
    session_id: str,
    scenario: str = "job_interview",
    difficulty: str = "medium",
    n_turns: int = 5
) -> ConversationLog:
    """
    Create a mock conversation for testing.

    Args:
        session_id: Session ID
        scenario: Scenario name
        difficulty: Difficulty level
        n_turns: Number of turns

    Returns:
        ConversationLog object
    """
    turns = []
    for i in range(n_turns):
        turns.append({
            "turn_number": i + 1,
            "user_input": f"User input for turn {i + 1}",
            "ai_response": f"AI response for turn {i + 1}",
            "correction_enabled": False,
            "timestamp": datetime.now().isoformat(),
        })

    return ConversationLog(
        session_id=session_id,
        user_id="test_user",
        scenario=scenario,
        difficulty=difficulty,
        turn_limit=20,
        timestamp=datetime.now().isoformat(),
        turns=turns,
        system_prompt="You are a helpful language tutor.",
        metadata={
            "test": True,
        }
    )


# Convenience functions
def sample_daily_conversations(
    date: Optional[str] = None,
    n: int = 50,
    log_dir: Optional[str] = None
) -> List[ConversationLog]:
    """
    Sample conversations for daily evaluation.

    Args:
        date: Date string (YYYY-MM-DD). If None, uses yesterday.
        n: Number of samples
        log_dir: Log directory

    Returns:
        List of sampled ConversationLog objects
    """
    manager = ConversationLogManager(log_dir)

    if date is None:
        return manager.sample_previous_day(n)
    else:
        return manager.sample_conversations(date, n)
