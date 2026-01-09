"""
Conversation configuration for Phase 1: Foundation & Customization.

This module defines configuration options for:
- Turn Control: Number of conversation turns before feedback
- Difficulty Scaling: Language complexity levels (A1-C2)
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DifficultyLevel(Enum):
    """CEFR-based difficulty levels for conversation complexity."""
    PRIMARY = "primary"      # A1/A2 - Beginner
    MEDIUM = "medium"        # B1/B2 - Intermediate
    ADVANCED = "advanced"    # C1/C2 - Advanced


class TurnOption(Enum):
    """Predefined turn count options for conversation length."""
    SHORT = 10
    STANDARD = 20
    EXTENDED = 30
    DEEP_DIVE = 50


# Difficulty level descriptions for prompt injection
DIFFICULTY_DESCRIPTIONS = {
    DifficultyLevel.PRIMARY: {
        "name": "Primary (A1/A2)",
        "description": "Use simple vocabulary and basic grammar structures. Speak slowly and clearly. Avoid complex idioms and slang. Provide more guidance and hints.",
        "vocabulary": "basic, everyday vocabulary (approximately 1000-2000 words)",
        "grammar": "simple present, past, and future tenses; basic sentence structures",
        "speaking_speed": "slow and clear pronunciation",
    },
    DifficultyLevel.MEDIUM: {
        "name": "Medium (B1/B2)",
        "description": "Use moderate vocabulary with some idiomatic expressions. Speak at a normal conversational pace. Include some complex grammar structures.",
        "vocabulary": "intermediate vocabulary with common idioms (approximately 3000-5000 words)",
        "grammar": "compound and complex sentences; conditional structures; passive voice",
        "speaking_speed": "normal conversational pace",
    },
    DifficultyLevel.ADVANCED: {
        "name": "Advanced (C1/C2)",
        "description": "Use sophisticated vocabulary including technical terms and advanced idioms. Speak at natural speed with complex sentence structures. Challenge the learner with nuanced expressions.",
        "vocabulary": "advanced vocabulary with technical terms and sophisticated idioms",
        "grammar": "all grammatical structures including subjunctive, inversion, and complex clause combinations",
        "speaking_speed": "natural native-speaker pace with varied intonation",
    },
}


@dataclass
class ConversationConfig:
    """
    Configuration for a conversation session.

    Attributes:
        turns: Number of conversation turns before providing feedback
        difficulty: Language difficulty level (CEFR-based)
    """
    turns: int = TurnOption.STANDARD.value
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM

    def get_difficulty_info(self) -> dict:
        """Get the detailed description for the current difficulty level."""
        return DIFFICULTY_DESCRIPTIONS[self.difficulty]

    def to_template_vars(self) -> dict:
        """
        Convert configuration to template variables for Jinja2 rendering.

        Returns:
            dict: Variables to inject into Jinja2 templates
        """
        diff_info = self.get_difficulty_info()
        return {
            "turns": self.turns,
            "difficulty_level": diff_info["name"],
            "difficulty_description": diff_info["description"],
            "vocabulary_level": diff_info["vocabulary"],
            "grammar_level": diff_info["grammar"],
            "speaking_speed": diff_info["speaking_speed"],
        }


def get_default_config() -> ConversationConfig:
    """Get the default conversation configuration."""
    return ConversationConfig()


def create_config(
    turns: Optional[int] = None,
    difficulty: Optional[str] = None
) -> ConversationConfig:
    """
    Create a conversation configuration with optional overrides.

    Args:
        turns: Number of conversation turns (10, 20, 30, or 50)
        difficulty: Difficulty level string ("primary", "medium", or "advanced")

    Returns:
        ConversationConfig: Configured instance

    Raises:
        ValueError: If invalid turns or difficulty value is provided
    """
    config = ConversationConfig()

    if turns is not None:
        valid_turns = [t.value for t in TurnOption]
        if turns not in valid_turns:
            raise ValueError(f"Invalid turns value: {turns}. Must be one of {valid_turns}")
        config.turns = turns

    if difficulty is not None:
        try:
            config.difficulty = DifficultyLevel(difficulty.lower())
        except ValueError:
            valid_difficulties = [d.value for d in DifficultyLevel]
            raise ValueError(f"Invalid difficulty: {difficulty}. Must be one of {valid_difficulties}")

    return config
