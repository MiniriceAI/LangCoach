"""
Unit tests for conversation_config module - Phase 1 functionality.
"""
import pytest

from src.agents.conversation_config import (
    ConversationConfig,
    DifficultyLevel,
    TurnOption,
    DIFFICULTY_DESCRIPTIONS,
    create_config,
    get_default_config,
)


class TestDifficultyLevel:
    """Test cases for DifficultyLevel enum"""

    def test_difficulty_levels_exist(self):
        """Test that all expected difficulty levels exist"""
        assert DifficultyLevel.PRIMARY.value == "primary"
        assert DifficultyLevel.MEDIUM.value == "medium"
        assert DifficultyLevel.ADVANCED.value == "advanced"

    def test_difficulty_descriptions_complete(self):
        """Test that all difficulty levels have descriptions"""
        for level in DifficultyLevel:
            assert level in DIFFICULTY_DESCRIPTIONS
            desc = DIFFICULTY_DESCRIPTIONS[level]
            assert "name" in desc
            assert "description" in desc
            assert "vocabulary" in desc
            assert "grammar" in desc
            assert "speaking_speed" in desc


class TestTurnOption:
    """Test cases for TurnOption enum"""

    def test_turn_options_values(self):
        """Test that turn options have correct values"""
        assert TurnOption.SHORT.value == 10
        assert TurnOption.STANDARD.value == 20
        assert TurnOption.EXTENDED.value == 30
        assert TurnOption.DEEP_DIVE.value == 50


class TestConversationConfig:
    """Test cases for ConversationConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ConversationConfig()
        assert config.turns == 20  # TurnOption.STANDARD.value
        assert config.difficulty == DifficultyLevel.MEDIUM

    def test_custom_config(self):
        """Test custom configuration values"""
        config = ConversationConfig(
            turns=10,
            difficulty=DifficultyLevel.PRIMARY
        )
        assert config.turns == 10
        assert config.difficulty == DifficultyLevel.PRIMARY

    def test_get_difficulty_info(self):
        """Test getting difficulty information"""
        config = ConversationConfig(difficulty=DifficultyLevel.PRIMARY)
        info = config.get_difficulty_info()

        assert info["name"] == "Primary (A1/A2)"
        assert "simple vocabulary" in info["description"].lower()

    def test_to_template_vars(self):
        """Test converting config to template variables"""
        config = ConversationConfig(turns=30, difficulty=DifficultyLevel.ADVANCED)
        vars = config.to_template_vars()

        assert vars["turns"] == 30
        assert vars["difficulty_level"] == "Advanced (C1/C2)"
        assert "vocabulary_level" in vars
        assert "grammar_level" in vars
        assert "speaking_speed" in vars


class TestCreateConfig:
    """Test cases for create_config function"""

    def test_create_config_defaults(self):
        """Test creating config with defaults"""
        config = create_config()
        assert config.turns == 20
        assert config.difficulty == DifficultyLevel.MEDIUM

    def test_create_config_with_turns(self):
        """Test creating config with custom turns"""
        config = create_config(turns=30)
        assert config.turns == 30
        assert config.difficulty == DifficultyLevel.MEDIUM

    def test_create_config_with_difficulty(self):
        """Test creating config with custom difficulty"""
        config = create_config(difficulty="primary")
        assert config.turns == 20
        assert config.difficulty == DifficultyLevel.PRIMARY

    def test_create_config_with_all_params(self):
        """Test creating config with all parameters"""
        config = create_config(turns=50, difficulty="advanced")
        assert config.turns == 50
        assert config.difficulty == DifficultyLevel.ADVANCED

    def test_create_config_invalid_turns(self):
        """Test creating config with invalid turns value"""
        with pytest.raises(ValueError, match="Invalid turns value"):
            create_config(turns=15)  # 15 is not a valid option

    def test_create_config_invalid_difficulty(self):
        """Test creating config with invalid difficulty"""
        with pytest.raises(ValueError, match="Invalid difficulty"):
            create_config(difficulty="expert")  # "expert" is not valid

    def test_create_config_case_insensitive_difficulty(self):
        """Test that difficulty is case insensitive"""
        config = create_config(difficulty="PRIMARY")
        assert config.difficulty == DifficultyLevel.PRIMARY

        config = create_config(difficulty="Medium")
        assert config.difficulty == DifficultyLevel.MEDIUM


class TestGetDefaultConfig:
    """Test cases for get_default_config function"""

    def test_get_default_config(self):
        """Test getting default config"""
        config = get_default_config()
        assert isinstance(config, ConversationConfig)
        assert config.turns == 20
        assert config.difficulty == DifficultyLevel.MEDIUM
