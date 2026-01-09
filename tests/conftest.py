"""
Pytest configuration and shared fixtures
"""
import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key-12345")
    return {"DEEPSEEK_API_KEY": "test-api-key-12345"}

@pytest.fixture
def mock_chat_openai():
    """Mock ChatOpenAI for testing"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test response")
    return mock

@pytest.fixture
def sample_prompt_file(tmp_path):
    """Create a sample prompt file for testing"""
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("You are a helpful assistant.")
    return str(prompt_file)

@pytest.fixture
def sample_intro_file(tmp_path):
    """Create a sample intro JSON file for testing"""
    import json
    intro_file = tmp_path / "test_intro.json"
    intro_data = ["Hello", "Hi there", "Welcome"]
    intro_file.write_text(json.dumps(intro_data), encoding="utf-8")
    return str(intro_file)

@pytest.fixture
def sample_jinja2_template_dir(tmp_path):
    """Create a sample Jinja2 templates directory for testing"""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    # Create a sample Jinja2 template
    template_content = """**System Prompt**

**Configuration:**
- Conversation Length: {{ turns }} rounds
- Difficulty Level: {{ difficulty_level }}

**Role**: You are a helpful assistant.

**Language Difficulty:**
{{ difficulty_description }}
- Vocabulary: {{ vocabulary_level }}
- Grammar: {{ grammar_level }}
- Speaking Speed: {{ speaking_speed }}

**Task**: After **{{ turns }} rounds**, provide feedback.
"""
    template_file = templates_dir / "test_prompt.j2"
    template_file.write_text(template_content, encoding="utf-8")

    return str(templates_dir)

@pytest.fixture
def clean_session_store():
    """Clear session store before and after each test"""
    from src.agents.session_history import store
    store.clear()
    yield
    store.clear()

