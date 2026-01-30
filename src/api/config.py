"""
Configuration management for LangCoach Mini Program API.

Centralizes all configuration settings to eliminate hardcoded values.
"""
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue with environment variables only


@dataclass
class ServiceConfig:
    """Service-specific configuration settings."""
    
    # Ollama/LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "hf.co/unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL"
    ollama_temperature: float = 0.8
    ollama_num_predict: int = 512
    ollama_timeout: int = 5
    ollama_stop_tokens: List[str] = field(default_factory=lambda: ["Human:", "Assistant:", "User:"])
    
    # STT Configuration
    stt_model: str = "unsloth/whisper-large-v3"
    stt_sample_rate: int = 16000
    
    # TTS Configuration
    tts_default_speaker: str = "Ceylia"
    tts_sample_rate: int = 24000
    tts_format: str = "mp3"
    
    # Edge-TTS Configuration
    edge_tts_voices: Dict[str, str] = field(default_factory=lambda: {
        "Ceylia": "en-US-JennyNeural",      # 美式女声 - 友好
        "Tifa": "en-US-AriaNeural",         # 美式女声 - 自然
        "David": "en-US-GuyNeural",         # 美式男声 - 温和
        "Tony": "en-US-TonyNeural",         # 美式男声 - 成熟
        "Emma": "en-GB-SoniaNeural",        # 英式女声 - 优雅
        "Ryan": "en-GB-RyanNeural",         # 英式男声 - 正式
        "Sarah": "en-AU-NatashaNeural",     # 澳式女声 - 活泼
        "William": "en-AU-WilliamNeural",   # 澳式男声 - 友好
        "default": "en-US-JennyNeural"
    })
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8600
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Session Configuration
    max_recent_messages: int = 6
    max_reply_length: int = 300
    max_reply_sentences: int = 200
    audio_cache_hours: int = 1


@dataclass
class ContentConfig:
    """Content and scenario configuration."""
    
    # Scenario Configuration
    available_scenarios: List[str] = field(default_factory=lambda: [
        "job_interview", "hotel_checkin", "renting", "salary_negotiation"
    ])
    
    # Default greetings for each scenario
    default_greetings: Dict[str, str] = field(default_factory=lambda: {
        "job_interview": "Hello! I'm your interviewer today. Please have a seat and let's begin. Could you start by telling me a little about yourself?",
        "hotel_checkin": "Good evening! Welcome to our hotel. I'll be helping you with check-in today. May I have your name and reservation details, please?",
        "renting": "Hi there! I'm the property manager. I understand you're interested in renting this apartment. Would you like me to show you around first?",
        "salary_negotiation": "Thank you for coming in today. We've reviewed your application and would like to discuss the compensation package. What are your salary expectations?",
        "default": "Hi there! I'm your English practice partner. What would you like to talk about today?"
    })
    
    # Level to difficulty mapping
    level_to_difficulty: Dict[str, str] = field(default_factory=lambda: {
        "A1": "primary", "A2": "primary",
        "B1": "medium", "B2": "medium", 
        "C1": "advanced", "C2": "advanced"
    })
    
    # Prompt template paths
    prompts_dir: str = "prompts"

    # Random scenario generation mode: "preset" or "llm"
    random_scenario_mode: str = "preset"




@dataclass 
class ScoreConfig:
    """Scoring and evaluation configuration."""
    
    # Default scoring ranges for session reports
    grammar_score_range: tuple = (70, 95)
    fluency_score_range: tuple = (70, 95)
    
    # Feedback tips pool
    feedback_tips: List[str] = field(default_factory=lambda: [
        "Try using more complex sentence structures",
        "Good use of vocabulary!",
        "Practice speaking more fluently",
        "Your grammar is improving!",
        "Consider using more varied expressions",
        "Excellent pronunciation!", 
        "Work on sentence connectors",
        "Great job staying on topic!"
    ])


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self):
        self.service = ServiceConfig()
        self.content = ContentConfig()
        self.scoring = ScoreConfig()
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        
        # Service Configuration
        self.service.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.service.ollama_base_url)
        self.service.ollama_model = os.getenv("OLLAMA_MODEL", self.service.ollama_model)
        self.service.ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", str(self.service.ollama_temperature)))
        self.service.ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", str(self.service.ollama_num_predict)))
        self.service.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", str(self.service.ollama_timeout)))
        
        self.service.stt_model = os.getenv("STT_MODEL", self.service.stt_model)
        self.service.stt_sample_rate = int(os.getenv("STT_SAMPLE_RATE", str(self.service.stt_sample_rate)))
        
        self.service.tts_default_speaker = os.getenv("TTS_DEFAULT_SPEAKER", self.service.tts_default_speaker)
        self.service.tts_sample_rate = int(os.getenv("TTS_SAMPLE_RATE", str(self.service.tts_sample_rate)))
        self.service.tts_format = os.getenv("TTS_FORMAT", self.service.tts_format)
        
        self.service.api_host = os.getenv("API_HOST", self.service.api_host)
        self.service.api_port = int(os.getenv("API_PORT", str(self.service.api_port)))
        
        # Session settings
        self.service.max_recent_messages = int(os.getenv("MAX_RECENT_MESSAGES", str(self.service.max_recent_messages)))
        self.service.max_reply_length = int(os.getenv("MAX_REPLY_LENGTH", str(self.service.max_reply_length)))
        self.service.max_reply_sentences = int(os.getenv("MAX_REPLY_SENTENCES", str(self.service.max_reply_sentences)))
        self.service.audio_cache_hours = int(os.getenv("AUDIO_CACHE_HOURS", str(self.service.audio_cache_hours)))
        
        # Content Configuration
        self.content.prompts_dir = os.getenv("PROMPTS_DIR", self.content.prompts_dir)
        self.content.random_scenario_mode = os.getenv("RANDOM_SCENARIO_MODE", self.content.random_scenario_mode)

        # Parse comma-separated values
        cors_origins = os.getenv("CORS_ORIGINS")
        if cors_origins:
            self.service.cors_origins = [origin.strip() for origin in cors_origins.split(",")]
            
        scenarios = os.getenv("AVAILABLE_SCENARIOS")
        if scenarios:
            self.content.available_scenarios = [scenario.strip() for scenario in scenarios.split(",")]
    
    def get_ollama_url(self, endpoint: str = "") -> str:
        """Get complete Ollama URL with optional endpoint."""
        return f"{self.service.ollama_base_url.rstrip('/')}/{endpoint.lstrip('/')}" if endpoint else self.service.ollama_base_url
    
    def get_prompt_path(self, scenario: str) -> str:
        """Get the full path to a scenario prompt file."""
        # If prompts_dir is relative, resolve it relative to the project root
        if os.path.isabs(self.content.prompts_dir):
            prompts_dir = self.content.prompts_dir
        else:
            # Assume we're running from /workspace/LangCoach
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            prompts_dir = os.path.join(project_root, self.content.prompts_dir)
        
        return os.path.join(prompts_dir, f"{scenario}_prompt.txt")
    
    def is_scenario_available(self, scenario: str) -> bool:
        """Check if a scenario is available."""
        return scenario in self.content.available_scenarios or scenario == "default"
    
    def get_edge_tts_voice(self, speaker: str) -> str:
        """Get Edge-TTS voice name for a speaker."""
        return self.service.edge_tts_voices.get(speaker, self.service.edge_tts_voices["default"])
    
    def get_difficulty_for_level(self, level: str) -> str:
        """Convert CEFR level to difficulty."""
        return self.content.level_to_difficulty.get(level, "medium")
    
    def generate_random_scores(self) -> Dict[str, int]:
        """Generate realistic random scores for session reports."""
        import random
        return {
            "grammarScore": random.randint(*self.scoring.grammar_score_range),
            "fluencyScore": random.randint(*self.scoring.fluency_score_range)
        }
    
    def get_random_tips(self, count: int = 3) -> List[str]:
        """Get random feedback tips."""
        import random
        return random.sample(self.scoring.feedback_tips, min(count, len(self.scoring.feedback_tips)))


# Global configuration instance
config = ConfigManager()