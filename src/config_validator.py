"""
Environment configuration validation for LangCoach.

Validates required environment variables and provides helpful error messages.
"""

import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def validate_database_config() -> Tuple[bool, List[str]]:
    """
    Validate database configuration.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        errors.append("DATABASE_URL is not set. Please configure database connection.")
    elif database_url == "postgresql://user:password@localhost:5432/langcoach":
        errors.append("DATABASE_URL is using default placeholder values. Please update with actual credentials.")

    return len(errors) == 0, errors


def validate_jwt_config() -> Tuple[bool, List[str]]:
    """
    Validate JWT authentication configuration.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    jwt_secret = os.getenv("JWT_SECRET_KEY")

    if not jwt_secret:
        errors.append("JWT_SECRET_KEY is not set. Authentication will not work.")
    elif jwt_secret == "your-secret-key-change-this-in-production" or len(jwt_secret) < 32:
        errors.append("JWT_SECRET_KEY is insecure. Please use a strong random key (32+ characters).")

    jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    if jwt_algorithm not in ["HS256", "HS384", "HS512"]:
        errors.append(f"JWT_ALGORITHM '{jwt_algorithm}' is not supported. Use HS256, HS384, or HS512.")

    return len(errors) == 0, errors


def validate_wechat_config() -> Tuple[bool, List[str]]:
    """
    Validate WeChat Mini Program configuration.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    warnings = []
    app_id = os.getenv("WECHAT_APP_ID")
    app_secret = os.getenv("WECHAT_APP_SECRET")

    if not app_id or app_id == "your_wechat_app_id_here":
        warnings.append("WECHAT_APP_ID is not configured. WeChat login will use mock data.")

    if not app_secret or app_secret == "your_wechat_app_secret_here":
        warnings.append("WECHAT_APP_SECRET is not configured. WeChat login will use mock data.")

    # WeChat config is optional for development, so return True but log warnings
    if warnings:
        logger.warning("WeChat configuration warnings: " + "; ".join(warnings))

    return True, warnings


def validate_llm_config() -> Tuple[bool, List[str]]:
    """
    Validate LLM provider configuration.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    warnings = []
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")

    if llm_provider == "ollama":
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if not ollama_url:
            warnings.append("OLLAMA_BASE_URL is not set. Using default: http://localhost:11434")
    elif llm_provider == "deepseek":
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_key or deepseek_key.startswith("sk-"):
            warnings.append("DEEPSEEK_API_KEY may not be configured correctly.")
    elif llm_provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your_openai_api_key_here":
            warnings.append("OPENAI_API_KEY is not configured.")

    if warnings:
        logger.warning("LLM configuration warnings: " + "; ".join(warnings))

    return True, warnings


def validate_all_config(strict: bool = False) -> bool:
    """
    Validate all configuration.

    Args:
        strict: If True, raise exception on any error. If False, only log warnings.

    Returns:
        True if all critical config is valid

    Raises:
        ConfigValidationError: If strict=True and validation fails
    """
    all_errors = []
    all_warnings = []

    # Validate critical configurations
    db_valid, db_errors = validate_database_config()
    if not db_valid:
        all_errors.extend(db_errors)

    jwt_valid, jwt_errors = validate_jwt_config()
    if not jwt_valid:
        all_errors.extend(jwt_errors)

    # Validate optional configurations
    _, wechat_warnings = validate_wechat_config()
    all_warnings.extend(wechat_warnings)

    _, llm_warnings = validate_llm_config()
    all_warnings.extend(llm_warnings)

    # Log results
    if all_errors:
        error_msg = "Configuration errors found:\n" + "\n".join(f"  - {e}" for e in all_errors)
        logger.error(error_msg)
        if strict:
            raise ConfigValidationError(error_msg)
        return False

    if all_warnings:
        warning_msg = "Configuration warnings:\n" + "\n".join(f"  - {w}" for w in all_warnings)
        logger.warning(warning_msg)

    logger.info("Configuration validation passed")
    return True


def print_config_status():
    """Print configuration status for debugging."""
    print("\n" + "="*60)
    print("LangCoach Configuration Status")
    print("="*60)

    # Database
    db_valid, db_errors = validate_database_config()
    print(f"\n📊 Database: {'✅ OK' if db_valid else '❌ ERROR'}")
    if db_errors:
        for error in db_errors:
            print(f"   - {error}")

    # JWT
    jwt_valid, jwt_errors = validate_jwt_config()
    print(f"\n🔐 JWT Auth: {'✅ OK' if jwt_valid else '❌ ERROR'}")
    if jwt_errors:
        for error in jwt_errors:
            print(f"   - {error}")

    # WeChat
    _, wechat_warnings = validate_wechat_config()
    print(f"\n💬 WeChat: {'✅ OK' if not wechat_warnings else '⚠️  WARNING'}")
    if wechat_warnings:
        for warning in wechat_warnings:
            print(f"   - {warning}")

    # LLM
    _, llm_warnings = validate_llm_config()
    print(f"\n🤖 LLM: {'✅ OK' if not llm_warnings else '⚠️  WARNING'}")
    if llm_warnings:
        for warning in llm_warnings:
            print(f"   - {warning}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Run validation when executed directly
    print_config_status()
    try:
        validate_all_config(strict=True)
        print("✅ All critical configuration is valid!")
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed:\n{e}")
        exit(1)
