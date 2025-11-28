"""
Unit tests for logger module
"""
import pytest
import sys
from unittest.mock import patch, MagicMock

from src.utils.logger import LOG, log_format


class TestLogger:
    """Test cases for logger module"""

    def test_log_format_defined(self):
        """Test that log format is defined"""
        assert log_format is not None
        assert isinstance(log_format, str)
        assert "time" in log_format
        assert "level" in log_format
        assert "message" in log_format

    def test_log_imported(self):
        """Test that LOG is imported correctly"""
        assert LOG is not None
        assert hasattr(LOG, "debug")
        assert hasattr(LOG, "info")
        assert hasattr(LOG, "warning")
        assert hasattr(LOG, "error")

    def test_log_methods_callable(self):
        """Test that logger methods are callable"""
        # These should not raise exceptions
        LOG.debug("Test debug message")
        LOG.info("Test info message")
        LOG.warning("Test warning message")
        LOG.error("Test error message")

    def test_log_all_exported(self):
        """Test that __all__ includes LOG"""
        from src.utils.logger import __all__
        assert "LOG" in __all__

