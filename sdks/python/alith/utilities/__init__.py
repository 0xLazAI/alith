"""
Utilities module for Alith SDK.

This module provides various utility tools and functions that can be used
with AI agents for common tasks like search, file operations, web scraping, etc.
"""

from .search import DuckDuckGoTool

__all__ = [
    # Search utilities
    "DuckDuckGoTool",
]
