"""
RAG (Retrieval-Augmented Generation) module for ASIL project.

Provides web search capabilities to augment football predictions with
real-time information about injuries, transfers, team news, etc.

Components:
- WebSearchRAG: Main interface for web search with match context generation
- SmartWebSearch: Multi-source search with Tavily + DuckDuckGo fallback
- SearchCache: Persistent SQLite cache for search results
"""

from src.rag.web_search_rag import WebSearchRAG
from src.rag.smart_web_search import SmartWebSearch
from src.rag.search_cache import SearchCache

__all__ = ["WebSearchRAG", "SmartWebSearch", "SearchCache"]
