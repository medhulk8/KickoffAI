"""
Smart Web Search with Multi-Source Fallback

Provides intelligent web search with:
- Session-level deduplication
- Persistent SQLite caching
- Tavily as primary source (rate-limited, high quality)
- DuckDuckGo as unlimited fallback
"""

import os
from typing import Dict, Any, List, Optional, Set
from tavily import TavilyClient
from duckduckgo_search import DDGS
from .search_cache import SearchCache


class SmartWebSearch:
    """Smart web search with caching and multi-source fallback."""

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        cache_ttl_hours: int = 24,
        max_tavily_per_session: int = 50
    ):
        """
        Initialize smart web search.

        Args:
            tavily_api_key: Tavily API key (defaults to env var)
            cache_ttl_hours: Cache TTL in hours
            max_tavily_per_session: Max Tavily calls per session
        """
        # Initialize Tavily client
        self.tavily_api_key = tavily_api_key or os.getenv('TAVILY_API_KEY')
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key) if self.tavily_api_key else None

        # Initialize cache
        self.cache = SearchCache(ttl_hours=cache_ttl_hours)

        # Session tracking
        self.session_queries: Set[str] = set()  # Queries made this session
        self.tavily_calls_this_session = 0
        self.max_tavily_per_session = max_tavily_per_session

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'tavily_calls': 0,
            'duckduckgo_calls': 0,
            'session_dedupes': 0,
            'tavily_errors': 0
        }

    def search(
        self,
        query: str,
        max_results: int = 5,
        force_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform smart web search with fallback.

        Priority: Session Cache → DB Cache → Tavily → DuckDuckGo

        Args:
            query: Search query
            max_results: Maximum results to return
            force_source: Force specific source ('tavily' or 'duckduckgo')

        Returns:
            Dict with results, source, and metadata
        """
        query_normalized = query.lower().strip()

        # 1. Session deduplication
        if query_normalized in self.session_queries:
            cached = self.cache.get(query)
            if cached:
                self.stats['session_dedupes'] += 1
                return {
                    'results': cached['results'][:max_results],
                    'source': 'session_cache',
                    'original_source': cached['source'],
                    'cached': True
                }

        # 2. Check DB cache (unless forcing specific source)
        if not force_source:
            cached = self.cache.get(query)
            if cached:
                self.stats['cache_hits'] += 1
                self.session_queries.add(query_normalized)
                return {
                    'results': cached['results'][:max_results],
                    'source': 'db_cache',
                    'original_source': cached['source'],
                    'cached': True,
                    'cached_at': cached['cached_at']
                }

        # 3. Try Tavily first (if available and under limit)
        if force_source != 'duckduckgo' and self._can_use_tavily():
            tavily_result = self._search_tavily(query, max_results)
            if tavily_result:
                self.session_queries.add(query_normalized)
                return tavily_result

        # 4. Fall back to DuckDuckGo
        ddg_result = self._search_duckduckgo(query, max_results)
        self.session_queries.add(query_normalized)
        return ddg_result

    def _can_use_tavily(self) -> bool:
        """Check if Tavily can be used."""
        return (
            self.tavily_client is not None and
            self.tavily_calls_this_session < self.max_tavily_per_session
        )

    def _search_tavily(self, query: str, max_results: int) -> Optional[Dict[str, Any]]:
        """Search using Tavily API."""
        try:
            response = self.tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="basic"
            )

            results = []
            for item in response.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'content': item.get('content', ''),
                    'score': item.get('score', 0)
                })

            # Update tracking
            self.tavily_calls_this_session += 1
            self.stats['tavily_calls'] += 1

            # Cache results
            self.cache.set(query, results, 'tavily')

            return {
                'results': results,
                'source': 'tavily',
                'cached': False,
                'tavily_calls_remaining': self.max_tavily_per_session - self.tavily_calls_this_session
            }

        except Exception as e:
            self.stats['tavily_errors'] += 1
            print(f"Tavily error (falling back to DuckDuckGo): {e}")
            return None

    def _search_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using DuckDuckGo (unlimited, free)."""
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))

            results = []
            for item in raw_results:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('href', ''),
                    'content': item.get('body', ''),
                    'score': 0.5  # Default score for DDG results
                })

            self.stats['duckduckgo_calls'] += 1

            # Cache results
            self.cache.set(query, results, 'duckduckgo')

            return {
                'results': results,
                'source': 'duckduckgo',
                'cached': False
            }

        except Exception as e:
            print(f"DuckDuckGo error: {e}")
            return {
                'results': [],
                'source': 'duckduckgo',
                'error': str(e),
                'cached': False
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        cache_stats = self.cache.get_stats()
        return {
            'session': {
                'queries_made': len(self.session_queries),
                'tavily_calls': self.tavily_calls_this_session,
                'tavily_remaining': self.max_tavily_per_session - self.tavily_calls_this_session,
                **self.stats
            },
            'cache': cache_stats
        }

    def reset_session(self):
        """Reset session tracking (for new evaluation run)."""
        self.session_queries.clear()
        self.tavily_calls_this_session = 0
        self.stats = {
            'cache_hits': 0,
            'tavily_calls': 0,
            'duckduckgo_calls': 0,
            'session_dedupes': 0,
            'tavily_errors': 0
        }
