"""
Search Cache for Smart Web Search

Provides persistent SQLite-based caching for web search results with TTL support.
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List


class SearchCache:
    """Persistent cache for web search results using SQLite."""

    def __init__(self, cache_dir: str = "data/cache", ttl_hours: int = 24):
        """
        Initialize the search cache.

        Args:
            cache_dir: Directory to store cache database
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "search_cache.db"
        self.ttl = timedelta(hours=ttl_hours)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with search cache table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    results TEXT,
                    source TEXT,
                    created_at TEXT,
                    expires_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON search_cache(expires_at)
            """)
            conn.commit()

    def _hash_query(self, query: str) -> str:
        """Generate hash for query to use as cache key."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached search results if available and not expired.

        Args:
            query: Search query string

        Returns:
            Cached results dict or None if not found/expired
        """
        query_hash = self._hash_query(query)
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT results, source, created_at
                FROM search_cache
                WHERE query_hash = ? AND expires_at > ?
            """, (query_hash, now))
            row = cursor.fetchone()

            if row:
                return {
                    'results': json.loads(row[0]),
                    'source': row[1],
                    'cached': True,
                    'cached_at': row[2]
                }
        return None

    def set(self, query: str, results: List[Dict], source: str):
        """
        Store search results in cache.

        Args:
            query: Original search query
            results: List of search result dictionaries
            source: Source of results ('tavily' or 'duckduckgo')
        """
        query_hash = self._hash_query(query)
        now = datetime.now()
        expires_at = now + self.ttl

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache
                (query_hash, query, results, source, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                query_hash,
                query,
                json.dumps(results),
                source,
                now.isoformat(),
                expires_at.isoformat()
            ))
            conn.commit()

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM search_cache WHERE expires_at < ?
            """, (now,))
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            total = conn.execute(
                "SELECT COUNT(*) FROM search_cache"
            ).fetchone()[0]

            # Valid (non-expired) entries
            valid = conn.execute(
                "SELECT COUNT(*) FROM search_cache WHERE expires_at > ?",
                (now,)
            ).fetchone()[0]

            # Entries by source
            by_source = {}
            cursor = conn.execute("""
                SELECT source, COUNT(*)
                FROM search_cache
                WHERE expires_at > ?
                GROUP BY source
            """, (now,))
            for row in cursor:
                by_source[row[0]] = row[1]

        return {
            'total_entries': total,
            'valid_entries': valid,
            'expired_entries': total - valid,
            'by_source': by_source,
            'db_path': str(self.db_path)
        }

    def clear(self):
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM search_cache")
            conn.commit()
