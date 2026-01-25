"""
Web Search RAG for Football Predictions

This module provides web search capabilities with smart fallback:
- Primary: Tavily API (high quality, rate-limited)
- Fallback: DuckDuckGo (unlimited, free)
- Persistent SQLite caching with TTL

Features:
- Automatic query generation for match context
- Multi-level caching (session + SQLite)
- Relevance-based result filtering
- Formatted context for LLM consumption
- Token budget management

Usage:
    from src.rag import WebSearchRAG

    rag = WebSearchRAG(tavily_api_key="your_api_key")
    context = rag.get_match_context("Liverpool", "Arsenal", "2024-01-15")
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from .smart_web_search import SmartWebSearch


class WebSearchRAG:
    """
    Web Search RAG client with smart multi-source search.

    Uses SmartWebSearch for intelligent caching and fallback:
    - Priority: Session Cache → DB Cache → Tavily → DuckDuckGo
    - Persistent SQLite caching reduces API calls
    - DuckDuckGo fallback when Tavily quota exhausted

    Attributes:
        smart_search: SmartWebSearch instance
        search_cache: Dict for session-level caching (legacy compatibility)
    """

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        cache_ttl_hours: int = 24,
        max_tavily_per_session: int = 50
    ):
        """
        Initialize WebSearchRAG with smart multi-source search.

        Args:
            tavily_api_key: API key from tavily.com
                           If not provided, reads from TAVILY_API_KEY env var
            cache_ttl_hours: How long to cache results (default 24 hours)
            max_tavily_per_session: Max Tavily calls per session (budget control)

        Note:
            No longer raises ValueError if API key missing - will use DuckDuckGo fallback
        """
        # Initialize smart search with multi-source fallback
        self.smart_search = SmartWebSearch(
            tavily_api_key=tavily_api_key,
            cache_ttl_hours=cache_ttl_hours,
            max_tavily_per_session=max_tavily_per_session
        )

        # Legacy compatibility - keep session cache dict
        self.search_cache: Dict[str, Dict[str, Any]] = {}

    def execute_searches(
        self,
        queries: List[str],
        max_searches: int = 5,
        max_results_per_query: int = 5,
        kg_insights: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute multiple searches using smart multi-source search.

        For each query:
        1. Check session cache (same query this session)
        2. Check SQLite cache (persistent across sessions)
        3. Try Tavily API (high quality, rate-limited)
        4. Fallback to DuckDuckGo (unlimited, free)

        Args:
            queries: List of search queries to execute
            max_searches: Maximum number of searches to perform
            max_results_per_query: Maximum results per search (1-10)
            kg_insights: Optional KG insights to include in formatted output

        Returns:
            Dictionary containing:
            - queries_executed: Number of actual API calls made
            - cached_hits: Number of cache hits (session + DB)
            - results: Dict mapping query -> list of results
            - all_content: Combined content for LLM context
            - sources: Dict showing which source was used per query

        Example:
            >>> rag = WebSearchRAG(api_key="...")
            >>> results = rag.execute_searches(["Liverpool injuries 2024"])
            >>> print(results["all_content"])
        """
        results = {}
        cached_hits = 0
        api_calls = 0
        sources = {}
        queries_to_execute = queries[:max_searches]

        for query in queries_to_execute:
            # Use SmartWebSearch which handles all caching and fallback
            search_result = self.smart_search.search(
                query=query,
                max_results=max_results_per_query
            )

            # Extract results
            query_results = search_result.get('results', [])

            # Sort by relevance score
            if query_results:
                query_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Track source and caching
            source = search_result.get('source', 'unknown')
            sources[query] = source

            if search_result.get('cached', False):
                cached_hits += 1
            else:
                api_calls += 1

            # Store results (also in legacy cache for compatibility)
            results[query] = query_results
            self.search_cache[query] = query_results

        # Combine all content for LLM context (with optional KG insights)
        all_content = self._format_for_llm(results, kg_insights=kg_insights)

        return {
            "queries_executed": api_calls,
            "cached_hits": cached_hits,
            "results": results,
            "all_content": all_content,
            "sources": sources
        }

    def _format_for_llm(
        self,
        results: Dict[str, Any],
        kg_insights: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format search results into a clean context string for LLM.

        Highlights injury/suspension information prominently.
        Optionally includes KG tactical insights at the top.

        Args:
            results: Dictionary of query -> results
            kg_insights: Optional KG insights to include at top

        Returns:
            Formatted string suitable for including in LLM prompts
        """
        output_parts = []

        # Add KG insights at top if available
        if kg_insights and kg_insights.get('confidence') not in ['none', None]:
            kg_section = self._format_kg_summary(kg_insights)
            output_parts.append(kg_section)

        # Separate injury results from other results
        injury_sections = []
        other_sections = []

        for query, items in results.items():
            if isinstance(items, dict) and "error" in items:
                continue

            if not items:
                continue

            # Check if this is an injury/suspension query
            is_injury_query = any(kw in query.lower() for kw in ['injury', 'suspension', 'injured', 'suspended'])

            # Format the section
            if is_injury_query:
                section = f"### INJURY/SUSPENSION UPDATE: {query}\n"
            else:
                section = f"### Search: {query}\n"

            for i, item in enumerate(items[:3], 1):  # Top 3 per query
                title = item.get("title", "Untitled")
                content = item.get("content", "")
                url = item.get("url", "")

                # Truncate content if too long
                if len(content) > 500:
                    content = content[:500] + "..."

                section += f"\n**{i}. {title}**\n"
                section += f"{content}\n"
                section += f"Source: {url}\n"

            if is_injury_query:
                injury_sections.append(section)
            else:
                other_sections.append(section)

        # Prioritize injury information
        if injury_sections:
            output_parts.append("## CRITICAL: INJURY & SUSPENSION NEWS\n")
            output_parts.extend(injury_sections)

        if other_sections:
            if injury_sections:
                output_parts.append("\n## OTHER CONTEXT\n")
            output_parts.extend(other_sections)

        if not injury_sections and not other_sections:
            return "No relevant web search results found."

        return "\n---\n".join(output_parts)

    def _format_kg_summary(self, kg_insights: Dict[str, Any]) -> str:
        """
        Format KG insights into a summary section.

        Args:
            kg_insights: Knowledge graph tactical insights

        Returns:
            Formatted KG summary string
        """
        home_styles = kg_insights.get('home_styles', [])
        away_styles = kg_insights.get('away_styles', [])
        matchup = kg_insights.get('matchup_summary', 'No clear tactical advantage')
        confidence = kg_insights.get('confidence', 'none').upper()

        home_style_str = ', '.join(home_styles) if home_styles else 'Unknown'
        away_style_str = ', '.join(away_styles) if away_styles else 'Unknown'

        return f"""## TACTICAL CONTEXT (from Knowledge Graph)
- **Home Team Style:** {home_style_str}
- **Away Team Style:** {away_style_str}
- **Matchup Analysis:** {matchup}
- **KG Confidence:** {confidence}
"""

    def generate_match_queries(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None,
        kg_insights: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate relevant search queries for a match.

        Dynamically generates queries based on what information is available:
        - Always: Recent form, injuries/suspensions
        - Conditional: Tactics (only if KG has no tactical info)
        - If space: Head-to-head history

        Args:
            home_team: Name of home team
            away_team: Name of away team
            match_date: Optional match date (YYYY-MM-DD format)
            kg_insights: Optional KG insights to conditionally skip tactics search

        Returns:
            List of search queries (5-7 queries depending on KG info)
        """
        year = datetime.now().year if not match_date else match_date[:4]

        queries = []

        # Priority 1: Recent form (ALWAYS - critical for prediction)
        queries.append(f"{home_team} last 5 matches results {year}")
        queries.append(f"{away_team} last 5 matches results {year}")

        # Priority 2: Injuries/Suspensions (CRITICAL - affects predictions significantly)
        queries.append(f"{home_team} injury news latest")
        queries.append(f"{away_team} injury suspension news")

        # Priority 3: Tactics (CONDITIONAL - only if KG has no info)
        kg_confidence = 'none'
        if kg_insights:
            kg_confidence = kg_insights.get('confidence', 'none')

        if kg_confidence == 'none' or kg_confidence == 'low':
            # No/low KG info, search for tactics
            queries.append(f"{home_team} playing style tactics {year}")
            queries.append(f"{away_team} playing style tactics {year}")
        else:
            # KG has tactics, search for tactical preview instead
            queries.append(f"{home_team} vs {away_team} tactical preview {year}")

        # Priority 4: Head-to-head (if space permits)
        if len(queries) < 7:
            queries.append(f"{home_team} vs {away_team} recent meetings")

        return queries[:7]  # Limit to 7 queries max

    def generate_basic_queries(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None
    ) -> List[str]:
        """
        Generate basic queries without KG awareness (backward compatible).

        Args:
            home_team: Name of home team
            away_team: Name of away team
            match_date: Optional match date (YYYY-MM-DD format)

        Returns:
            List of 5 search queries
        """
        return self.generate_match_queries(home_team, away_team, match_date, kg_insights=None)

    def get_match_context(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None,
        max_searches: int = 5,
        kg_insights: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive web context for a match.

        Convenience method that:
        1. Generates relevant queries for the match (conditionally based on KG)
        2. Executes searches
        3. Returns formatted context with injury info highlighted

        Args:
            home_team: Name of home team
            away_team: Name of away team
            match_date: Optional match date (YYYY-MM-DD)
            max_searches: Maximum searches to perform
            kg_insights: Optional KG insights to conditionally skip tactics search

        Returns:
            Dictionary with search results and formatted context

        Example:
            >>> rag = WebSearchRAG(api_key="...")
            >>> context = rag.get_match_context("Liverpool", "Arsenal")
            >>> print(context["all_content"])
        """
        queries = self.generate_match_queries(
            home_team, away_team, match_date, kg_insights=kg_insights
        )
        return self.execute_searches(
            queries, max_searches=max_searches, kg_insights=kg_insights
        )

    def search_single(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a single search query.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of search result dictionaries
        """
        result = self.execute_searches([query], max_results_per_query=max_results)
        return result["results"].get(query, [])

    def clear_cache(self):
        """Clear both session and persistent cache."""
        self.search_cache.clear()
        self.smart_search.cache.clear()

    def reset_session(self):
        """Reset session tracking (for new evaluation run)."""
        self.search_cache.clear()
        self.smart_search.reset_session()

    def get_search_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        return self.smart_search.get_stats()


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print(" WEB SEARCH RAG TEST (Tavily)")
    print("=" * 70)

    # Get API key from environment or command line
    api_key = os.environ.get("TAVILY_API_KEY")

    if len(sys.argv) > 1:
        api_key = sys.argv[1]

    if not api_key:
        print("\nNo API key found!")
        print("\nTo run tests, either:")
        print("  1. Set TAVILY_API_KEY environment variable")
        print("  2. Pass API key as command line argument:")
        print("     python web_search_rag.py YOUR_API_KEY")
        print("\nGet a free API key at: https://tavily.com")
        sys.exit(1)

    print(f"\nAPI Key: {api_key[:10]}...{api_key[-4:]}")

    # Initialize RAG
    print("\nInitializing WebSearchRAG...")
    rag = WebSearchRAG(tavily_api_key=api_key)
    print("Initialized successfully!")

    # Test 1: Single search
    print("\n" + "=" * 70)
    print(" TEST 1: Single Search - Liverpool Recent Form")
    print("=" * 70)

    query = "Liverpool recent form 2024 Premier League"
    print(f"\nQuery: {query}")
    print("\nSearching...")

    results = rag.search_single(query, max_results=5)

    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   URL: {result['url']}")
        content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        print(f"   Content: {content_preview}")

    # Test 2: Match context
    print("\n" + "=" * 70)
    print(" TEST 2: Full Match Context - Liverpool vs Arsenal")
    print("=" * 70)

    print("\nGenerating match context...")
    context = rag.get_match_context("Liverpool", "Arsenal", max_searches=3)

    print(f"\nQueries executed: {context['queries_executed']}")
    print(f"Cache hits: {context['cached_hits']}")

    print("\n" + "-" * 70)
    print(" FORMATTED CONTEXT FOR LLM")
    print("-" * 70)
    print(context["all_content"])

    # Test 3: Cache test
    print("\n" + "=" * 70)
    print(" TEST 3: Cache Test")
    print("=" * 70)

    print("\nRunning same query again...")
    context2 = rag.get_match_context("Liverpool", "Arsenal", max_searches=3)
    print(f"Queries executed: {context2['queries_executed']} (should be 0)")
    print(f"Cache hits: {context2['cached_hits']} (should be 3)")

    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETED!")
    print("=" * 70)
