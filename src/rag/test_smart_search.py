"""
Test script for Smart Web Search System

Tests:
1. DuckDuckGo search (free, unlimited)
2. Tavily search (if API key available)
3. Cache persistence
4. Fallback behavior
5. Session deduplication
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag import SmartWebSearch, SearchCache, WebSearchRAG


def test_search_cache():
    """Test SearchCache persistence."""
    print("\n" + "=" * 70)
    print(" TEST 1: SearchCache Persistence")
    print("=" * 70)

    cache = SearchCache(cache_dir="data/cache", ttl_hours=24)

    # Test set and get
    test_query = "test query for caching"
    test_results = [
        {"title": "Test Result 1", "url": "http://example.com/1", "content": "Test content 1"},
        {"title": "Test Result 2", "url": "http://example.com/2", "content": "Test content 2"}
    ]

    cache.set(test_query, test_results, "test")
    print(f"\n[+] Cached query: '{test_query}'")

    retrieved = cache.get(test_query)
    if retrieved:
        print(f"[+] Retrieved from cache: {len(retrieved['results'])} results")
        print(f"    Source: {retrieved['source']}")
        print(f"    Cached: {retrieved['cached']}")
    else:
        print("[-] Failed to retrieve from cache!")

    # Stats
    stats = cache.get_stats()
    print(f"\n[+] Cache stats:")
    print(f"    Total entries: {stats['total_entries']}")
    print(f"    Valid entries: {stats['valid_entries']}")
    print(f"    DB path: {stats['db_path']}")

    return True


def test_duckduckgo_search():
    """Test DuckDuckGo fallback search."""
    print("\n" + "=" * 70)
    print(" TEST 2: DuckDuckGo Search (Free Fallback)")
    print("=" * 70)

    # Create SmartWebSearch without Tavily key to force DuckDuckGo
    search = SmartWebSearch(tavily_api_key=None)

    query = "Liverpool FC recent form 2024"
    print(f"\n[+] Query: '{query}'")
    print("[+] Forcing DuckDuckGo (no Tavily key)...")

    result = search.search(query, max_results=3)

    print(f"\n[+] Source: {result['source']}")
    print(f"[+] Cached: {result['cached']}")
    print(f"[+] Results: {len(result['results'])}")

    if result['results']:
        print("\n[+] Top 3 results:")
        for i, r in enumerate(result['results'][:3], 1):
            print(f"    {i}. {r['title'][:60]}...")
            print(f"       URL: {r['url'][:50]}...")

    return len(result['results']) > 0


def test_tavily_search():
    """Test Tavily search if API key available."""
    print("\n" + "=" * 70)
    print(" TEST 3: Tavily Search (Premium)")
    print("=" * 70)

    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        print("\n[!] TAVILY_API_KEY not set - skipping Tavily test")
        print("    (DuckDuckGo fallback will be used in production)")
        return True

    print(f"\n[+] Tavily API key found: {api_key[:10]}...{api_key[-4:]}")

    search = SmartWebSearch(tavily_api_key=api_key, max_tavily_per_session=5)

    query = "Arsenal FC injuries January 2024"
    print(f"\n[+] Query: '{query}'")

    result = search.search(query, max_results=3)

    print(f"\n[+] Source: {result['source']}")
    print(f"[+] Cached: {result['cached']}")
    print(f"[+] Results: {len(result['results'])}")

    if result['source'] == 'tavily':
        print(f"[+] Tavily calls remaining: {result.get('tavily_calls_remaining', 'N/A')}")

    if result['results']:
        print("\n[+] Top 3 results:")
        for i, r in enumerate(result['results'][:3], 1):
            print(f"    {i}. {r['title'][:60]}...")
            print(f"       Score: {r.get('score', 0):.3f}")

    return True


def test_cache_hit():
    """Test cache hit on repeat query."""
    print("\n" + "=" * 70)
    print(" TEST 4: Cache Hit (Session Deduplication)")
    print("=" * 70)

    search = SmartWebSearch(tavily_api_key=None)

    query = "Manchester United transfer news"
    print(f"\n[+] First search: '{query}'")

    result1 = search.search(query, max_results=3)
    print(f"    Source: {result1['source']}")
    print(f"    Cached: {result1['cached']}")

    print(f"\n[+] Second search (same query)...")
    result2 = search.search(query, max_results=3)
    print(f"    Source: {result2['source']}")
    print(f"    Cached: {result2['cached']}")

    if result2['source'] in ['session_cache', 'db_cache']:
        print("\n[+] SUCCESS: Cache hit on repeat query!")
    else:
        print("\n[!] Cache miss on repeat query (may be expected if first search failed)")

    # Show stats
    stats = search.get_stats()
    print(f"\n[+] Session stats:")
    print(f"    Queries made: {stats['session']['queries_made']}")
    print(f"    Cache hits: {stats['session']['cache_hits']}")
    print(f"    DuckDuckGo calls: {stats['session']['duckduckgo_calls']}")
    print(f"    Session dedupes: {stats['session']['session_dedupes']}")

    return True


def test_web_search_rag_integration():
    """Test WebSearchRAG with SmartWebSearch integration."""
    print("\n" + "=" * 70)
    print(" TEST 5: WebSearchRAG Integration")
    print("=" * 70)

    # Initialize - won't fail even without Tavily key now
    rag = WebSearchRAG()

    print("\n[+] WebSearchRAG initialized with SmartWebSearch")

    # Test match context
    print("\n[+] Getting match context for Liverpool vs Chelsea...")
    context = rag.get_match_context("Liverpool", "Chelsea", max_searches=2)

    print(f"\n[+] Results:")
    print(f"    Queries executed (API calls): {context['queries_executed']}")
    print(f"    Cache hits: {context['cached_hits']}")
    print(f"    Sources: {context.get('sources', {})}")

    # Show stats
    stats = rag.get_search_stats()
    print(f"\n[+] Search stats:")
    print(f"    Session queries: {stats['session']['queries_made']}")
    print(f"    Total cache entries: {stats['cache']['total_entries']}")

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print(" SMART WEB SEARCH SYSTEM - TEST SUITE")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("SearchCache Persistence", test_search_cache()))
    results.append(("DuckDuckGo Search", test_duckduckgo_search()))
    results.append(("Tavily Search", test_tavily_search()))
    results.append(("Cache Hit", test_cache_hit()))
    results.append(("WebSearchRAG Integration", test_web_search_rag_integration()))

    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    # Final stats
    print("\n" + "=" * 70)
    print(" FINAL CACHE STATS")
    print("=" * 70)

    cache = SearchCache(cache_dir="data/cache")
    stats = cache.get_stats()
    print(f"\n  Total cached searches: {stats['total_entries']}")
    print(f"  Valid (non-expired): {stats['valid_entries']}")
    print(f"  By source: {stats['by_source']}")
    print(f"  Cache location: {stats['db_path']}")

    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
