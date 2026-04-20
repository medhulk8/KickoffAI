"""
Injury/lineup extractor for KickoffAI ML model.

Uses Tavily web search to fetch current injury news, then passes the text
to a local Ollama LLM (llama3.1:8b) to extract structured injury features.

Design:
  - 1 targeted search query per match (minimal API usage)
  - LLM extracts: confirmed_out / doubtful players + disruption score (0-1)
  - Disruption score is the key output fed into InjuryAdjuster
  - Fails gracefully: if search or LLM fails, returns zero-disruption (no adjustment)

Disruption scale:
  0.0 = no notable absences
  0.3 = fringe / rotation player out
  0.5 = key regular starter out
  0.8 = multiple key players out
  1.0 = catastrophic (GK + strikers + key midfielder all out)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.web_search_rag import WebSearchRAG

_OLLAMA_MODEL = "llama3.1:8b"

_EXTRACT_PROMPT = """\
You are extracting injury and suspension information for a Premier League match.

MATCH: {home} vs {away}

NEWS SNIPPETS:
{text}

Respond ONLY with a JSON object, no explanation, no markdown:
{{
  "home_confirmed_out": [],
  "home_doubtful": [],
  "away_confirmed_out": [],
  "away_doubtful": [],
  "home_disruption": 0.0,
  "away_disruption": 0.0,
  "summary": ""
}}

Rules:
- home_confirmed_out / away_confirmed_out: player names confirmed absent (injury/suspension)
- home_doubtful / away_doubtful: players listed as doubtful or 50/50
- home_disruption / away_disruption: float 0.0-1.0
  0.0 = no absences found, 0.3 = fringe/rotation player out, 0.5 = key starter out,
  0.8 = multiple key players out, 1.0 = catastrophic (GK + striker both out etc.)
- summary: 1-2 sentence plain-English note on the key injury situation
- If the news does not mention a team, set disruption to 0.0 for that team
- Do not invent players; only use what the text says
"""


def _parse_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, which may have surrounding text."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _empty_result(home: str, away: str, reason: str = "") -> dict:
    return {
        "home_team": home,
        "away_team": away,
        "home_confirmed_out": [],
        "home_doubtful": [],
        "away_confirmed_out": [],
        "away_doubtful": [],
        "home_disruption": 0.0,
        "away_disruption": 0.0,
        "summary": reason or "No injury information available.",
        "raw_text": "",
        "search_source": "none",
    }


class InjuryExtractor:
    """
    Fetches and parses pre-match injury/suspension news for a fixture.

    Usage:
        extractor = InjuryExtractor()
        result = extractor.extract("Liverpool", "Arsenal", "2025-04-20")
        # result["home_disruption"] -> 0.5
        # result["home_confirmed_out"] -> ["Salah"]
    """

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        ollama_model: str = _OLLAMA_MODEL,
    ):
        api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        self._rag = WebSearchRAG(tavily_api_key=api_key)
        self._model = ollama_model

    def extract(self, home_team: str, away_team: str, match_date: Optional[str] = None) -> dict:
        """
        Fetch injury news and extract structured features.

        Returns dict with:
          home_team, away_team,
          home_confirmed_out, home_doubtful,
          away_confirmed_out, away_doubtful,
          home_disruption (0-1), away_disruption (0-1),
          summary, raw_text, search_source
        """
        # Step 1: fetch news
        try:
            search_result = self._rag.get_match_context(
                home_team, away_team, match_date, strategy="minimal"
            )
            raw_text = search_result.get("all_content", "")
            source = list(search_result.get("sources", {}).values())
            search_source = source[0] if source else "unknown"
        except Exception as e:
            return _empty_result(home_team, away_team, f"Search failed: {e}")

        if not raw_text or raw_text.strip() == "No relevant web search results found.":
            return _empty_result(home_team, away_team, "No search results found.")

        # Truncate to avoid overwhelming small LLM
        text = raw_text[:2000]

        # Step 2: LLM extraction
        try:
            import ollama
            prompt = _EXTRACT_PROMPT.format(home=home_team, away=away_team, text=text)
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 512},
            )
            response_text = response["message"]["content"]
        except Exception as e:
            return {
                **_empty_result(home_team, away_team, f"LLM extraction failed: {e}"),
                "raw_text": raw_text,
                "search_source": search_source,
            }

        # Step 3: parse JSON
        parsed = _parse_json_from_response(response_text)
        if not parsed:
            return {
                **_empty_result(home_team, away_team, "Could not parse LLM response."),
                "raw_text": raw_text,
                "search_source": search_source,
            }

        # Clamp disruption scores
        home_dis = max(0.0, min(1.0, float(parsed.get("home_disruption", 0.0))))
        away_dis = max(0.0, min(1.0, float(parsed.get("away_disruption", 0.0))))

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_confirmed_out": parsed.get("home_confirmed_out", []),
            "home_doubtful":      parsed.get("home_doubtful", []),
            "away_confirmed_out": parsed.get("away_confirmed_out", []),
            "away_doubtful":      parsed.get("away_doubtful", []),
            "home_disruption":    round(home_dis, 3),
            "away_disruption":    round(away_dis, 3),
            "summary":            parsed.get("summary", ""),
            "raw_text":           raw_text,
            "search_source":      search_source,
        }


if __name__ == "__main__":
    extractor = InjuryExtractor()
    result = extractor.extract("Manchester City", "Arsenal", "2025-04-20")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_text"}, indent=2))
    print(f"\nSummary: {result['summary']}")
