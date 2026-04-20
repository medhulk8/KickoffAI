"""
Fetch upcoming Premier League fixtures from the FPL API.

Returns the next unfinished gameweek as a list of
{home_team, away_team, date} dicts with DB-canonical team names.

Usage:
    from src.data.fetch_fixtures import get_upcoming_fixtures
    fixtures = get_upcoming_fixtures()

    # or as a script:
    python src/data/fetch_fixtures.py
"""

from __future__ import annotations

from datetime import timezone
import requests
from datetime import datetime

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS  = {"User-Agent": "Mozilla/5.0"}

# FPL short name → DB canonical name
TEAM_MAP = {
    "Man City":       "Manchester City",
    "Man Utd":        "Manchester United",
    "Spurs":          "Tottenham Hotspur",
    "Nott'm Forest":  "Nottingham Forest",
    "Newcastle":      "Newcastle United",
    "West Ham":       "West Ham United",
    "Wolves":         "Wolverhampton",
    "Brighton":       "Brighton & Hove Albion",
    "Leeds":          "Leeds United",
    "Leicester":      "Leicester City",
    "Sheffield Utd":  "Sheffield United",
    "Ipswich":        "Ipswich",
    "Luton":          "Luton",
}


def _norm(name: str) -> str:
    return TEAM_MAP.get(name.strip(), name.strip())


def get_upcoming_fixtures() -> list[dict]:
    """
    Returns fixtures for the next unstarted GW as:
      [{"home_team": ..., "away_team": ..., "date": "YYYY-MM-DD", "gw": N}, ...]

    Raises requests.HTTPError on network failure.
    """
    bootstrap = requests.get(f"{BASE_URL}/bootstrap-static/", headers=HEADERS, timeout=20).json()

    # Build team id → name map
    id_to_name = {t["id"]: _norm(t["short_name"]) for t in bootstrap["teams"]}

    # Find next unfinished GW
    next_gw = None
    for event in bootstrap["events"]:
        if not event["finished"]:
            next_gw = event["id"]
            break

    if next_gw is None:
        return []

    fixtures_raw = requests.get(
        f"{BASE_URL}/fixtures/?event={next_gw}", headers=HEADERS, timeout=20
    ).json()

    results = []
    for f in fixtures_raw:
        home_team = id_to_name.get(f["team_h"], str(f["team_h"]))
        away_team = id_to_name.get(f["team_a"], str(f["team_a"]))

        kickoff = f.get("kickoff_time")
        if kickoff:
            date = datetime.fromisoformat(kickoff.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        else:
            date = None

        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "date":      date,
            "gw":        next_gw,
        })

    return sorted(results, key=lambda x: (x["date"] or "", x["home_team"]))


if __name__ == "__main__":
    fixtures = get_upcoming_fixtures()
    if not fixtures:
        print("No upcoming fixtures found.")
    else:
        gw = fixtures[0]["gw"]
        print(f"GW {gw} — {len(fixtures)} fixtures:\n")
        for f in fixtures:
            print(f"  {f['date']}  {f['home_team']} vs {f['away_team']}")
