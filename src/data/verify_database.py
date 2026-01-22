"""
Database verification module for ASIL football prediction project.

Runs comprehensive data quality checks on the SQLite database to ensure
data integrity before using it for predictions.

Usage:
    python src/data/verify_database.py [db_path]

If no db_path is provided, defaults to data/processed/asil.db
"""

import sqlite3
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"
FALLBACK_DB_PATH = PROJECT_ROOT / "data" / "matches.db"


class DatabaseVerifier:
    """
    Runs data quality checks on the ASIL database.

    Checks performed:
    - Record counts (matches, teams)
    - NULL value detection in critical columns
    - Probability normalization (should sum to ~1.0)
    - Results distribution
    - Team match counts
    - Data anomaly detection
    """

    def __init__(self, db_path: Path):
        """Initialize verifier with database path."""
        self.db_path = db_path
        self.conn = None
        self.issues = []

    def connect(self) -> bool:
        """Establish database connection."""
        if not self.db_path.exists():
            print(f"ERROR: Database not found at {self.db_path}")
            return False

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return True

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def print_header(self, title: str):
        """Print formatted section header."""
        print()
        print("=" * 70)
        print(f" {title}")
        print("=" * 70)

    def check_record_counts(self):
        """Check 1: Count total matches and teams."""
        self.print_header("1. RECORD COUNTS")
        cursor = self.conn.cursor()

        # Total matches
        cursor.execute("SELECT COUNT(*) FROM matches")
        total_matches = cursor.fetchone()[0]
        print(f"Total matches: {total_matches}")

        # Matches per season
        cursor.execute("""
            SELECT season, COUNT(*) as count
            FROM matches
            GROUP BY season
            ORDER BY season
        """)
        print("\nMatches by season:")
        for row in cursor.fetchall():
            print(f"  {row['season']}: {row['count']} matches")

        # Total teams
        cursor.execute("SELECT COUNT(*) FROM teams")
        total_teams = cursor.fetchone()[0]
        print(f"\nTotal teams: {total_teams}")

        # Verify expected counts
        if total_matches == 0:
            self.issues.append("CRITICAL: No matches found in database")
        elif total_matches < 1000:
            self.issues.append(f"WARNING: Only {total_matches} matches (expected ~1140 for 3 seasons)")

        if total_teams == 0:
            self.issues.append("CRITICAL: No teams found in database")

    def check_null_values(self):
        """Check 2: Find NULL values in critical columns."""
        self.print_header("2. NULL VALUE CHECK")
        cursor = self.conn.cursor()

        critical_columns = [
            ('date', 'matches'),
            ('season', 'matches'),
            ('home_team', 'matches'),
            ('away_team', 'matches'),
            ('home_goals', 'matches'),
            ('away_goals', 'matches'),
            ('result', 'matches'),
        ]

        optional_columns = [
            ('home_shots', 'matches'),
            ('away_shots', 'matches'),
            ('home_corners', 'matches'),
            ('away_corners', 'matches'),
            ('b365_home_prob', 'matches'),
            ('avg_home_prob', 'matches'),
        ]

        print("Critical columns (must not have NULLs):")
        all_critical_ok = True
        for col, table in critical_columns:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL")
            null_count = cursor.fetchone()[0]
            status = "✓ OK" if null_count == 0 else f"✗ {null_count} NULLs"
            print(f"  {table}.{col}: {status}")
            if null_count > 0:
                all_critical_ok = False
                self.issues.append(f"CRITICAL: {null_count} NULL values in {table}.{col}")

        print("\nOptional columns (NULLs allowed but tracked):")
        for col, table in optional_columns:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL")
            null_count = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total = cursor.fetchone()[0]
            pct = (null_count / total * 100) if total > 0 else 0
            print(f"  {table}.{col}: {null_count} NULLs ({pct:.1f}%)")
            if pct > 10:
                self.issues.append(f"WARNING: {pct:.1f}% NULL values in {table}.{col}")

    def check_probability_sums(self):
        """Check 3: Verify probabilities sum to approximately 1.0."""
        self.print_header("3. PROBABILITY NORMALIZATION CHECK")
        cursor = self.conn.cursor()

        tolerance = 0.01

        # Check Bet365 probabilities
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN b365_home_prob IS NOT NULL
                    AND ABS((b365_home_prob + b365_draw_prob + b365_away_prob) - 1.0) > ?
                    THEN 1 ELSE 0 END) as bad_b365,
                SUM(CASE WHEN avg_home_prob IS NOT NULL
                    AND ABS((avg_home_prob + avg_draw_prob + avg_away_prob) - 1.0) > ?
                    THEN 1 ELSE 0 END) as bad_avg
            FROM matches
        """, (tolerance, tolerance))

        row = cursor.fetchone()
        total = row['total']
        bad_b365 = row['bad_b365']
        bad_avg = row['bad_avg']

        print(f"Tolerance: ±{tolerance}")
        print(f"\nBet365 probabilities:")
        print(f"  Properly normalized: {total - bad_b365}/{total} ({(total - bad_b365)/total*100:.1f}%)")
        if bad_b365 > 0:
            print(f"  ✗ {bad_b365} matches with probabilities not summing to 1.0")
            self.issues.append(f"WARNING: {bad_b365} matches with B365 probs not normalized")

        print(f"\nAverage market probabilities:")
        print(f"  Properly normalized: {total - bad_avg}/{total} ({(total - bad_avg)/total*100:.1f}%)")
        if bad_avg > 0:
            print(f"  ✗ {bad_avg} matches with probabilities not summing to 1.0")
            self.issues.append(f"WARNING: {bad_avg} matches with avg probs not normalized")

        # Show sample of probability sums
        cursor.execute("""
            SELECT
                MIN(b365_home_prob + b365_draw_prob + b365_away_prob) as min_sum,
                MAX(b365_home_prob + b365_draw_prob + b365_away_prob) as max_sum,
                AVG(b365_home_prob + b365_draw_prob + b365_away_prob) as avg_sum
            FROM matches
            WHERE b365_home_prob IS NOT NULL
        """)
        row = cursor.fetchone()
        print(f"\nB365 probability sum statistics:")
        print(f"  Min: {row['min_sum']:.4f}, Max: {row['max_sum']:.4f}, Avg: {row['avg_sum']:.4f}")

    def check_results_distribution(self):
        """Check 4: Show distribution of match results."""
        self.print_header("4. RESULTS DISTRIBUTION")
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                result,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM matches), 2) as percentage
            FROM matches
            GROUP BY result
            ORDER BY result
        """)

        result_labels = {'H': 'Home wins', 'D': 'Draws', 'A': 'Away wins'}

        print(f"{'Result':<12} {'Count':>8} {'Percentage':>12}")
        print("-" * 35)

        for row in cursor.fetchall():
            label = result_labels.get(row['result'], row['result'])
            print(f"{label:<12} {row['count']:>8} {row['percentage']:>11.1f}%")

        # Check for invalid results
        cursor.execute("""
            SELECT COUNT(*) FROM matches WHERE result NOT IN ('H', 'D', 'A')
        """)
        invalid = cursor.fetchone()[0]
        if invalid > 0:
            print(f"\n✗ Found {invalid} matches with invalid result codes!")
            self.issues.append(f"CRITICAL: {invalid} matches with invalid result values")

        # Historical comparison (typical Premier League distribution)
        print("\nComparison with historical PL averages:")
        print("  Typical: Home ~46%, Draw ~25%, Away ~29%")

    def check_team_match_counts(self):
        """Check 5: Show top teams by number of matches."""
        self.print_header("5. TOP TEAMS BY MATCH COUNT")
        cursor = self.conn.cursor()

        cursor.execute("""
            WITH team_matches AS (
                SELECT home_team as team, COUNT(*) as matches FROM matches GROUP BY home_team
                UNION ALL
                SELECT away_team as team, COUNT(*) as matches FROM matches GROUP BY away_team
            )
            SELECT team, SUM(matches) as total_matches
            FROM team_matches
            GROUP BY team
            ORDER BY total_matches DESC
            LIMIT 10
        """)

        print(f"{'Rank':<6} {'Team':<30} {'Matches':>10}")
        print("-" * 50)

        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"{i:<6} {row['team']:<30} {row['total_matches']:>10}")

        # Check for teams with unusual match counts
        cursor.execute("""
            WITH team_matches AS (
                SELECT home_team as team, COUNT(*) as matches FROM matches GROUP BY home_team
                UNION ALL
                SELECT away_team as team, COUNT(*) as matches FROM matches GROUP BY away_team
            )
            SELECT team, SUM(matches) as total_matches
            FROM team_matches
            GROUP BY team
            HAVING SUM(matches) < 38
            ORDER BY total_matches
        """)

        low_match_teams = cursor.fetchall()
        if low_match_teams:
            print(f"\nTeams with fewer than 38 matches (promoted/relegated):")
            for row in low_match_teams:
                print(f"  {row['team']}: {row['total_matches']} matches")

    def check_data_anomalies(self):
        """Check 6: Find potential data issues."""
        self.print_header("6. DATA ANOMALY DETECTION")
        cursor = self.conn.cursor()

        anomalies_found = 0

        # Check for duplicate matches
        print("Checking for duplicate matches...")
        cursor.execute("""
            SELECT date, home_team, away_team, COUNT(*) as count
            FROM matches
            GROUP BY date, home_team, away_team
            HAVING COUNT(*) > 1
        """)
        duplicates = cursor.fetchall()
        if duplicates:
            print(f"  ✗ Found {len(duplicates)} duplicate match entries:")
            for row in duplicates[:5]:
                print(f"    {row['date']}: {row['home_team']} vs {row['away_team']} ({row['count']}x)")
            self.issues.append(f"CRITICAL: {len(duplicates)} duplicate matches found")
            anomalies_found += len(duplicates)
        else:
            print("  ✓ No duplicate matches found")

        # Check for impossible scores
        print("\nChecking for unusual scores...")
        cursor.execute("""
            SELECT date, home_team, away_team, home_goals, away_goals
            FROM matches
            WHERE home_goals > 9 OR away_goals > 9 OR home_goals < 0 OR away_goals < 0
        """)
        unusual_scores = cursor.fetchall()
        if unusual_scores:
            print(f"  ✗ Found {len(unusual_scores)} matches with unusual scores:")
            for row in unusual_scores[:5]:
                print(f"    {row['date']}: {row['home_team']} {row['home_goals']}-{row['away_goals']} {row['away_team']}")
            self.issues.append(f"WARNING: {len(unusual_scores)} matches with unusual scores")
            anomalies_found += len(unusual_scores)
        else:
            print("  ✓ All scores within normal range")

        # Check for result/score mismatch
        print("\nChecking result-score consistency...")
        cursor.execute("""
            SELECT date, home_team, away_team, home_goals, away_goals, result
            FROM matches
            WHERE (result = 'H' AND home_goals <= away_goals)
               OR (result = 'A' AND away_goals <= home_goals)
               OR (result = 'D' AND home_goals != away_goals)
        """)
        mismatches = cursor.fetchall()
        if mismatches:
            print(f"  ✗ Found {len(mismatches)} result-score mismatches:")
            for row in mismatches[:5]:
                print(f"    {row['date']}: {row['home_team']} {row['home_goals']}-{row['away_goals']} {row['away_team']} (recorded as {row['result']})")
            self.issues.append(f"CRITICAL: {len(mismatches)} result-score mismatches")
            anomalies_found += len(mismatches)
        else:
            print("  ✓ All results match scores")

        # Check for probability outliers
        print("\nChecking for probability outliers...")
        cursor.execute("""
            SELECT date, home_team, away_team, avg_home_prob, avg_draw_prob, avg_away_prob
            FROM matches
            WHERE avg_home_prob > 0.95 OR avg_away_prob > 0.95
               OR avg_home_prob < 0.02 OR avg_away_prob < 0.02
        """)
        outliers = cursor.fetchall()
        if outliers:
            print(f"  ! Found {len(outliers)} matches with extreme probabilities:")
            for row in outliers[:5]:
                print(f"    {row['date']}: {row['home_team']} vs {row['away_team']}")
                print(f"      Probs: H={row['avg_home_prob']:.3f}, D={row['avg_draw_prob']:.3f}, A={row['avg_away_prob']:.3f}")
        else:
            print("  ✓ No extreme probability outliers")

        # Check date range
        print("\nChecking date range...")
        cursor.execute("""
            SELECT MIN(date) as earliest, MAX(date) as latest
            FROM matches
        """)
        row = cursor.fetchone()
        print(f"  Date range: {row['earliest']} to {row['latest']}")

        # Check for future dates
        cursor.execute("""
            SELECT COUNT(*) FROM matches WHERE date > date('now')
        """)
        future = cursor.fetchone()[0]
        if future > 0:
            print(f"  ✗ Found {future} matches with future dates")
            self.issues.append(f"WARNING: {future} matches with future dates")

        if anomalies_found == 0:
            print("\n✓ No major data anomalies detected")

    def print_summary(self):
        """Print final summary of all issues found."""
        self.print_header("VERIFICATION SUMMARY")

        if not self.issues:
            print("✓ All checks passed! Database is healthy.")
        else:
            critical = [i for i in self.issues if i.startswith("CRITICAL")]
            warnings = [i for i in self.issues if i.startswith("WARNING")]

            if critical:
                print(f"\n🔴 CRITICAL ISSUES ({len(critical)}):")
                for issue in critical:
                    print(f"   {issue}")

            if warnings:
                print(f"\n🟡 WARNINGS ({len(warnings)}):")
                for issue in warnings:
                    print(f"   {issue}")

            print(f"\nTotal issues found: {len(self.issues)}")

        print()

    def run_all_checks(self):
        """Execute all verification checks."""
        print("=" * 70)
        print(" ASIL DATABASE VERIFICATION REPORT")
        print(f" Database: {self.db_path}")
        print("=" * 70)

        if not self.connect():
            return False

        try:
            self.check_record_counts()
            self.check_null_values()
            self.check_probability_sums()
            self.check_results_distribution()
            self.check_team_match_counts()
            self.check_data_anomalies()
            self.print_summary()
        finally:
            self.close()

        return len([i for i in self.issues if i.startswith("CRITICAL")]) == 0


def main():
    """Run database verification."""
    # Determine database path
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    elif DEFAULT_DB_PATH.exists():
        db_path = DEFAULT_DB_PATH
    elif FALLBACK_DB_PATH.exists():
        db_path = FALLBACK_DB_PATH
        print(f"Note: Using fallback database at {db_path}")
    else:
        print(f"ERROR: No database found at:")
        print(f"  - {DEFAULT_DB_PATH}")
        print(f"  - {FALLBACK_DB_PATH}")
        print("\nRun load_data.py first to create the database.")
        sys.exit(1)

    verifier = DatabaseVerifier(db_path)
    success = verifier.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
