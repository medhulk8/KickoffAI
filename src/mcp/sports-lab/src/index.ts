/**
 * ASIL Sports Lab MCP Server
 *
 * Model Context Protocol server providing tools for football match
 * prediction and evaluation. Connects to the ASIL SQLite database
 * to retrieve match data, log predictions, and evaluate results.
 *
 * Tools:
 * - get_match: Retrieve details for a specific match
 * - get_recent_matches: Get recent matches for a team
 * - get_baseline_probs: Get baseline probabilities for a match
 * - log_prediction: Record a prediction in the database
 * - evaluate_prediction: Compute accuracy metrics for a prediction
 * - get_team_form: Analyze a team's recent performance
 * - get_league_stats: Calculate league-wide statistics
 * - aggregate_evaluations: Aggregate prediction performance metrics
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import Database from "better-sqlite3";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { z } from "zod";

// ============================================================================
// Types
// ============================================================================

/** Match record from the database */
interface Match {
  match_id: number;
  date: string;
  season: string;
  home_team: string;
  away_team: string;
  home_goals: number;
  away_goals: number;
  result: "H" | "D" | "A";
  home_shots: number | null;
  away_shots: number | null;
  home_corners: number | null;
  away_corners: number | null;
  b365_home_prob: number | null;
  b365_draw_prob: number | null;
  b365_away_prob: number | null;
  avg_home_prob: number | null;
  avg_draw_prob: number | null;
  avg_away_prob: number | null;
}

/** Prediction record from the database */
interface Prediction {
  prediction_id: number;
  match_id: number;
  timestamp: string;
  baseline_home_prob: number;
  baseline_draw_prob: number;
  baseline_away_prob: number;
  llm_home_prob: number;
  llm_draw_prob: number;
  llm_away_prob: number;
  rationale_text: string | null;
}

/** Baseline probabilities with source indicator */
interface BaselineProbs {
  home_prob: number | null;
  draw_prob: number | null;
  away_prob: number | null;
  source: "avg_market" | "b365" | "none";
}

/** Evaluation result with Brier scores */
interface EvaluationResult {
  outcome: "H" | "D" | "A";
  baseline_brier_score: number;
  llm_brier_score: number;
  baseline_correct: boolean;
  llm_correct: boolean;
  baseline_predicted: "H" | "D" | "A";
  llm_predicted: "H" | "D" | "A";
}

/** Team form analysis result */
interface TeamForm {
  team: string;
  window: number;
  matches_analyzed: number;
  wins: number;
  draws: number;
  losses: number;
  points: number;
  goals_scored: number;
  goals_conceded: number;
  goal_difference: number;
  form_string: string;
  points_per_game: number;
  win_rate: number;
  recent_matches: Array<{
    date: string;
    opponent: string;
    venue: "H" | "A";
    goals_for: number;
    goals_against: number;
    result: "W" | "D" | "L";
  }>;
}

/** League statistics result */
interface LeagueStats {
  total_matches: number;
  seasons_included: string[];
  home_wins: number;
  draws: number;
  away_wins: number;
  home_win_pct: number;
  draw_pct: number;
  away_win_pct: number;
  total_goals: number;
  avg_goals_per_match: number;
  home_goals: number;
  away_goals: number;
  avg_home_goals: number;
  avg_away_goals: number;
  home_advantage: number;
}

/** Aggregated evaluation metrics */
interface AggregatedEvaluations {
  total_predictions: number;
  baseline_correct: number;
  llm_correct: number;
  baseline_accuracy: number;
  llm_accuracy: number;
  avg_baseline_brier: number;
  avg_llm_brier: number;
  brier_improvement: number;
  filters_applied: {
    start_date?: string;
    end_date?: string;
    team?: string;
  };
}

// ============================================================================
// Database Setup
// ============================================================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Database paths (try processed first, fallback to matches.db)
const DB_PATH_PRIMARY = join(__dirname, "..", "..", "..", "..", "data", "processed", "asil.db");
const DB_PATH_FALLBACK = join(__dirname, "..", "..", "..", "..", "data", "matches.db");

/**
 * Get database connection, trying primary path first then fallback.
 */
function getDatabase(): Database.Database {
  try {
    // Try primary path first
    return new Database(DB_PATH_PRIMARY);
  } catch {
    // Fallback to matches.db
    try {
      return new Database(DB_PATH_FALLBACK);
    } catch (error) {
      throw new Error(
        `Database not found at:\n  - ${DB_PATH_PRIMARY}\n  - ${DB_PATH_FALLBACK}\n` +
        `Run load_data.py first to create the database.`
      );
    }
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute Brier score for a probability prediction.
 *
 * Brier score measures the accuracy of probabilistic predictions.
 * Formula: sum((predicted_prob - actual)^2) for each outcome
 *
 * @param homeProb - Predicted probability of home win
 * @param drawProb - Predicted probability of draw
 * @param awayProb - Predicted probability of away win
 * @param actualOutcome - Actual match result ('H', 'D', or 'A')
 * @returns Brier score (0 = perfect, 2 = worst possible)
 */
function computeBrierScore(
  homeProb: number,
  drawProb: number,
  awayProb: number,
  actualOutcome: "H" | "D" | "A"
): number {
  // Actual outcomes as binary indicators
  const actualHome = actualOutcome === "H" ? 1 : 0;
  const actualDraw = actualOutcome === "D" ? 1 : 0;
  const actualAway = actualOutcome === "A" ? 1 : 0;

  // Brier score = sum of squared differences
  const brierScore =
    Math.pow(homeProb - actualHome, 2) +
    Math.pow(drawProb - actualDraw, 2) +
    Math.pow(awayProb - actualAway, 2);

  return Math.round(brierScore * 10000) / 10000; // Round to 4 decimal places
}

/**
 * Determine predicted outcome from probabilities.
 *
 * @param homeProb - Probability of home win
 * @param drawProb - Probability of draw
 * @param awayProb - Probability of away win
 * @returns Predicted outcome ('H', 'D', or 'A')
 */
function getPredictedOutcome(
  homeProb: number,
  drawProb: number,
  awayProb: number
): "H" | "D" | "A" {
  if (homeProb >= drawProb && homeProb >= awayProb) return "H";
  if (awayProb >= homeProb && awayProb >= drawProb) return "A";
  return "D";
}

/**
 * Validate that probabilities sum to approximately 1.0.
 *
 * @param homeProb - Home win probability
 * @param drawProb - Draw probability
 * @param awayProb - Away win probability
 * @param tolerance - Allowed deviation from 1.0 (default 0.01)
 * @returns True if probabilities are valid
 */
function validateProbabilities(
  homeProb: number,
  drawProb: number,
  awayProb: number,
  tolerance: number = 0.01
): boolean {
  const sum = homeProb + drawProb + awayProb;
  return (
    homeProb >= 0 &&
    homeProb <= 1 &&
    drawProb >= 0 &&
    drawProb <= 1 &&
    awayProb >= 0 &&
    awayProb <= 1 &&
    Math.abs(sum - 1.0) <= tolerance
  );
}

// ============================================================================
// MCP Server Setup
// ============================================================================

const server = new McpServer({
  name: "sports-lab",
  version: "1.0.0",
});

// ============================================================================
// Tool 1: get_match
// ============================================================================

/**
 * Get detailed information for a specific match.
 *
 * Retrieves all stored data for a match including teams, scores,
 * statistics, and betting probabilities.
 *
 * @param match_id - Unique identifier for the match
 * @returns Match details as JSON, or error if not found
 */
server.tool(
  "get_match",
  "Get detailed information for a specific match by ID",
  {
    match_id: z.number().int().positive().describe("Unique match identifier"),
  },
  async ({ match_id }): Promise<{ content: Array<{ type: "text"; text: string }> }> => {
    const db = getDatabase();

    try {
      const match = db
        .prepare("SELECT * FROM matches WHERE match_id = ?")
        .get(match_id) as Match | undefined;

      if (!match) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ error: `Match with ID ${match_id} not found` }),
            },
          ],
        };
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(match, null, 2),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

// ============================================================================
// Tool 2: get_recent_matches
// ============================================================================

/**
 * Get recent matches for a specific team.
 *
 * Retrieves the last N matches where the team played either home or away.
 * Results are ordered by date descending (most recent first).
 *
 * @param team_name - Name of the team to search for
 * @param n - Number of matches to return (default 5, max 50)
 * @returns Array of match records
 */
server.tool(
  "get_recent_matches",
  "Get the last N matches for a team (home or away)",
  {
    team_name: z.string().min(1).describe("Team name to search for"),
    n: z.number().int().min(1).max(50).default(5).describe("Number of matches to return"),
  },
  async ({ team_name, n }): Promise<{ content: Array<{ type: "text"; text: string }> }> => {
    const db = getDatabase();

    try {
      const matches = db
        .prepare(
          `SELECT * FROM matches
           WHERE home_team = ? OR away_team = ?
           ORDER BY date DESC
           LIMIT ?`
        )
        .all(team_name, team_name, n) as Match[];

      if (matches.length === 0) {
        // Try case-insensitive search
        const fuzzyMatches = db
          .prepare(
            `SELECT * FROM matches
             WHERE LOWER(home_team) LIKE LOWER(?) OR LOWER(away_team) LIKE LOWER(?)
             ORDER BY date DESC
             LIMIT ?`
          )
          .all(`%${team_name}%`, `%${team_name}%`, n) as Match[];

        if (fuzzyMatches.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  error: `No matches found for team "${team_name}"`,
                  suggestion: "Check team name spelling or try a partial name",
                }),
              },
            ],
          };
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                matches: fuzzyMatches,
                note: `Exact match not found. Showing results for partial match "${team_name}"`,
              }, null, 2),
            },
          ],
        };
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ matches, count: matches.length }, null, 2),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

// ============================================================================
// Tool 3: get_baseline_probs
// ============================================================================

/**
 * Get baseline probabilities for a match.
 *
 * Returns market-derived probabilities that can be used as a baseline
 * for LLM predictions. Uses average market probabilities if available,
 * otherwise falls back to Bet365 odds.
 *
 * @param match_id - Unique identifier for the match
 * @returns Probabilities with source indicator
 */
server.tool(
  "get_baseline_probs",
  "Get baseline probabilities for a match (avg market or Bet365 fallback)",
  {
    match_id: z.number().int().positive().describe("Unique match identifier"),
  },
  async ({ match_id }): Promise<{ content: Array<{ type: "text"; text: string }> }> => {
    const db = getDatabase();

    try {
      const row = db
        .prepare(
          `SELECT avg_home_prob, b365_home_prob,
                  avg_draw_prob, b365_draw_prob,
                  avg_away_prob, b365_away_prob
           FROM matches WHERE match_id = ?`
        )
        .get(match_id) as {
          avg_home_prob: number | null;
          b365_home_prob: number | null;
          avg_draw_prob: number | null;
          b365_draw_prob: number | null;
          avg_away_prob: number | null;
          b365_away_prob: number | null;
        } | undefined;

      if (!row) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ error: `Match with ID ${match_id} not found` }),
            },
          ],
        };
      }

      let result: BaselineProbs;

      // Prefer average market probabilities
      if (row.avg_home_prob !== null) {
        result = {
          home_prob: row.avg_home_prob,
          draw_prob: row.avg_draw_prob,
          away_prob: row.avg_away_prob,
          source: "avg_market",
        };
      }
      // Fallback to Bet365
      else if (row.b365_home_prob !== null) {
        result = {
          home_prob: row.b365_home_prob,
          draw_prob: row.b365_draw_prob,
          away_prob: row.b365_away_prob,
          source: "b365",
        };
      }
      // No probabilities available
      else {
        result = {
          home_prob: null,
          draw_prob: null,
          away_prob: null,
          source: "none",
        };
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

// ============================================================================
// Tool 4: log_prediction
// ============================================================================

/**
 * Log a prediction for a match.
 *
 * Records both baseline and LLM-generated probabilities along with
 * the LLM's rationale for its prediction adjustments.
 *
 * @param match_id - Match being predicted
 * @param baseline_*_prob - Baseline model probabilities
 * @param llm_*_prob - LLM-adjusted probabilities
 * @param rationale_text - LLM's explanation for adjustments
 * @param timestamp - When prediction was made (ISO format)
 * @returns Prediction ID and success status
 */
server.tool(
  "log_prediction",
  "Record a prediction for a match in the database",
  {
    match_id: z.number().int().positive().describe("Match ID to predict"),
    baseline_home_prob: z.number().min(0).max(1).describe("Baseline home win probability"),
    baseline_draw_prob: z.number().min(0).max(1).describe("Baseline draw probability"),
    baseline_away_prob: z.number().min(0).max(1).describe("Baseline away win probability"),
    llm_home_prob: z.number().min(0).max(1).describe("LLM-adjusted home win probability"),
    llm_draw_prob: z.number().min(0).max(1).describe("LLM-adjusted draw probability"),
    llm_away_prob: z.number().min(0).max(1).describe("LLM-adjusted away win probability"),
    rationale_text: z.string().optional().describe("LLM rationale for prediction"),
    timestamp: z.string().optional().describe("ISO timestamp (defaults to now)"),
  },
  async ({
    match_id,
    baseline_home_prob,
    baseline_draw_prob,
    baseline_away_prob,
    llm_home_prob,
    llm_draw_prob,
    llm_away_prob,
    rationale_text,
    timestamp,
  }): Promise<{ content: Array<{ type: "text"; text: string }> }> => {
    const db = getDatabase();

    try {
      // Validate match exists
      const match = db
        .prepare("SELECT match_id FROM matches WHERE match_id = ?")
        .get(match_id);

      if (!match) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ error: `Match with ID ${match_id} not found`, success: false }),
            },
          ],
        };
      }

      // Validate baseline probabilities
      if (!validateProbabilities(baseline_home_prob, baseline_draw_prob, baseline_away_prob)) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                error: "Baseline probabilities must be between 0-1 and sum to ~1.0",
                success: false,
              }),
            },
          ],
        };
      }

      // Validate LLM probabilities
      if (!validateProbabilities(llm_home_prob, llm_draw_prob, llm_away_prob)) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                error: "LLM probabilities must be between 0-1 and sum to ~1.0",
                success: false,
              }),
            },
          ],
        };
      }

      // Use current timestamp if not provided
      const ts = timestamp || new Date().toISOString();

      // Insert prediction
      const result = db
        .prepare(
          `INSERT INTO predictions (
            match_id, timestamp,
            baseline_home_prob, baseline_draw_prob, baseline_away_prob,
            llm_home_prob, llm_draw_prob, llm_away_prob,
            rationale_text
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          match_id,
          ts,
          baseline_home_prob,
          baseline_draw_prob,
          baseline_away_prob,
          llm_home_prob,
          llm_draw_prob,
          llm_away_prob,
          rationale_text || null
        );

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              prediction_id: result.lastInsertRowid,
              success: true,
            }),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

// ============================================================================
// Tool 5: evaluate_prediction
// ============================================================================

/**
 * Evaluate a prediction against actual match outcome.
 *
 * Computes Brier scores for both baseline and LLM predictions,
 * and checks if each correctly predicted the match outcome.
 *
 * Brier score: Lower is better (0 = perfect, 2 = worst)
 * Correct: Whether highest probability outcome matched actual result
 *
 * @param match_id - Match ID to evaluate
 * @returns Evaluation metrics including Brier scores and accuracy
 */
server.tool(
  "evaluate_prediction",
  "Evaluate prediction accuracy against actual match outcome",
  {
    match_id: z.number().int().positive().describe("Match ID to evaluate"),
  },
  async ({ match_id }): Promise<{ content: Array<{ type: "text"; text: string }> }> => {
    const db = getDatabase();

    try {
      // Get match outcome
      const match = db
        .prepare("SELECT result FROM matches WHERE match_id = ?")
        .get(match_id) as { result: "H" | "D" | "A" } | undefined;

      if (!match) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ error: `Match with ID ${match_id} not found` }),
            },
          ],
        };
      }

      // Get most recent prediction for this match
      const prediction = db
        .prepare(
          `SELECT * FROM predictions
           WHERE match_id = ?
           ORDER BY timestamp DESC
           LIMIT 1`
        )
        .get(match_id) as Prediction | undefined;

      if (!prediction) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ error: `No prediction found for match ${match_id}` }),
            },
          ],
        };
      }

      const outcome = match.result;

      // Compute Brier scores
      const baselineBrier = computeBrierScore(
        prediction.baseline_home_prob,
        prediction.baseline_draw_prob,
        prediction.baseline_away_prob,
        outcome
      );

      const llmBrier = computeBrierScore(
        prediction.llm_home_prob,
        prediction.llm_draw_prob,
        prediction.llm_away_prob,
        outcome
      );

      // Determine predicted outcomes
      const baselinePredicted = getPredictedOutcome(
        prediction.baseline_home_prob,
        prediction.baseline_draw_prob,
        prediction.baseline_away_prob
      );

      const llmPredicted = getPredictedOutcome(
        prediction.llm_home_prob,
        prediction.llm_draw_prob,
        prediction.llm_away_prob
      );

      const result: EvaluationResult = {
        outcome,
        baseline_brier_score: baselineBrier,
        llm_brier_score: llmBrier,
        baseline_correct: baselinePredicted === outcome,
        llm_correct: llmPredicted === outcome,
        baseline_predicted: baselinePredicted,
        llm_predicted: llmPredicted,
      };

      // Optionally insert into evaluations table
      try {
        db.prepare(
          `INSERT INTO evaluations (
            match_id, prediction_id, outcome,
            baseline_brier_score, llm_brier_score,
            baseline_correct, llm_correct
          ) VALUES (?, ?, ?, ?, ?, ?, ?)`
        ).run(
          match_id,
          prediction.prediction_id,
          outcome,
          baselineBrier,
          llmBrier,
          result.baseline_correct ? 1 : 0,
          result.llm_correct ? 1 : 0
        );
      } catch {
        // Evaluation may already exist, that's okay
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

// ============================================================================
// Tool 6: get_team_form
// ============================================================================

/**
 * Analyze a team's recent performance over last N matches.
 *
 * Calculates wins, draws, losses, points, goals scored/conceded,
 * and generates a form string (e.g., "WWDLW" for Win-Win-Draw-Loss-Win).
 *
 * Handles both home and away matches correctly:
 * - If team is home_team: won if result='H', drew if result='D', lost if result='A'
 * - If team is away_team: won if result='A', drew if result='D', lost if result='H'
 *
 * @param team_name - Name of the team to analyze
 * @param window - Number of recent matches to analyze (default 5)
 * @returns TeamForm object with comprehensive form statistics
 */
server.tool(
  "get_team_form",
  "Analyze a team's recent performance over last N matches",
  {
    team_name: z.string().min(1).describe("Name of the team to analyze"),
    window: z.number().int().min(1).max(38).default(5).describe("Number of recent matches to analyze"),
  },
  async ({ team_name, window }): Promise<{ content: Array<{ type: "text"; text: string }> }> => {
    const db = getDatabase();

    try {
      const matches = db
        .prepare(
          `SELECT * FROM matches
           WHERE home_team = ? OR away_team = ?
           ORDER BY date DESC
           LIMIT ?`
        )
        .all(team_name, team_name, window) as Match[];

      if (matches.length === 0) {
        // Try fuzzy match
        const fuzzyMatches = db
          .prepare(
            `SELECT DISTINCT home_team FROM matches WHERE LOWER(home_team) LIKE LOWER(?)
             UNION
             SELECT DISTINCT away_team FROM matches WHERE LOWER(away_team) LIKE LOWER(?)`
          )
          .all(`%${team_name}%`, `%${team_name}%`) as Array<{ home_team: string }>;

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                error: `No matches found for team "${team_name}"`,
                suggestions: fuzzyMatches.map((r) => r.home_team).slice(0, 5),
              }),
            },
          ],
        };
      }

      // Analyze each match from team's perspective
      let wins = 0;
      let draws = 0;
      let losses = 0;
      let goalsScored = 0;
      let goalsConceded = 0;
      const formChars: string[] = [];
      const recentMatches: TeamForm["recent_matches"] = [];

      for (const match of matches) {
        const isHome = match.home_team === team_name;
        const venue: "H" | "A" = isHome ? "H" : "A";
        const opponent = isHome ? match.away_team : match.home_team;
        const goalsFor = isHome ? match.home_goals : match.away_goals;
        const goalsAgainst = isHome ? match.away_goals : match.home_goals;

        goalsScored += goalsFor;
        goalsConceded += goalsAgainst;

        let result: "W" | "D" | "L";
        if (match.result === "D") {
          draws++;
          result = "D";
          formChars.push("D");
        } else if ((isHome && match.result === "H") || (!isHome && match.result === "A")) {
          wins++;
          result = "W";
          formChars.push("W");
        } else {
          losses++;
          result = "L";
          formChars.push("L");
        }

        recentMatches.push({
          date: match.date,
          opponent,
          venue,
          goals_for: goalsFor,
          goals_against: goalsAgainst,
          result,
        });
      }

      const points = wins * 3 + draws;
      const matchesAnalyzed = matches.length;

      const form: TeamForm = {
        team: team_name,
        window,
        matches_analyzed: matchesAnalyzed,
        wins,
        draws,
        losses,
        points,
        goals_scored: goalsScored,
        goals_conceded: goalsConceded,
        goal_difference: goalsScored - goalsConceded,
        form_string: formChars.join(""),
        points_per_game: Math.round((points / matchesAnalyzed) * 100) / 100,
        win_rate: Math.round((wins / matchesAnalyzed) * 100) / 100,
        recent_matches: recentMatches,
      };

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(form, null, 2),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

// ============================================================================
// Tool 7: get_league_stats
// ============================================================================

/**
 * Calculate league-wide statistics for a given season.
 *
 * Provides aggregate statistics including:
 * - Match counts and result distribution
 * - Average goals per match
 * - Home advantage calculation
 *
 * @param season - Season like "2122" or "2324", if null uses all seasons
 * @returns LeagueStats object with comprehensive league statistics
 */
server.tool(
  "get_league_stats",
  "Calculate league-wide statistics for a given season",
  {
    season: z.string().optional().describe("Season like '2122', '2223', '2324'. If not provided, uses all seasons"),
  },
  async ({ season }): Promise<{ content: Array<{ type: "text"; text: string }> }> => {
    const db = getDatabase();

    try {
      // Build query based on whether season is specified
      let query = "SELECT * FROM matches";
      const params: string[] = [];

      if (season) {
        query += " WHERE season = ?";
        params.push(season);
      }

      const matches = db.prepare(query).all(...params) as Match[];

      if (matches.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                error: season
                  ? `No matches found for season "${season}"`
                  : "No matches found in database",
                available_seasons: (db
                  .prepare("SELECT DISTINCT season FROM matches ORDER BY season")
                  .all() as Array<{ season: string }>)
                  .map((r) => r.season),
              }),
            },
          ],
        };
      }

      // Calculate statistics
      let homeWins = 0;
      let draws = 0;
      let awayWins = 0;
      let totalHomeGoals = 0;
      let totalAwayGoals = 0;
      const seasonsSet = new Set<string>();

      for (const match of matches) {
        seasonsSet.add(match.season);
        totalHomeGoals += match.home_goals;
        totalAwayGoals += match.away_goals;

        if (match.result === "H") homeWins++;
        else if (match.result === "D") draws++;
        else awayWins++;
      }

      const totalMatches = matches.length;
      const totalGoals = totalHomeGoals + totalAwayGoals;

      const stats: LeagueStats = {
        total_matches: totalMatches,
        seasons_included: Array.from(seasonsSet).sort(),
        home_wins: homeWins,
        draws: draws,
        away_wins: awayWins,
        home_win_pct: Math.round((homeWins / totalMatches) * 1000) / 10,
        draw_pct: Math.round((draws / totalMatches) * 1000) / 10,
        away_win_pct: Math.round((awayWins / totalMatches) * 1000) / 10,
        total_goals: totalGoals,
        avg_goals_per_match: Math.round((totalGoals / totalMatches) * 100) / 100,
        home_goals: totalHomeGoals,
        away_goals: totalAwayGoals,
        avg_home_goals: Math.round((totalHomeGoals / totalMatches) * 100) / 100,
        avg_away_goals: Math.round((totalAwayGoals / totalMatches) * 100) / 100,
        home_advantage: Math.round(((homeWins - awayWins) / totalMatches) * 1000) / 10,
      };

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(stats, null, 2),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

// ============================================================================
// Tool 8: aggregate_evaluations
// ============================================================================

/**
 * Aggregate prediction performance metrics across multiple evaluations.
 *
 * Calculates overall accuracy and Brier scores for both baseline and LLM
 * predictions, with optional filtering by date range or team.
 *
 * @param start_date - Filter evaluations after this date (ISO format)
 * @param end_date - Filter evaluations before this date (ISO format)
 * @param team - Filter for matches involving this team
 * @returns AggregatedEvaluations object with performance metrics
 */
server.tool(
  "aggregate_evaluations",
  "Aggregate prediction performance metrics across multiple evaluations",
  {
    start_date: z.string().optional().describe("Filter evaluations after this date (YYYY-MM-DD)"),
    end_date: z.string().optional().describe("Filter evaluations before this date (YYYY-MM-DD)"),
    team: z.string().optional().describe("Filter for matches involving this team"),
  },
  async ({ start_date, end_date, team }): Promise<{ content: Array<{ type: "text"; text: string }> }> => {
    const db = getDatabase();

    try {
      // Build query with optional filters
      let query = `
        SELECT e.*, m.home_team, m.away_team, m.date
        FROM evaluations e
        JOIN matches m ON e.match_id = m.match_id
        WHERE 1=1
      `;
      const params: string[] = [];

      if (start_date) {
        query += " AND m.date >= ?";
        params.push(start_date);
      }

      if (end_date) {
        query += " AND m.date <= ?";
        params.push(end_date);
      }

      if (team) {
        query += " AND (m.home_team = ? OR m.away_team = ?)";
        params.push(team, team);
      }

      const evaluations = db.prepare(query).all(...params) as Array<{
        evaluation_id: number;
        match_id: number;
        prediction_id: number;
        outcome: string;
        baseline_brier_score: number;
        llm_brier_score: number;
        baseline_correct: number;
        llm_correct: number;
        error_tags: string | null;
        critique_text: string | null;
        home_team: string;
        away_team: string;
        date: string;
      }>;

      if (evaluations.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                error: "No evaluations found matching the specified filters",
                filters_applied: { start_date, end_date, team },
                suggestion: "Try broadening your filters or run more predictions first",
              }),
            },
          ],
        };
      }

      // Aggregate metrics
      let baselineCorrectCount = 0;
      let llmCorrectCount = 0;
      let totalBaselineBrier = 0;
      let totalLlmBrier = 0;

      for (const evaluation of evaluations) {
        if (evaluation.baseline_correct) baselineCorrectCount++;
        if (evaluation.llm_correct) llmCorrectCount++;
        totalBaselineBrier += evaluation.baseline_brier_score;
        totalLlmBrier += evaluation.llm_brier_score;
      }

      const totalPredictions = evaluations.length;
      const avgBaselineBrier = totalBaselineBrier / totalPredictions;
      const avgLlmBrier = totalLlmBrier / totalPredictions;

      const result: AggregatedEvaluations = {
        total_predictions: totalPredictions,
        baseline_correct: baselineCorrectCount,
        llm_correct: llmCorrectCount,
        baseline_accuracy: Math.round((baselineCorrectCount / totalPredictions) * 1000) / 10,
        llm_accuracy: Math.round((llmCorrectCount / totalPredictions) * 1000) / 10,
        avg_baseline_brier: Math.round(avgBaselineBrier * 10000) / 10000,
        avg_llm_brier: Math.round(avgLlmBrier * 10000) / 10000,
        brier_improvement: Math.round((avgBaselineBrier - avgLlmBrier) * 10000) / 10000,
        filters_applied: {
          start_date,
          end_date,
          team,
        },
      };

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

// ============================================================================
// Server Startup
// ============================================================================

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Sports Lab MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
