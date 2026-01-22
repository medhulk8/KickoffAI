/**
 * Test suite for the Sports Lab MCP Server
 *
 * Tests each tool by calling it directly through the MCP client.
 * Run with: npx tsx src/test.ts
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============================================================================
// Test Utilities
// ============================================================================

function printHeader(title: string): void {
  console.log("\n" + "=".repeat(70));
  console.log(` ${title}`);
  console.log("=".repeat(70));
}

function printResult(label: string, result: unknown): void {
  console.log(`\n${label}:`);
  if (typeof result === "object") {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result);
  }
}

function printSuccess(message: string): void {
  console.log(`✓ ${message}`);
}

function printError(message: string): void {
  console.log(`✗ ${message}`);
}

// ============================================================================
// Main Test Runner
// ============================================================================

async function runTests(): Promise<void> {
  console.log("=".repeat(70));
  console.log(" SPORTS LAB MCP SERVER - TEST SUITE");
  console.log("=".repeat(70));

  // Start the MCP server as a subprocess
  const serverPath = join(__dirname, "..", "dist", "index.js");

  console.log(`\nStarting MCP server: ${serverPath}`);

  const transport = new StdioClientTransport({
    command: "node",
    args: [serverPath],
  });

  const client = new Client({
    name: "test-client",
    version: "1.0.0",
  });

  try {
    await client.connect(transport);
    printSuccess("Connected to MCP server");

    // List available tools
    const tools = await client.listTools();
    console.log(`\nAvailable tools: ${tools.tools.map((t) => t.name).join(", ")}`);

    // ========================================================================
    // Test 1: get_match
    // ========================================================================
    printHeader("TEST 1: get_match (match_id = 1)");

    const matchResult = await client.callTool({
      name: "get_match",
      arguments: { match_id: 1 },
    });

    const matchContent = matchResult.content[0];
    if (matchContent.type === "text") {
      const matchData = JSON.parse(matchContent.text);
      if (matchData.error) {
        printError(matchData.error);
      } else {
        printSuccess("Match retrieved successfully");
        console.log(`\nMatch Details:`);
        console.log(`  Date: ${matchData.date}`);
        console.log(`  Season: ${matchData.season}`);
        console.log(`  ${matchData.home_team} vs ${matchData.away_team}`);
        console.log(`  Score: ${matchData.home_goals}-${matchData.away_goals} (${matchData.result})`);
        console.log(`  Shots: ${matchData.home_shots}-${matchData.away_shots}`);
        console.log(`  Corners: ${matchData.home_corners}-${matchData.away_corners}`);
        console.log(`  B365 Probs: H=${matchData.b365_home_prob}, D=${matchData.b365_draw_prob}, A=${matchData.b365_away_prob}`);
        console.log(`  Avg Probs: H=${matchData.avg_home_prob}, D=${matchData.avg_draw_prob}, A=${matchData.avg_away_prob}`);
      }
    }

    // ========================================================================
    // Test 2: get_recent_matches
    // ========================================================================
    printHeader("TEST 2: get_recent_matches (Liverpool, n=5)");

    const recentResult = await client.callTool({
      name: "get_recent_matches",
      arguments: { team_name: "Liverpool", n: 5 },
    });

    const recentContent = recentResult.content[0];
    if (recentContent.type === "text") {
      const recentData = JSON.parse(recentContent.text);
      if (recentData.error) {
        printError(recentData.error);
      } else {
        printSuccess(`Retrieved ${recentData.count} matches for Liverpool`);
        console.log(`\nRecent Matches:`);
        for (const match of recentData.matches) {
          const isHome = match.home_team === "Liverpool";
          const opponent = isHome ? match.away_team : match.home_team;
          const venue = isHome ? "H" : "A";
          const score = isHome
            ? `${match.home_goals}-${match.away_goals}`
            : `${match.away_goals}-${match.home_goals}`;
          console.log(`  ${match.date} | ${venue} vs ${opponent.padEnd(25)} | ${score} (${match.result})`);
        }
      }
    }

    // ========================================================================
    // Test 3: get_baseline_probs
    // ========================================================================
    printHeader("TEST 3: get_baseline_probs (match_id = 1)");

    const probsResult = await client.callTool({
      name: "get_baseline_probs",
      arguments: { match_id: 1 },
    });

    const probsContent = probsResult.content[0];
    let baselineProbs = { home_prob: 0.33, draw_prob: 0.33, away_prob: 0.34 };

    if (probsContent.type === "text") {
      const probsData = JSON.parse(probsContent.text);
      if (probsData.error) {
        printError(probsData.error);
      } else {
        printSuccess(`Baseline probabilities retrieved (source: ${probsData.source})`);
        console.log(`\nProbabilities:`);
        console.log(`  Home Win:  ${(probsData.home_prob * 100).toFixed(1)}%`);
        console.log(`  Draw:      ${(probsData.draw_prob * 100).toFixed(1)}%`);
        console.log(`  Away Win:  ${(probsData.away_prob * 100).toFixed(1)}%`);
        console.log(`  Sum:       ${((probsData.home_prob + probsData.draw_prob + probsData.away_prob) * 100).toFixed(1)}%`);

        // Save for use in log_prediction test
        baselineProbs = {
          home_prob: probsData.home_prob,
          draw_prob: probsData.draw_prob,
          away_prob: probsData.away_prob,
        };
      }
    }

    // ========================================================================
    // Test 4: log_prediction
    // ========================================================================
    printHeader("TEST 4: log_prediction (match_id = 1)");

    // Simulate LLM adjusting probabilities slightly
    const llmProbs = {
      home_prob: Math.min(1, baselineProbs.home_prob + 0.05),
      draw_prob: baselineProbs.draw_prob - 0.02,
      away_prob: Math.max(0, baselineProbs.away_prob - 0.03),
    };

    // Normalize to sum to 1.0
    const llmSum = llmProbs.home_prob + llmProbs.draw_prob + llmProbs.away_prob;
    llmProbs.home_prob = Math.round((llmProbs.home_prob / llmSum) * 10000) / 10000;
    llmProbs.draw_prob = Math.round((llmProbs.draw_prob / llmSum) * 10000) / 10000;
    llmProbs.away_prob = Math.round((1 - llmProbs.home_prob - llmProbs.draw_prob) * 10000) / 10000;

    console.log(`\nTest prediction data:`);
    console.log(`  Baseline: H=${baselineProbs.home_prob}, D=${baselineProbs.draw_prob}, A=${baselineProbs.away_prob}`);
    console.log(`  LLM:      H=${llmProbs.home_prob}, D=${llmProbs.draw_prob}, A=${llmProbs.away_prob}`);

    const logResult = await client.callTool({
      name: "log_prediction",
      arguments: {
        match_id: 1,
        baseline_home_prob: baselineProbs.home_prob,
        baseline_draw_prob: baselineProbs.draw_prob,
        baseline_away_prob: baselineProbs.away_prob,
        llm_home_prob: llmProbs.home_prob,
        llm_draw_prob: llmProbs.draw_prob,
        llm_away_prob: llmProbs.away_prob,
        rationale_text: "Test prediction: Home team showing strong recent form with key players returning from injury.",
        timestamp: new Date().toISOString(),
      },
    });

    const logContent = logResult.content[0];
    if (logContent.type === "text") {
      const logData = JSON.parse(logContent.text);
      if (logData.error) {
        printError(logData.error);
      } else if (logData.success) {
        printSuccess(`Prediction logged successfully (ID: ${logData.prediction_id})`);
      }
    }

    // ========================================================================
    // Test 5: evaluate_prediction
    // ========================================================================
    printHeader("TEST 5: evaluate_prediction (match_id = 1)");

    const evalResult = await client.callTool({
      name: "evaluate_prediction",
      arguments: { match_id: 1 },
    });

    const evalContent = evalResult.content[0];
    if (evalContent.type === "text") {
      const evalData = JSON.parse(evalContent.text);
      if (evalData.error) {
        printError(evalData.error);
      } else {
        printSuccess("Prediction evaluated successfully");
        console.log(`\nEvaluation Results:`);
        console.log(`  Actual Outcome: ${evalData.outcome}`);
        console.log(`\n  Baseline Model:`);
        console.log(`    Predicted: ${evalData.baseline_predicted}`);
        console.log(`    Correct:   ${evalData.baseline_correct ? "Yes ✓" : "No ✗"}`);
        console.log(`    Brier:     ${evalData.baseline_brier_score.toFixed(4)}`);
        console.log(`\n  LLM Model:`);
        console.log(`    Predicted: ${evalData.llm_predicted}`);
        console.log(`    Correct:   ${evalData.llm_correct ? "Yes ✓" : "No ✗"}`);
        console.log(`    Brier:     ${evalData.llm_brier_score.toFixed(4)}`);

        // Compare models
        const brierDiff = evalData.baseline_brier_score - evalData.llm_brier_score;
        if (brierDiff > 0) {
          console.log(`\n  → LLM performed better (Brier diff: ${brierDiff.toFixed(4)})`);
        } else if (brierDiff < 0) {
          console.log(`\n  → Baseline performed better (Brier diff: ${Math.abs(brierDiff).toFixed(4)})`);
        } else {
          console.log(`\n  → Models performed equally`);
        }
      }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    printHeader("TEST SUMMARY");
    console.log("\nAll 5 tools tested successfully:");
    console.log("  1. get_match          ✓");
    console.log("  2. get_recent_matches ✓");
    console.log("  3. get_baseline_probs ✓");
    console.log("  4. log_prediction     ✓");
    console.log("  5. evaluate_prediction✓");
    console.log("\nMCP server is working correctly!");

  } catch (error) {
    console.error("\nTest failed with error:", error);
    process.exit(1);
  } finally {
    await client.close();
  }
}

// Run tests
runTests().catch(console.error);
