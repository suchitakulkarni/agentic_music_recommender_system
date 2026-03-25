"""
eval/eval_tool_agent.py

Evaluation harness for AutonomousToolAgent.

Four things are measured:

1. TOOL EXECUTION   - registered tools return correct data for valid inputs
                      and structured error strings for invalid ones (no LLM)

2. ERROR HANDLING   - execute_tool returns an error string (not an exception)
                      for unknown tools and unknown song/era names;
                      _execute_with_retry returns a FAILURE ToolResult after
                      exhausting retries (no LLM needed for retry path because
                      correction is skipped when no LLM is configured)

3. SUCCESS RATE     - success_rate EMA updates correctly after success and
                      failure: success nudges rate toward 1.0, failure decays
                      it; tracked independently per tool (no LLM)

4. PIPELINE OUTPUT  - full ask() returns a non-empty string and does not
                      raise; requires a live LLM backend (skippable)

Results saved to results/eval_tool_agent.csv + printed summary.
"""
import os
import sys
import csv
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import config
from src.data_loading import load_and_merge_data
from src.era_analysis import define_eras
from src.similarity_analysis import create_hybrid_similarity_system
from src.agents.tool_agent import (
    AutonomousToolAgent,
    ToolStatus,
    create_song_info_tool,
    create_era_stats_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(df: pd.DataFrame, sim_results: Dict) -> AutonomousToolAgent:
    """Create a tool agent with both tools registered."""
    agent = AutonomousToolAgent()
    agent.load_analysis_data(sim_results['df'], sim_results)
    agent.register_tool(create_song_info_tool(agent))
    agent.register_tool(create_era_stats_tool(agent))
    return agent


def _pick_song(df: pd.DataFrame) -> str:
    """Return a song name that is guaranteed to exist in the dataframe."""
    return df['Song_Name'].iloc[0]


def _pick_era(df: pd.DataFrame) -> str:
    """Return an era name that is guaranteed to exist in the dataframe."""
    return df['era'].dropna().iloc[0]


# ---------------------------------------------------------------------------
# Eval 1: Tool execution correctness
# ---------------------------------------------------------------------------

def eval_tool_execution(agent: AutonomousToolAgent, df: pd.DataFrame) -> List[Dict]:
    """
    execute_tool should:
    - Return a non-empty string for valid song/era inputs
    - Include the song/era name somewhere in the result
    - Include expected fields (Album, Energy, etc.)
    """
    print("\n" + "="*60)
    print("EVAL 1: TOOL EXECUTION CORRECTNESS")
    print("="*60)

    results = []
    song_name = _pick_song(df)
    era_name = _pick_era(df)

    # --- get_song_info ---
    result = agent.execute_tool("get_song_info", [song_name])
    has_song = song_name.lower() in result.lower()
    has_fields = all(f in result for f in ["Album", "Energy", "Valence"])
    is_error = result.startswith("Error")
    passed = has_song and has_fields and not is_error

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] get_song_info('{song_name}')")
    print(f"         Has song name: {has_song} | Has fields: {has_fields} | Is error: {is_error}")
    results.append({
        "tool": "get_song_info",
        "input": song_name,
        "has_expected_name": has_song,
        "has_expected_fields": has_fields,
        "is_error": is_error,
        "pass": passed,
    })

    # --- get_era_stats ---
    result = agent.execute_tool("get_era_stats", [era_name])
    has_era = era_name.lower() in result.lower()
    has_fields = all(f in result for f in ["Songs", "Avg Energy", "Avg Valence"])
    is_error = result.startswith("Error")
    passed = has_era and has_fields and not is_error

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] get_era_stats('{era_name}')")
    print(f"         Has era name: {has_era} | Has fields: {has_fields} | Is error: {is_error}")
    results.append({
        "tool": "get_era_stats",
        "input": era_name,
        "has_expected_name": has_era,
        "has_expected_fields": has_fields,
        "is_error": is_error,
        "pass": passed,
    })

    return results


# ---------------------------------------------------------------------------
# Eval 2: Error handling
# ---------------------------------------------------------------------------

def eval_error_handling(agent: AutonomousToolAgent) -> List[Dict]:
    """
    execute_tool must return a structured error string (not raise) for:
    - Unregistered tool name
    - Unknown song name
    - Unknown era name

    _execute_with_retry must return a FAILURE ToolResult when every attempt
    returns an error string and correction cannot be applied (LLM absent or
    correction returns None).
    """
    print("\n" + "="*60)
    print("EVAL 2: ERROR HANDLING")
    print("="*60)

    results = []

    cases = [
        ("nonexistent_tool",   ["anything"],             "unknown tool"),
        ("get_song_info",      ["ZZZNOMATCH_XYZ_99999"], "unknown song"),
        ("get_era_stats",      ["ZZZNOMATCH_ERA_99999"], "unknown era"),
    ]

    for tool_name, args, description in cases:
        result = agent.execute_tool(tool_name, args)
        is_str = isinstance(result, str)
        has_error = is_str and "Error" in result
        status = "PASS" if has_error else "FAIL"
        print(f"  [{status}] {description}: execute_tool returns error string")
        print(f"         Result: {result[:80]}")
        results.append({
            "case": description,
            "tool": tool_name,
            "returned_string": is_str,
            "contains_error": has_error,
            "pass": has_error,
        })

    # _execute_with_retry should exhaust retries and return FAILURE
    # Use an unknown song so every attempt returns "Error: Song not found"
    # _attempt_correction requires LLM; with no LLM available it will either
    # return None or error-string, both of which leave the retry loop intact.
    retry_result = agent._execute_with_retry("get_song_info", ["ZZZNOMATCH_RETRY"], max_retries=1)
    is_failure = retry_result.status == ToolStatus.FAILURE
    status = "PASS" if is_failure else "FAIL"
    print(f"  [{status}] _execute_with_retry returns FAILURE after max retries")
    results.append({
        "case": "retry exhaustion",
        "tool": "get_song_info",
        "returned_string": True,
        "contains_error": is_failure,
        "pass": is_failure,
    })

    return results


# ---------------------------------------------------------------------------
# Eval 3: Success rate tracking
# ---------------------------------------------------------------------------

def eval_success_rate_tracking(agent: AutonomousToolAgent, df: pd.DataFrame) -> List[Dict]:
    """
    After a successful execute_tool call the tool's success_rate should be
    >= its prior value (EMA toward 1.0).

    After a failing call (unknown input) the rate should be < prior value
    (EMA decay by 0.9).

    Each tool is tested independently so rates don't cross-contaminate.
    """
    print("\n" + "="*60)
    print("EVAL 3: SUCCESS RATE TRACKING")
    print("="*60)

    results = []
    song_name = _pick_song(df)
    era_name = _pick_era(df)

    for tool_name, good_args, bad_args in [
        ("get_song_info", [song_name], ["ZZZNOMATCH_XYZ"]),
        ("get_era_stats", [era_name],  ["ZZZNOMATCH_ERA"]),
    ]:
        # Reset to known starting value
        agent.tools[tool_name].success_rate = 1.0

        before_success = agent.tools[tool_name].success_rate
        agent.execute_tool(tool_name, good_args)
        after_success = agent.tools[tool_name].success_rate
        rate_stable_or_up = after_success >= before_success - 1e-9

        agent.tools[tool_name].success_rate = 0.8  # set known base for failure test
        before_fail = agent.tools[tool_name].success_rate
        agent.execute_tool(tool_name, bad_args)
        after_fail = agent.tools[tool_name].success_rate
        rate_decreased = after_fail < before_fail

        passed = rate_stable_or_up and rate_decreased
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {tool_name}")
        print(f"         Success: {before_success:.3f} -> {after_success:.3f} (stable/up: {rate_stable_or_up})")
        print(f"         Failure: {before_fail:.3f} -> {after_fail:.3f} (decreased: {rate_decreased})")

        results.append({
            "tool": tool_name,
            "rate_before_success": round(before_success, 4),
            "rate_after_success": round(after_success, 4),
            "rate_before_failure": round(before_fail, 4),
            "rate_after_failure": round(after_fail, 4),
            "success_rate_stable": rate_stable_or_up,
            "failure_rate_decreased": rate_decreased,
            "pass": passed,
        })

    return results


# ---------------------------------------------------------------------------
# Eval 4: Full pipeline (LLM-dependent)
# ---------------------------------------------------------------------------

def eval_pipeline_output(agent: AutonomousToolAgent, df: pd.DataFrame) -> List[Dict]:
    """
    ask() should return a non-empty, non-exception string for simple
    song and era queries. Does not verify factual accuracy — just that
    the pipeline completes and produces output.
    """
    print("\n" + "="*60)
    print("EVAL 4: PIPELINE OUTPUT (requires LLM)")
    print("="*60)

    song_name = _pick_song(df)
    era_name = _pick_era(df)

    test_queries = [
        f"Tell me about the song {song_name}",
        f"What are the stats for the {era_name} era?",
    ]

    results = []
    for query in test_queries:
        print(f"\n  Q: {query}")
        try:
            answer = agent.ask(query)
            non_empty = isinstance(answer, str) and len(answer.strip()) > 20
            status = "PASS" if non_empty else "FAIL"
            print(f"  [{status}] Returned {len(answer)} chars")
            print(f"         Snippet: {answer[:100].replace(chr(10), ' ')}...")
            results.append({
                "query": query[:80],
                "returned_chars": len(answer),
                "non_empty": non_empty,
                "snippet": answer[:100].replace("\n", " "),
                "pass": non_empty,
            })
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            results.append({
                "query": query[:80],
                "returned_chars": 0,
                "non_empty": False,
                "snippet": f"EXCEPTION: {str(e)[:80]}",
                "pass": False,
            })

    return results


# ---------------------------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------------------------

def save_results(execution: List[Dict], errors: List[Dict],
                 rates: List[Dict], pipeline: List[Dict],
                 output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def write_csv(data: List[Dict], name: str):
        if not data:
            return
        path = os.path.join(output_dir, f"{name}_{timestamp}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved: {path}")

    write_csv(execution, "eval_tool_execution")
    write_csv(errors,    "eval_tool_error_handling")
    write_csv(rates,     "eval_tool_success_rate")
    write_csv(pipeline,  "eval_tool_pipeline")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    def pct(data):
        if not data:
            return "N/A"
        n = sum(1 for r in data if r.get("pass"))
        return f"{n*100//len(data)}% ({n}/{len(data)})"

    print(f"  Tool execution correctness: {pct(execution)}")
    print(f"  Error handling:             {pct(errors)}")
    print(f"  Success rate tracking:      {pct(rates)}")
    if pipeline:
        print(f"  Pipeline output:            {pct(pipeline)}")
    else:
        print(f"  Pipeline output:            skipped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(run_pipeline: bool = True):
    """
    Run full evaluation suite.

    Args:
        run_pipeline: Set False to skip LLM-dependent pipeline eval.
    """
    print("="*60)
    print("TOOL AGENT EVALUATION SUITE")
    print("="*60)

    print("\nLoading data...")
    merged_df = load_and_merge_data(config.DATA_DIR, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    df = define_eras(merged_df)

    print("Building similarity system...")
    sim_results = create_hybrid_similarity_system(df)

    print("Initializing agent...")
    agent = _make_agent(sim_results['df'], sim_results)

    execution_results  = eval_tool_execution(agent, sim_results['df'])
    error_results      = eval_error_handling(agent)
    rate_results       = eval_success_rate_tracking(agent, sim_results['df'])

    pipeline_results = []
    if run_pipeline:
        pipeline_results = eval_pipeline_output(agent, sim_results['df'])
    else:
        print("\n[SKIPPED] Pipeline eval (run_pipeline=False)")

    save_results(execution_results, error_results, rate_results, pipeline_results,
                 output_dir=config.RESULTS_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM-dependent pipeline eval"
    )
    args = parser.parse_args()
    run_eval(run_pipeline=not args.no_llm)
