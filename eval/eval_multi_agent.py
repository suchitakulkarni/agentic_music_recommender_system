"""
eval/eval_multi_agent.py

Evaluation harness for AutonomousOrchestrator and its specialist agents.

Four things are measured:

1. CONFIDENCE PARSING   - _parse_response_with_confidence correctly assigns
                          HIGH / MEDIUM / LOW from keyword presence; responses
                          with no markers default to MEDIUM (no LLM)

2. CONFIDENCE WEIGHTING - _confidence_based_weighting normalises weights to
                          sum to 1.0 and orders HIGH > MEDIUM > LOW (no LLM)

3. AGENT ASSEMBLY       - _activate_agents instantiates the correct agent
                          types for each combination of specialist names,
                          and clears previously active agents (no LLM)

4. ANALYSIS PIPELINE    - analyze_song returns a dict with all required keys,
                          each specialist produces non-empty content, and
                          the synthesis is a non-empty string; requires a
                          live LLM backend (skippable)

Results saved to results/eval_multi_agent.csv + printed summary.
"""
import os
import sys
import csv
from datetime import datetime
from typing import List, Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import config
from src.data_loading import load_and_merge_data
from src.era_analysis import define_eras
from src.agents.multi_agent_system import (
    AutonomousOrchestrator,
    LyricalAnalystAgent,
    MusicalAnalystAgent,
    ContextualAnalystAgent,
    AgentResponse,
    Confidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator() -> AutonomousOrchestrator:
    return AutonomousOrchestrator()


def _make_agent_response(content: str) -> AgentResponse:
    """Build an AgentResponse by running content through the parser."""
    agent = LyricalAnalystAgent()
    return agent._parse_response_with_confidence(content)


def _pick_song_data(df: pd.DataFrame) -> Dict:
    """Return a song row as dict, guaranteed to have required fields."""
    row = df.dropna(subset=['Song_Name', 'Album']).iloc[0]
    return row.to_dict()


# ---------------------------------------------------------------------------
# Eval 1: Confidence parsing
# ---------------------------------------------------------------------------

def eval_confidence_parsing() -> List[Dict]:
    """
    _parse_response_with_confidence should:
    - Assign HIGH when the text contains 'clearly', 'definitely', 'certainly',
      or 'strongly'
    - Assign LOW when the text contains 'unclear', 'uncertain', 'possibly',
      or 'might'
    - Default to MEDIUM when no markers are present
    - Extract key_points from lines starting with -, •, or 1. 2. 3.
    - Extract uncertainties from sentences containing uncertainty markers
    """
    print("\n" + "="*60)
    print("EVAL 1: CONFIDENCE PARSING")
    print("="*60)

    agent = LyricalAnalystAgent()

    test_cases = [
        (
            "This song clearly demonstrates high energy production. "
            "The tempo is definitely above average for the era.",
            Confidence.HIGH,
            "HIGH markers: clearly, definitely",
        ),
        (
            "It is unclear whether the song fits this era. "
            "The theme might be intentional ambiguity.",
            Confidence.LOW,
            "LOW markers: unclear, might",
        ),
        (
            "The song has moderate energy and a mid-range tempo. "
            "It represents the typical sound of the album.",
            Confidence.MEDIUM,
            "No markers → default MEDIUM",
        ),
        (
            "Certainly one of the strongest tracks. "
            "- High valence\n- Strong narrative\n1. Emotional resonance",
            Confidence.HIGH,
            "HIGH + key_points extraction",
        ),
        (
            "The theme is somewhat uncertain in this track. "
            "It is not sure how the bridge connects to the chorus.",
            Confidence.LOW,
            "LOW markers: uncertain, not sure",
        ),
    ]

    results = []
    for text, expected_confidence, description in test_cases:
        response = agent._parse_response_with_confidence(text)
        conf_correct = response.confidence == expected_confidence
        status = "PASS" if conf_correct else "FAIL"
        print(f"  [{status}] {description}")
        print(f"         Expected: {expected_confidence.value} | Got: {response.confidence.value}")

        results.append({
            "description": description,
            "expected_confidence": expected_confidence.value,
            "predicted_confidence": response.confidence.value,
            "n_key_points": len(response.key_points),
            "n_uncertainties": len(response.uncertainties),
            "pass": conf_correct,
        })

    # Extra check: key_points are extracted from bullet/numbered lines
    bullet_text = "Analysis:\n- Point one\n- Point two\n• Point three\n1. Point four"
    response = agent._parse_response_with_confidence(bullet_text)
    key_points_extracted = len(response.key_points) >= 3
    status = "PASS" if key_points_extracted else "FAIL"
    print(f"  [{status}] Key point extraction from bullets (got {len(response.key_points)} points)")
    results.append({
        "description": "key_point extraction",
        "expected_confidence": "MEDIUM",
        "predicted_confidence": response.confidence.value,
        "n_key_points": len(response.key_points),
        "n_uncertainties": 0,
        "pass": key_points_extracted,
    })

    return results


# ---------------------------------------------------------------------------
# Eval 2: Confidence weighting
# ---------------------------------------------------------------------------

def eval_confidence_weighting() -> List[Dict]:
    """
    _confidence_based_weighting should:
    - Return weights that sum to 1.0 (normalised)
    - Assign the highest weight to HIGH-confidence agents
    - Assign the lowest weight to LOW-confidence agents
    """
    print("\n" + "="*60)
    print("EVAL 2: CONFIDENCE WEIGHTING")
    print("="*60)

    orchestrator = _make_orchestrator()

    def _fake_response(conf: Confidence) -> AgentResponse:
        return AgentResponse(
            content="placeholder",
            confidence=conf,
            key_points=[],
            uncertainties=[],
        )

    results = []

    # Case 1: all three confidence levels present
    analyses = {
        "lyrical":    _fake_response(Confidence.HIGH),
        "musical":    _fake_response(Confidence.MEDIUM),
        "contextual": _fake_response(Confidence.LOW),
    }
    weights = orchestrator._confidence_based_weighting(analyses)

    total = sum(weights.values())
    sums_to_one = abs(total - 1.0) < 1e-6
    ordered = weights["lyrical"] > weights["musical"] > weights["contextual"]
    passed = sums_to_one and ordered

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Three-agent mix (HIGH > MEDIUM > LOW)")
    print(f"         Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
    print(f"         Sums to 1.0: {sums_to_one} | Ordered correctly: {ordered}")
    results.append({
        "case": "three agents HIGH/MEDIUM/LOW",
        "weight_sum": round(total, 6),
        "sums_to_one": sums_to_one,
        "ordered_correctly": ordered,
        "pass": passed,
    })

    # Case 2: all HIGH — weights should be equal
    analyses_equal = {
        "lyrical":    _fake_response(Confidence.HIGH),
        "musical":    _fake_response(Confidence.HIGH),
        "contextual": _fake_response(Confidence.HIGH),
    }
    weights_equal = orchestrator._confidence_based_weighting(analyses_equal)
    total_equal = sum(weights_equal.values())
    sums_to_one_eq = abs(total_equal - 1.0) < 1e-6
    all_equal = all(abs(v - 1/3) < 1e-6 for v in weights_equal.values())
    passed_eq = sums_to_one_eq and all_equal

    status = "PASS" if passed_eq else "FAIL"
    print(f"  [{status}] All HIGH — weights should be equal (1/3 each)")
    print(f"         Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights_equal.items())}")
    results.append({
        "case": "all agents HIGH",
        "weight_sum": round(total_equal, 6),
        "sums_to_one": sums_to_one_eq,
        "ordered_correctly": all_equal,
        "pass": passed_eq,
    })

    # Case 3: single agent — weight should be 1.0
    analyses_single = {"lyrical": _fake_response(Confidence.MEDIUM)}
    weights_single = orchestrator._confidence_based_weighting(analyses_single)
    single_is_one = abs(weights_single.get("lyrical", 0) - 1.0) < 1e-6
    status = "PASS" if single_is_one else "FAIL"
    print(f"  [{status}] Single agent — weight should be 1.0")
    results.append({
        "case": "single agent",
        "weight_sum": round(sum(weights_single.values()), 6),
        "sums_to_one": single_is_one,
        "ordered_correctly": True,
        "pass": single_is_one,
    })

    return results


# ---------------------------------------------------------------------------
# Eval 3: Agent assembly
# ---------------------------------------------------------------------------

def eval_agent_assembly() -> List[Dict]:
    """
    _activate_agents should:
    - Instantiate the correct class for each named specialist
    - Replace (not append to) the active_agents dict each call
    - Handle an empty list gracefully (no active agents)
    """
    print("\n" + "="*60)
    print("EVAL 3: AGENT ASSEMBLY")
    print("="*60)

    orchestrator = _make_orchestrator()
    results = []

    type_map = {
        "lyrical":    LyricalAnalystAgent,
        "musical":    MusicalAnalystAgent,
        "contextual": ContextualAnalystAgent,
    }

    # Case: activate all three
    orchestrator._activate_agents(["lyrical", "musical", "contextual"])
    all_present   = set(orchestrator.active_agents.keys()) == {"lyrical", "musical", "contextual"}
    types_correct = all(
        isinstance(orchestrator.active_agents[k], type_map[k])
        for k in orchestrator.active_agents
    )
    passed = all_present and types_correct
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Activate all three specialists")
    print(f"         Active: {list(orchestrator.active_agents.keys())} | Types correct: {types_correct}")
    results.append({
        "case": "all three agents",
        "expected": "lyrical, musical, contextual",
        "got": str(sorted(orchestrator.active_agents.keys())),
        "types_correct": types_correct,
        "pass": passed,
    })

    # Case: partial activation clears previous set
    orchestrator._activate_agents(["lyrical"])
    only_lyrical  = list(orchestrator.active_agents.keys()) == ["lyrical"]
    old_cleared   = "musical" not in orchestrator.active_agents
    passed_partial = only_lyrical and old_cleared
    status = "PASS" if passed_partial else "FAIL"
    print(f"  [{status}] Partial activation replaces prior set")
    print(f"         Active: {list(orchestrator.active_agents.keys())}")
    results.append({
        "case": "partial activation replaces prior",
        "expected": "lyrical only",
        "got": str(list(orchestrator.active_agents.keys())),
        "types_correct": isinstance(orchestrator.active_agents.get("lyrical"), LyricalAnalystAgent),
        "pass": passed_partial,
    })

    # Case: empty list → no active agents
    orchestrator._activate_agents([])
    empty_ok = len(orchestrator.active_agents) == 0
    status = "PASS" if empty_ok else "FAIL"
    print(f"  [{status}] Empty list → no active agents")
    results.append({
        "case": "empty activation",
        "expected": "empty",
        "got": str(list(orchestrator.active_agents.keys())),
        "types_correct": True,
        "pass": empty_ok,
    })

    # Case: unknown agent name ignored, valid ones still created
    orchestrator._activate_agents(["lyrical", "nonexistent_agent"])
    only_valid = list(orchestrator.active_agents.keys()) == ["lyrical"]
    status = "PASS" if only_valid else "FAIL"
    print(f"  [{status}] Unknown agent name silently ignored")
    results.append({
        "case": "unknown agent name ignored",
        "expected": "lyrical only",
        "got": str(list(orchestrator.active_agents.keys())),
        "types_correct": True,
        "pass": only_valid,
    })

    return results


# ---------------------------------------------------------------------------
# Eval 4: Full analysis pipeline (LLM-dependent)
# ---------------------------------------------------------------------------

def eval_analysis_pipeline(orchestrator: AutonomousOrchestrator,
                            df: pd.DataFrame) -> List[Dict]:
    """
    analyze_song should return a dict containing all required keys.
    Each specialist's initial analysis must have non-empty content.
    The synthesis must be a non-empty string.
    Does not verify factual accuracy — just structural completeness.
    """
    print("\n" + "="*60)
    print("EVAL 4: ANALYSIS PIPELINE (requires LLM)")
    print("="*60)

    required_keys = {"initial_analyses", "debate_results", "final_analyses",
                     "confidence_weights", "synthesis"}

    # Pick two songs: one with rich data fields, one from a different era
    test_songs = []
    for era in sorted(df['era'].dropna().unique())[:2]:
        row = df[df['era'] == era].dropna(subset=['energy', 'valence']).iloc[0]
        test_songs.append(row.to_dict())

    results = []
    for song_data in test_songs:
        song_name = song_data.get('Song_Name', 'Unknown')
        question  = f"Provide a comprehensive analysis of {song_name}"
        print(f"\n  Analyzing: {song_name}")

        try:
            result = orchestrator.analyze_song(question, song_data)

            has_keys      = required_keys.issubset(result.keys())
            analyses_ok   = all(
                isinstance(v.content, str) and len(v.content.strip()) > 10
                for v in result["initial_analyses"].values()
            )
            synthesis_ok  = isinstance(result["synthesis"], str) and len(result["synthesis"].strip()) > 20
            weights_sum   = abs(sum(result["confidence_weights"].values()) - 1.0) < 1e-5

            passed = has_keys and analyses_ok and synthesis_ok and weights_sum
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {song_name}")
            print(f"         Keys present: {has_keys} | Analyses non-empty: {analyses_ok} "
                  f"| Synthesis non-empty: {synthesis_ok} | Weights sum to 1: {weights_sum}")
            print(f"         Synthesis snippet: {result['synthesis'][:100].replace(chr(10), ' ')}...")

            results.append({
                "song": song_name,
                "has_required_keys": has_keys,
                "analyses_non_empty": analyses_ok,
                "synthesis_non_empty": synthesis_ok,
                "weights_sum_to_one": weights_sum,
                "n_agents_activated": len(result["initial_analyses"]),
                "pass": passed,
            })

        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            results.append({
                "song": song_name,
                "has_required_keys": False,
                "analyses_non_empty": False,
                "synthesis_non_empty": False,
                "weights_sum_to_one": False,
                "n_agents_activated": 0,
                "pass": False,
            })

    return results


# ---------------------------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------------------------

def save_results(confidence: List[Dict], weighting: List[Dict],
                 assembly: List[Dict], pipeline: List[Dict],
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

    write_csv(confidence, "eval_multi_confidence_parsing")
    write_csv(weighting,  "eval_multi_confidence_weighting")
    write_csv(assembly,   "eval_multi_agent_assembly")
    write_csv(pipeline,   "eval_multi_pipeline")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    def pct(data):
        if not data:
            return "N/A"
        n = sum(1 for r in data if r.get("pass"))
        return f"{n*100//len(data)}% ({n}/{len(data)})"

    print(f"  Confidence parsing:   {pct(confidence)}")
    print(f"  Confidence weighting: {pct(weighting)}")
    print(f"  Agent assembly:       {pct(assembly)}")
    if pipeline:
        print(f"  Analysis pipeline:    {pct(pipeline)}")
    else:
        print(f"  Analysis pipeline:    skipped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(run_pipeline: bool = True):
    """
    Run full evaluation suite.

    Args:
        run_pipeline: Set False to skip the LLM-dependent pipeline eval.
    """
    print("="*60)
    print("MULTI-AGENT SYSTEM EVALUATION SUITE")
    print("="*60)

    confidence_results = eval_confidence_parsing()
    weighting_results  = eval_confidence_weighting()
    assembly_results   = eval_agent_assembly()

    pipeline_results = []
    if run_pipeline:
        print("\nLoading data...")
        merged_df = load_and_merge_data(config.DATA_DIR, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
        df = define_eras(merged_df)
        orchestrator = _make_orchestrator()
        pipeline_results = eval_analysis_pipeline(orchestrator, df)
    else:
        print("\n[SKIPPED] Pipeline eval (run_pipeline=False)")

    save_results(confidence_results, weighting_results, assembly_results, pipeline_results,
                 output_dir=config.RESULTS_DIR)


if __name__ == "__main__":
    import argparse
    from src.data_loading import load_and_merge_data
    from src.era_analysis import define_eras

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM-dependent pipeline eval"
    )
    args = parser.parse_args()
    run_eval(run_pipeline=not args.no_llm)
