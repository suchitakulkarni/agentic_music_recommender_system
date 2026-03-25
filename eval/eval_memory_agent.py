"""
eval/eval_memory_agent.py

Evaluation harness for AutonomousMemoryAgent.

Two things are measured:
1. RETRIEVAL PRECISION - given a query, do retrieved memories rank genuinely
   relevant ones above irrelevant ones? Measured without any LLM calls using
   ground-truth topic labels.

2. DATA GROUNDING - does the agent refuse out-of-scope questions and cite
   real numbers for in-scope ones? Requires a live LLM backend.

Results are saved to results/eval_memory_agent.csv and printed as a summary.
"""
import os
import sys
import csv
import json
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import config
from src.data_loading import load_and_merge_data
from src.era_analysis import define_eras
from src.agents.memory_agent import AutonomousMemoryAgent, Memory, _build_data_context


# ---------------------------------------------------------------------------
# Ground-truth test cases
# ---------------------------------------------------------------------------

# Each entry: question, expected_topic, is_in_scope (True = answerable from data)
FACTUAL_QUESTIONS = [
    {
        "question": "What is the average energy of folklore songs?",
        "expected_topic": "musical_analysis",
        "in_scope": True,
        "ground_truth_col": "energy",
        "ground_truth_filter": {"era": "folklore"},
        "ground_truth_agg": "mean",
    },
    {
        "question": "Which era has the highest average valence?",
        "expected_topic": "musical_analysis",
        "in_scope": True,
        "ground_truth_col": "valence",
        "ground_truth_filter": {},
        "ground_truth_agg": "max_era",
    },
    {
        "question": "How many songs are in the reputation era?",
        "expected_topic": "career_evolution",
        "in_scope": True,
        "ground_truth_col": "era",
        "ground_truth_filter": {"era": "reputation"},
        "ground_truth_agg": "count",
    },
    {
        "question": "What is the average tempo across all songs?",
        "expected_topic": "musical_analysis",
        "in_scope": True,
        "ground_truth_col": "tempo",
        "ground_truth_filter": {},
        "ground_truth_agg": "mean",
    },
    {
        "question": "Which era has the most acoustic songs on average?",
        "expected_topic": "musical_analysis",
        "in_scope": True,
        "ground_truth_col": "acousticness",
        "ground_truth_filter": {},
        "ground_truth_agg": "max_era",
    },
]

# Out-of-scope questions - agent should refuse these
OUT_OF_SCOPE_QUESTIONS = [
    "How many Grammy awards has Taylor Swift won?",
    "What is Taylor Swift's net worth?",
    "When did Taylor Swift go on the Eras Tour?",
    "What is Taylor Swift's relationship history?",
]

# Retrieval precision test pairs:
# (query, list of memory topics that are relevant, list that are not)
RETRIEVAL_TEST_PAIRS = [
    (
        "How does energy change across eras?",
        ["musical_analysis", "career_evolution"],
        ["recommendations", "lyrical_analysis"]
    ),
    (
        "Compare folklore and 1989 albums",
        ["comparisons", "career_evolution"],
        ["recommendations"]
    ),
    (
        "Suggest songs similar to All Too Well",
        ["recommendations", "lyrical_analysis"],
        ["career_evolution"]
    ),
    (
        "Explain the lyrical themes in the reputation era",
        ["lyrical_analysis", "career_evolution"],
        ["recommendations"]
    ),
]


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

def compute_ground_truth(df: pd.DataFrame, case: Dict) -> float:
    """Compute the expected numerical answer for a factual question."""
    col = case["ground_truth_col"]
    filters = case["ground_truth_filter"]
    agg = case["ground_truth_agg"]

    filtered = df.copy()
    for k, v in filters.items():
        filtered = filtered[filtered[k].str.lower() == v.lower()]

    if agg == "mean":
        return round(filtered[col].mean(), 3)
    elif agg == "count":
        return len(filtered)
    elif agg == "max_era":
        return df.groupby("era")[col].mean().idxmax()
    return None


# ---------------------------------------------------------------------------
# Retrieval precision evaluation (no LLM calls needed)
# ---------------------------------------------------------------------------

def eval_retrieval_precision(agent: AutonomousMemoryAgent) -> List[Dict]:
    """
    Seed long-term memory with synthetic entries covering all topics,
    then measure whether retrieval ranks relevant topics above irrelevant ones.
    """
    print("\n" + "="*60)
    print("EVAL 1: RETRIEVAL PRECISION")
    print("="*60)

    # Seed memory with one entry per topic
    synthetic_memories = [
        Memory(
            question="How does energy vary across eras?",
            answer="Energy averages 0.65 in early eras dropping to 0.48 in folklore.",
            timestamp=datetime.now().isoformat(),
            topic="musical_analysis",
            importance=0.7,
        ),
        Memory(
            question="How has the career evolved from debut to midnights?",
            answer="Career spans 10 eras with notable genre shifts per album.",
            timestamp=datetime.now().isoformat(),
            topic="career_evolution",
            importance=0.7,
        ),
        Memory(
            question="Recommend songs similar to All Too Well",
            answer="Similar songs include Red, The Last Time based on valence and tempo.",
            timestamp=datetime.now().isoformat(),
            topic="recommendations",
            importance=0.6,
        ),
        Memory(
            question="What are the lyrical themes in folklore?",
            answer="Folklore is dominated by introspective narratives and fictional storytelling.",
            timestamp=datetime.now().isoformat(),
            topic="lyrical_analysis",
            importance=0.7,
        ),
        Memory(
            question="Compare reputation and 1989 albums",
            answer="Reputation has higher energy (0.68) vs 1989 (0.62).",
            timestamp=datetime.now().isoformat(),
            topic="comparisons",
            importance=0.7,
        ),
    ]

    # Build embeddings and insert into long-term memory
    for mem in synthetic_memories:
        mem.embedding = agent._create_embedding(mem.question + " " + mem.answer)
    agent.long_term_memory = synthetic_memories

    results = []
    for query, relevant_topics, irrelevant_topics in RETRIEVAL_TEST_PAIRS:
        retrieved = agent._retrieve_relevant_memories(query, k=3)
        retrieved_topics = [m.topic for m, _ in retrieved]
        retrieved_scores = [s for _, s in retrieved]

        # Precision: fraction of top-k that are relevant
        relevant_retrieved = sum(1 for t in retrieved_topics if t in relevant_topics)
        precision = relevant_retrieved / len(retrieved_topics) if retrieved_topics else 0.0

        # Check that relevant topics score higher than irrelevant ones on average
        relevant_scores = [s for (m, s) in retrieved if m.topic in relevant_topics]
        irrelevant_scores = [s for (m, s) in retrieved if m.topic in irrelevant_topics]
        rank_correct = (
            (np.mean(relevant_scores) > np.mean(irrelevant_scores))
            if relevant_scores and irrelevant_scores
            else None
        )

        result = {
            "query": query[:60],
            "retrieved_topics": str(retrieved_topics),
            "precision_at_3": round(precision, 3),
            "rank_order_correct": rank_correct,
            "top_score": round(retrieved_scores[0], 3) if retrieved_scores else 0.0,
        }
        results.append(result)

        status = "PASS" if precision >= 0.5 else "FAIL"
        print(f"  [{status}] {query[:50]}")
        print(f"         Retrieved: {retrieved_topics} | Precision@3: {precision:.2f}")

    # Reset long-term memory after eval
    agent.long_term_memory = []
    return results


# ---------------------------------------------------------------------------
# Data grounding evaluation (requires LLM)
# ---------------------------------------------------------------------------

def eval_data_grounding(agent: AutonomousMemoryAgent, df: pd.DataFrame) -> List[Dict]:
    """
    Run factual questions and compare agent answers against pandas ground truth.
    Also verify that out-of-scope questions are refused.
    """
    print("\n" + "="*60)
    print("EVAL 2: DATA GROUNDING (requires LLM)")
    print("="*60)

    results = []

    # In-scope factual questions
    for case in FACTUAL_QUESTIONS:
        ground_truth = compute_ground_truth(df, case)
        print(f"\n  Q: {case['question']}")
        print(f"  Ground truth ({case['ground_truth_agg']} of {case['ground_truth_col']}): {ground_truth}")

        answer = agent.ask(case["question"], use_memory=False)

        # Check if the ground truth value appears in the answer (as string)
        gt_str = str(ground_truth)
        # For floats, check if a rounded version appears
        if isinstance(ground_truth, float):
            gt_rounded = str(round(ground_truth, 2))
            gt_in_answer = gt_rounded in answer or gt_str in answer
        else:
            gt_in_answer = gt_str.lower() in answer.lower()

        refused = any(
            phrase in answer.lower()
            for phrase in ["don't have data", "cannot answer", "not in the dataset"]
        )

        result = {
            "question": case["question"][:60],
            "type": "in_scope",
            "ground_truth": ground_truth,
            "answer_snippet": answer[:150].replace("\n", " "),
            "ground_truth_cited": gt_in_answer,
            "incorrectly_refused": refused,
            "pass": gt_in_answer and not refused,
        }
        results.append(result)

        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{status}] Ground truth cited: {gt_in_answer} | Refused: {refused}")

    # Out-of-scope questions - expect refusal
    for question in OUT_OF_SCOPE_QUESTIONS:
        print(f"\n  Q (out-of-scope): {question}")
        answer = agent.ask(question, use_memory=False)

        refused = any(
            phrase in answer.lower()
            for phrase in ["don't have data", "cannot answer", "not in the dataset",
                           "no data", "unable to"]
        )

        result = {
            "question": question[:60],
            "type": "out_of_scope",
            "ground_truth": "REFUSE",
            "answer_snippet": answer[:150].replace("\n", " "),
            "ground_truth_cited": refused,
            "incorrectly_refused": False,
            "pass": refused,
        }
        results.append(result)

        status = "PASS" if refused else "FAIL"
        print(f"  [{status}] Correctly refused: {refused}")

    return results


# ---------------------------------------------------------------------------
# Topic classification evaluation (no LLM calls)
# ---------------------------------------------------------------------------

def eval_topic_classification(agent: AutonomousMemoryAgent) -> List[Dict]:
    """
    Verify that _classify_topic assigns expected labels to known question types.
    """
    print("\n" + "="*60)
    print("EVAL 3: TOPIC CLASSIFICATION")
    print("="*60)

    test_cases = [
        # career_evolution: 'era' as standalone word, not substring of 'average'
        ("How has the sound evolved across eras?", "musical evolution across eras", "career_evolution"),
        # lyrical_analysis: clear lyric/theme keywords, no recommendation intent
        ("What are the lyrical themes in folklore?", "narrative introspective storytelling", "lyrical_analysis"),
        # recommendations: intent keyword 'recommend' should fire before content keywords
        ("Recommend songs with high energy", "high energy tempo similar feel", "recommendations"),
        # comparisons: 'vs' as standalone word fires comparisons before musical_analysis
        ("Compare reputation vs 1989 energy levels", "energy 0.68 vs 0.62 difference", "comparisons"),
        # musical_analysis: 'tempo' as standalone word, no era/album/career words present
        ("What is the average tempo?", "tempo 120 bpm", "musical_analysis"),
    ]

    results = []
    for question, answer, expected_topic in test_cases:
        predicted = agent._classify_topic(question, answer)
        passed = predicted == expected_topic
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{question[:45]}' -> {predicted} (expected: {expected_topic})")

        results.append({
            "question": question[:60],
            "expected_topic": expected_topic,
            "predicted_topic": predicted,
            "pass": passed,
        })

    return results


# ---------------------------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------------------------

def save_results(retrieval: List[Dict], grounding: List[Dict],
                 classification: List[Dict], output_dir: str):
    """Save all eval results to results/ as CSV files and print summary."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def write_csv(data: List[Dict], name: str):
        path = os.path.join(output_dir, f"{name}_{timestamp}.csv")
        if not data:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved: {path}")

    write_csv(retrieval, "eval_memory_retrieval")
    write_csv(grounding, "eval_memory_grounding")
    write_csv(classification, "eval_memory_classification")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if retrieval:
        p = sum(1 for r in retrieval if r["precision_at_3"] >= 0.5) / len(retrieval)
        print(f"  Retrieval precision@3 >= 0.5:  {p*100:.0f}% ({sum(1 for r in retrieval if r['precision_at_3'] >= 0.5)}/{len(retrieval)})")

    if grounding:
        in_scope = [r for r in grounding if r["type"] == "in_scope"]
        out_scope = [r for r in grounding if r["type"] == "out_of_scope"]
        if in_scope:
            p = sum(1 for r in in_scope if r["pass"]) / len(in_scope)
            print(f"  In-scope grounding accuracy:   {p*100:.0f}% ({sum(1 for r in in_scope if r['pass'])}/{len(in_scope)})")
        if out_scope:
            p = sum(1 for r in out_scope if r["pass"]) / len(out_scope)
            print(f"  Out-of-scope refusal rate:     {p*100:.0f}% ({sum(1 for r in out_scope if r['pass'])}/{len(out_scope)})")

    if classification:
        p = sum(1 for r in classification if r["pass"]) / len(classification)
        print(f"  Topic classification accuracy: {p*100:.0f}% ({sum(1 for r in classification if r['pass'])}/{len(classification)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(run_grounding: bool = True):
    """
    Run full evaluation suite.

    Args:
        run_grounding: Set False to skip LLM-dependent grounding eval
                       (useful for CI or offline testing).
    """
    print("="*60)
    print("MEMORY AGENT EVALUATION SUITE")
    print("="*60)

    print("\nLoading data...")
    merged_df = load_and_merge_data(config.DATA_DIR, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    df = define_eras(merged_df)

    print("Initializing agent (fresh, no memory file)...")
    agent = AutonomousMemoryAgent(
        df=df,
        memory_file="/tmp/eval_memory_agent_tmp.json"
    )

    retrieval_results = eval_retrieval_precision(agent)
    classification_results = eval_topic_classification(agent)

    grounding_results = []
    if run_grounding:
        grounding_results = eval_data_grounding(agent, df)
    else:
        print("\n[SKIPPED] Data grounding eval (run_grounding=False)")

    save_results(
        retrieval_results,
        grounding_results,
        classification_results,
        output_dir=config.RESULTS_DIR
    )

    # Clean up temp memory file
    if os.path.exists("/tmp/eval_memory_agent_tmp.json"):
        os.remove("/tmp/eval_memory_agent_tmp.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM-dependent grounding eval (retrieval and classification only)"
    )
    args = parser.parse_args()
    run_eval(run_grounding=not args.no_llm)
