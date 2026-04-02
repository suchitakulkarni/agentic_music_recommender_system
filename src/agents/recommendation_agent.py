"""
Autonomous Recommendation Agent with:
- Active preference learning
- Exploration vs exploitation strategy
- Multi-step iterative refinement
- Explanation depth calibration
- User preference modeling
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from src import config

import csv
import os


@dataclass
class SongRecommendationExplanation:
    """
    Structured explanation for a single recommended song.

    Layer 1 fields are computed deterministically from the dataset.
    Layer 2 (grounded_explanation) is LLM-rendered from Layer 1 only.
    Layer 3 (easter_egg_aside) is optional and only populated when
    confirmed callbacks exist between songs in the recommendation set.
    """
    # Layer 1: pure data, no LLM
    song_name: str
    similarity_score: float
    lyric_similarity: float
    audio_similarity: float
    dominant_signal: str          # "lyric" | "audio" | "balanced"
    shared_era: bool
    era: str
    shared_topic_cluster: bool
    topic_cluster_name: str
    feature_deltas: Dict[str, float]
    explore_or_exploit: str       # "exploit" | "explore"
    memory_reference: Optional[str] = None

    # Layer 2: LLM renders Layer 1 into natural language, grounded strictly
    grounded_explanation: str = ""

    # Layer 3: Easter egg aside, only when callbacks confirmed in dataset
    easter_egg_aside: Optional[str] = None


class ExplanationDepth(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"


# Positive and negative sentiment phrase lists for feedback parsing.
# Using multi-word phrases avoids misclassifying negations like "not bad".
POSITIVE_PHRASES = ['not bad', 'good', 'great', 'love', 'perfect', 'better', 'liked it', 'like it']
NEGATIVE_PHRASES = ['bad', 'dislike', 'too slow', 'too fast', 'too sad', 'too happy',
                    'too electronic', 'too acoustic', 'too loud', 'too quiet', 'not good',
                    'not great', 'not feeling']


def _classify_sentiment(feedback_lower: str) -> Tuple[bool, bool]:
    """
    Classify feedback as positive and/or negative using phrase matching.

    Returns (is_positive, is_negative). Both can be False for neutral feedback.
    Longer phrases are checked first so negations such as 'not bad' are caught
    before the bare word 'bad'.
    """
    phrases_sorted_pos = sorted(POSITIVE_PHRASES, key=len, reverse=True)
    phrases_sorted_neg = sorted(NEGATIVE_PHRASES, key=len, reverse=True)

    is_positive = any(p in feedback_lower for p in phrases_sorted_pos)
    is_negative = any(p in feedback_lower for p in phrases_sorted_neg)
    return is_positive, is_negative


@dataclass
class UserPreferenceModel:
    """
    Dynamic user preference model that learns over time.

    KEY CONCEPT: Instead of stateless recommendations, the agent
    builds a MODEL of user preferences that improves with each interaction.
    """
    # Feature preferences (learned weights)
    preferred_energy_range: Tuple[float, float] = (0.0, 1.0)
    preferred_valence_range: Tuple[float, float] = (0.0, 1.0)
    preferred_tempo_range: Tuple[float, float] = (0, 300)
    preferred_acousticness: Optional[float] = None

    # Era preferences
    liked_eras: List[str] = field(default_factory=list)
    disliked_eras: List[str] = field(default_factory=list)

    # Interaction history
    liked_songs: List[str] = field(default_factory=list)
    disliked_songs: List[str] = field(default_factory=list)

    # Meta-preferences
    explanation_preference: ExplanationDepth = ExplanationDepth.STANDARD
    exploration_tolerance: float = 0.2  # How willing to suggest outliers

    def update_from_feedback(self, song_data: Dict, feedback: str):
        """
        Learn from user feedback.

        This is ACTIVE LEARNING - the model improves with each interaction.
        Uses phrase-level sentiment classification to avoid misclassifying
        negations such as "not bad".
        """
        feedback_lower = feedback.lower()
        is_positive, is_negative = _classify_sentiment(feedback_lower)

        song_name = song_data.get('Song_Name', '')

        if is_positive and not is_negative:
            if song_name not in self.liked_songs:
                self.liked_songs.append(song_name)
            era = song_data.get('era')
            if era and era not in self.liked_eras:
                self.liked_eras.append(era)
        elif is_negative and not is_positive:
            if song_name not in self.disliked_songs:
                self.disliked_songs.append(song_name)
            era = song_data.get('era')
            if era and era not in self.disliked_eras:
                self.disliked_eras.append(era)

        # Update feature preferences based on feedback keywords
        if 'slow' in feedback_lower or 'tempo' in feedback_lower:
            if is_negative:
                # User wants faster
                self.preferred_tempo_range = (
                    max(120, self.preferred_tempo_range[0]),
                    self.preferred_tempo_range[1]
                )

        if 'sad' in feedback_lower or 'happy' in feedback_lower:
            current_valence = song_data.get('valence', 0.5)
            if 'sad' in feedback_lower and is_negative:
                # User wants happier
                self.preferred_valence_range = (
                    max(0.5, current_valence),
                    self.preferred_valence_range[1]
                )
            elif 'happy' in feedback_lower and is_negative:
                # User wants sadder
                self.preferred_valence_range = (
                    self.preferred_valence_range[0],
                    min(0.5, current_valence)
                )

        if 'electronic' in feedback_lower or 'acoustic' in feedback_lower:
            if 'electronic' in feedback_lower and is_negative:
                self.preferred_acousticness = 0.6  # Prefer more acoustic
            elif 'acoustic' in feedback_lower and is_negative:
                self.preferred_acousticness = 0.3  # Prefer less acoustic

    def get_preference_summary(self) -> str:
        """Get human-readable summary of learned preferences."""
        summary = "Learned preferences:\n"

        if self.liked_songs:
            summary += f"  Liked songs: {', '.join(self.liked_songs[-3:])}\n"

        if self.preferred_valence_range != (0.0, 1.0):
            summary += (
                f"  Valence range: "
                f"{self.preferred_valence_range[0]:.2f}-{self.preferred_valence_range[1]:.2f}\n"
            )

        if self.preferred_tempo_range != (0, 300):
            summary += (
                f"  Tempo range: "
                f"{self.preferred_tempo_range[0]:.0f}-{self.preferred_tempo_range[1]:.0f} BPM\n"
            )

        if self.liked_eras:
            summary += f"  Preferred eras: {', '.join(self.liked_eras)}\n"

        if self.disliked_eras:
            summary += f"  Avoided eras: {', '.join(self.disliked_eras)}\n"

        return summary


class AutonomousRecommendationAgent:
    """
    Recommendation agent with autonomous learning and adaptation capabilities.
    """

    def __init__(self, similarity_results, df, model: str = config.MODEL,
                 easter_egg_csv: Optional[str] = None):
        if config.USE_OPENAI:
            self.client = OpenAIClient()
        else:
            self.client = OllamaClient(model=model)

        self.similarity_results = similarity_results
        self.df = df
        self.user_model = UserPreferenceModel()
        self.interaction_count = 0

        # Store lyric and audio similarity matrices separately for Layer 1 computation.
        # Falls back gracefully if not present in similarity_results.
        self.lyric_sim = similarity_results.get('lyric_similarity')
        self.audio_sim = similarity_results.get('audio_similarity')
        self.hybrid_sim = similarity_results['hybrid_similarity']

        # Load Easter egg dataset if provided.
        # Keyed as (song_name_lower, song_name_lower) -> list of easter egg strings.
        self.easter_eggs: Dict[tuple, List[str]] = {}
        if easter_egg_csv and os.path.exists(easter_egg_csv):
            self._load_easter_eggs(easter_egg_csv)

        # Track which recommendations used exploration for Layer 1 signal.
        self._last_explore_indices: set = set()

    def _load_easter_eggs(self, csv_path: str) -> None:
        """
        Load Easter egg dataset from CSV into a lookup keyed by song pair.
        Only loads high-confidence cross-song lyrical callbacks.
        """
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('easter_egg_type') != 'lyrical_callback':
                    continue
                if row.get('confidence', 'medium') not in ('high',):
                    continue
                song = row['song'].lower().replace(' ', '')
                egg_text = row['easter_egg']
                # Store under the song itself; cross-song matching done at query time.
                key = song
                if key not in self.easter_eggs:
                    self.easter_eggs[key] = []
                self.easter_eggs[key].append(egg_text)

    def _probe_preferences(self, initial_recs: pd.DataFrame, feedback: str) -> Dict[str, any]:
        """
        ACTIVE PREFERENCE PROBING - ask clarifying questions.

        Instead of guessing what user means, ASK for clarification.
        This is ACTIVE LEARNING through dialogue.
        """
        probe_prompt = f"""The user gave this feedback on song recommendations:
"{feedback}"

The recommendations had these characteristics:
- Avg energy: {initial_recs['energy'].mean():.2f}
- Avg valence: {initial_recs['valence'].mean():.2f}
- Avg tempo: {initial_recs['tempo'].mean():.0f} BPM
- Avg acousticness: {initial_recs['acousticness'].mean():.2f}
- Eras: {initial_recs['era'].unique().tolist()}

The feedback is somewhat ambiguous. Generate 2-3 clarifying questions to understand preferences better.

Format each question on a new line starting with "Q:"

Examples:
Q: By "too slow", do you mean the tempo specifically, or the overall energy level?
Q: Would you prefer more upbeat themes, or just more energetic production?
"""

        self.client.reset_conversation()
        response = self.client.chat_interactive(probe_prompt)

        questions = []
        for line in response.split('\n'):
            if line.strip().startswith('Q:'):
                questions.append(line.strip()[2:].strip())

        return {
            'questions': questions[:3],
            'raw_response': response
        }

    def _exploration_vs_exploitation(self, song_idx: int, similar_indices: np.ndarray,
                                     n_recommendations: int, iteration: int) -> np.ndarray:
        """
        EXPLORATION VS EXPLOITATION strategy.

        Early iterations: Explore diverse options.
        Later iterations: Exploit known preferences.

        Exploration is active from iteration 0 onward. Diverse songs are
        interleaved into the candidate pool at fixed positions so they
        survive the downstream head() call rather than being silently dropped.
        """
        exploration_rate = self.user_model.exploration_tolerance / (1 + iteration * 0.5)

        if np.random.random() < exploration_rate:
            print(f"  [EXPLORATION] Including diverse recommendations (rate: {exploration_rate:.2f})")

            current_era = self.df.iloc[song_idx]['era']
            other_eras_mask = self.df['era'] != current_era
            other_eras_positions = np.where(other_eras_mask)[0]

            #if len(other_eras_positions) > 0:
            #    n_diverse = min(2, len(other_eras_positions))
            #    diverse_indices = np.random.choice(
            #        other_eras_positions, size=n_diverse, replace=False
            #    )
            if len(other_eras_positions) > 0:
                viable_diverse = np.array([
                pos for pos in other_eras_positions
                if self.hybrid_sim[song_idx, pos] >= 0.3
                ])
                if len(viable_diverse) == 0:
                    self._last_explore_indices = set()
                    return similar_indices
                n_diverse = min(2, len(viable_diverse))
                diverse_indices = np.random.choice(
                    viable_diverse, size=n_diverse, replace=False
                    )
                # Record which positions are explore picks for Layer 1 labelling.
                self._last_explore_indices = set(diverse_indices.tolist())
                exploit_slots = n_recommendations - n_diverse
                combined = np.concatenate([similar_indices[:exploit_slots], diverse_indices])
                return combined

        self._last_explore_indices = set()
        return similar_indices

    def _filter_by_preferences(self, recommendations: pd.DataFrame,
                               n_recommendations: int) -> pd.DataFrame:
        """
        Filter recommendations based on learned preferences.

        Falls back gracefully if filters are too restrictive, warning the user.
        The minimum viable size is tied to n_recommendations rather than a
        hard-coded constant.
        """
        filtered = recommendations.copy()

        valence_min, valence_max = self.user_model.preferred_valence_range
        if valence_min > 0.0 or valence_max < 1.0:
            filtered = filtered[
                (filtered['valence'] >= valence_min) &
                (filtered['valence'] <= valence_max)
            ]

        tempo_min, tempo_max = self.user_model.preferred_tempo_range
        if tempo_min > 0 or tempo_max < 300:
            filtered = filtered[
                (filtered['tempo'] >= tempo_min) &
                (filtered['tempo'] <= tempo_max)
            ]

        if self.user_model.disliked_eras:
            filtered = filtered[~filtered['era'].isin(self.user_model.disliked_eras)]

        if self.user_model.disliked_songs:
            filtered = filtered[~filtered['Song_Name'].isin(self.user_model.disliked_songs)]

        if len(filtered) < n_recommendations:
            print("  [WARNING] Preferences too restrictive, relaxing filters")
            return recommendations

        print(f"  [FILTERING] Applied preferences: {len(recommendations)} -> {len(filtered)} songs")
        return filtered

    def _calibrate_explanation_depth(self, request: Optional[str] = None) -> ExplanationDepth:
        """
        EXPLANATION DEPTH CALIBRATION - adapt to user preferences.

        This shows ADAPTIVE BEHAVIOR - adjusting output style to user needs.
        """
        if request:
            if any(word in request.lower() for word in ['brief', 'quick', 'just', 'simply']):
                self.user_model.explanation_preference = ExplanationDepth.MINIMAL
                print("  [CALIBRATION] User prefers minimal explanations")
            elif any(word in request.lower() for word in ['detailed', 'deep', 'thorough', 'explain']):
                self.user_model.explanation_preference = ExplanationDepth.DETAILED
                print("  [CALIBRATION] User prefers detailed explanations")

        return self.user_model.explanation_preference

    def recommend_with_learning(self, song_name: str, n_recommendations: int = 5,
                                iteration: int = 0) -> Dict:
        """
        Main recommendation method with autonomous learning.

        Integrates all autonomous capabilities:
        - Input song excluded from candidates
        - Preference filtering
        - Exploration/exploitation
        - Adaptive explanations
        """
        print(f"\n{'='*80}")
        print(f"AUTONOMOUS RECOMMENDATION (Iteration {iteration + 1})")
        print(f"{'='*80}")

        matches = self.df[self.df['Song_Name'].str.lower() == song_name.lower()]
        if matches.empty:
            partial = self.df[self.df['Song_Name'].str.lower().str.contains(song_name.lower(), na=False)]
            if not partial.empty:
                candidates = partial['Song_Name'].tolist()
                suggestion = ', '.join(candidates[:3])
                return {'error': f"Song '{song_name}' not found. Did you mean: {suggestion}?"}
            return {'error': f"Song '{song_name}' not found in dataset."}

        song_label = matches.index[0]
        song_pos = self.df.index.get_loc(song_label)  # positional index for numpy
        song_data = self.df.loc[song_label]

        # Sort all indices by descending similarity, then exclude the input song itself.
        all_sorted = np.argsort(self.hybrid_sim[song_pos])[::-1]
        all_sorted = all_sorted[all_sorted != song_pos]
        similar_indices = all_sorted[:n_recommendations * 2]

        # Apply exploration/exploitation with interleaving
        similar_indices = self._exploration_vs_exploitation(
            song_pos, similar_indices, n_recommendations, iteration
        )

        recommendations = self.df.iloc[similar_indices]

        if iteration > 0 or self.user_model.liked_songs:
            recommendations = self._filter_by_preferences(recommendations, n_recommendations)

        recommendations = recommendations.head(n_recommendations)

        if iteration > 0:
            print(f"\n[USER MODEL]")
            print(self.user_model.get_preference_summary())

        depth = self._calibrate_explanation_depth()

        # Compute similarity scores for Layer 1.
        rec_positions = [self.df.index.get_loc(i) for i in recommendations.index]
        similarity_scores = [self.hybrid_sim[song_pos, pos] for pos in rec_positions]

        # Layer 1: deterministic, no LLM.
        layer1 = self._compute_layer1(song_pos, song_data, recommendations, similarity_scores)

        # Layer 3: Easter egg aside, only when callbacks confirmed.
        easter_egg_aside = self._get_easter_egg_aside(song_data['Song_Name'], recommendations)

        # Layer 2: grounded LLM explanation.
        explanation = self._generate_explanation(
            song_data, recommendations, depth, iteration, layer1, easter_egg_aside
        )

        return {
            'song': song_data['Song_Name'],
            'recommendations': recommendations[['Song_Name', 'Album', 'era']].to_dict('records'),
            'explanation': explanation,
            'similarity_scores': similarity_scores,
            'layer1': layer1,
            'user_model_summary': self.user_model.get_preference_summary() if iteration > 0 else None,
            'recommendations_df': recommendations
        }

    def _compute_layer1(self, song_pos: int, song_data: pd.Series,
                        recommendations: pd.DataFrame,
                        similarity_scores: List[float]) -> List[SongRecommendationExplanation]:
        """
        Compute all Layer 1 fields deterministically from the dataset.
        No LLM involvement. Called once per recommendation set.
        """
        audio_features = ['energy', 'valence', 'tempo', 'acousticness',
                          'danceability', 'loudness']
        input_cluster = song_data.get('dominant_topic', -1)
        input_era = song_data.get('era', '')

        explanations = []
        for idx, (label, rec) in enumerate(recommendations.iterrows()):
            rec_pos = self.df.index.get_loc(label)
            score = similarity_scores[idx]

            # Lyric and audio similarity scores.
            lyric_score = (
                float(self.lyric_sim[song_pos, rec_pos])
                if self.lyric_sim is not None else 0.5
            )
            audio_score = (
                float(self.audio_sim[song_pos, rec_pos])
                if self.audio_sim is not None else 0.5
            )

            # Dominant signal: whichever component contributes more to hybrid score.
            lyric_contrib = 0.6 * lyric_score
            audio_contrib = 0.4 * audio_score
            if abs(lyric_contrib - audio_contrib) < 0.05:
                dominant = "balanced"
            elif lyric_contrib > audio_contrib:
                dominant = "lyric"
            else:
                dominant = "audio"

            # Era match.
            shared_era = rec.get('era', '') == input_era

            # Topic cluster match and name.
            rec_cluster = rec.get('dominant_topic', -1)
            shared_cluster = (rec_cluster == input_cluster) and input_cluster != -1

            # Map topic cluster id to human-readable name using BERTopic assignment.
            # Topic 0 is Core Narrative per the stable clustering result.
            cluster_name = "core_narrative" if rec_cluster == 0 else "stylistic_departure"

            # Feature deltas: absolute difference per audio feature.
            feature_deltas = {}
            for feat in audio_features:
                input_val = song_data.get(feat, None)
                rec_val = rec.get(feat, None)
                if input_val is not None and rec_val is not None:
                    feature_deltas[feat] = round(float(rec_val) - float(input_val), 3)

            # Explore vs exploit: was this index in the diverse set?
            explore_or_exploit = (
                "explore" if rec_pos in self._last_explore_indices else "exploit"
            )

            rec_name_key = rec['Song_Name'].lower().replace(' ', '')
            explanations.append(SongRecommendationExplanation(
                song_name=rec['Song_Name'],
                similarity_score=round(score, 3),
                lyric_similarity=round(lyric_score, 3),
                audio_similarity=round(audio_score, 3),
                dominant_signal=dominant,
                shared_era=shared_era,
                era=rec.get('era', ''),
                shared_topic_cluster=shared_cluster,
                topic_cluster_name=cluster_name,
                feature_deltas=feature_deltas,
                explore_or_exploit=explore_or_exploit,
                memory_reference=None,
            ))

        return explanations

    def _get_easter_egg_aside(self, input_song: str,
                               recommendations: pd.DataFrame) -> Optional[str]:
        """
        Check for confirmed lyrical callbacks between input song and recommendations.
        Returns a formatted aside string if any high-confidence callbacks exist,
        None otherwise. Silence is correct when no callbacks are found.
        """
        if not self.easter_eggs:
            return None

        input_key = input_song.lower().replace(' ', '')
        rec_keys = [
            r['Song_Name'].lower().replace(' ', '')
            for _, r in recommendations.iterrows()
        ]

        found_callbacks = []
        # Check input song's callbacks referencing any recommended song.
        for egg in self.easter_eggs.get(input_key, []):
            for rec_key in rec_keys:
                if rec_key in egg.lower():
                    found_callbacks.append(egg)
                    break

        # Check recommended songs' callbacks referencing the input song.
        for rec_key in rec_keys:
            for egg in self.easter_eggs.get(rec_key, []):
                if input_key in egg.lower():
                    found_callbacks.append(egg)
                    break

        if not found_callbacks:
            return None

        # One separate LLM call, strictly scoped to the callback data.
        callback_text = "\n".join(f"- {e}" for e in found_callbacks[:3])
        aside_prompt = (
            f"The following confirmed lyrical callbacks exist between "
            f"'{input_song}' and songs in this recommendation set:\n\n"
            f"{callback_text}\n\n"
            f"Write a single conversational sentence starting with "
            f"'By the way,' that names the specific callback. "
            f"Do not describe the songs generally. "
            f"Do not use prior knowledge beyond what is listed above."
        )
        self.client.reset_conversation()
        return self.client.chat_interactive(aside_prompt)

    def _generate_explanation(self, song_data: pd.Series, recommendations: pd.DataFrame,
                              depth: ExplanationDepth, iteration: int,
                              layer1: Optional[List[SongRecommendationExplanation]] = None,
                              easter_egg_aside: Optional[str] = None) -> str:
        """
        Generate explanation grounded strictly in Layer 1 structured fields.

        The LLM prompt explicitly forbids use of prior knowledge about the songs.
        All claims must reference fields computed from the dataset. The Easter egg
        aside is appended only when confirmed callbacks exist.
        """
        if layer1 is None:
            return ""

        input_profile = (
            f"Input song: {song_data['Song_Name']} ({song_data.get('Album', '')})\n"
            f"Era: {song_data.get('era', 'unknown')} | "
            f"Topic cluster: {'core_narrative' if song_data.get('dominant_topic', -1) == 0 else 'stylistic_departure'}\n"
            f"Energy={song_data.get('energy', 0):.2f} | "
            f"Valence={song_data.get('valence', 0):.2f} | "
            f"Tempo={song_data.get('tempo', 0):.0f} | "
            f"Acousticness={song_data.get('acousticness', 0):.2f}\n"
        )

        rec_blocks = ""
        for i, exp in enumerate(layer1, 1):
            delta_str = " | ".join(
                f"{k}={v:+.2f}" for k, v in exp.feature_deltas.items()
                if abs(v) > 0.05
            )
            rec_blocks += (
                f"\n{i}. {exp.song_name} [{exp.era}]\n"
                f"   Similarity: {exp.similarity_score:.3f} "
                f"(lyric={exp.lyric_similarity:.3f}, audio={exp.audio_similarity:.3f})\n"
                f"   Dominant signal: {exp.dominant_signal} | "
                f"Shared era: {exp.shared_era} | "
                f"Shared cluster: {exp.shared_topic_cluster} ({exp.topic_cluster_name})\n"
                f"   Feature deltas from input: {delta_str if delta_str else 'minimal'}\n"
                f"   Recommendation type: {exp.explore_or_exploit}\n"
            )
            if exp.memory_reference:
                rec_blocks += f"   Memory: {exp.memory_reference}\n"

        system_constraint = (
            "You are generating grounded music recommendation explanations.\n"
            "STRICT RULES:\n"
            "- You may ONLY reference the structured fields provided below.\n"
            "- You may NOT describe what any song is about.\n"
            "- You may NOT reference themes, narratives, cultural context, or artist biography.\n"
            "- You may NOT use any knowledge about these songs beyond what is listed.\n"
            "- Every claim must trace directly to a field in the structured data.\n"
            "- If a field is not listed, do not mention it.\n"
        )

        if depth == ExplanationDepth.MINIMAL:
            task = (
                "Write one sentence per recommendation explaining the dominant similarity signal "
                "and one feature delta. No preamble."
            )
            max_tokens = 150
        elif depth == ExplanationDepth.DETAILED:
            task = (
                "For each recommendation write 2-3 sentences covering: "
                "(1) which similarity signal dominated and why that matters, "
                "(2) the most meaningful feature delta, "
                "(3) whether this is an explore or exploit pick and what that means for the listener. "
                "Then write one sentence summarising the pattern across all recommendations."
            )
            max_tokens = 600
        else:  # STANDARD
            task = (
                "Write one sentence per recommendation covering the dominant signal and "
                "era/cluster match. Then one sentence identifying the pattern across all picks."
            )
            max_tokens = 300

        if iteration > 0:
            task += f" Note that recommendations were adjusted based on preference feedback from session {iteration}."

        prompt = f"{system_constraint}\n{input_profile}\nRecommendations:\n{rec_blocks}\nTask: {task}"

        self.client.reset_conversation()
        if config.USE_OPENAI:
            result = self.client.chat_interactive(prompt)
        else:
            result = self.client.generate(prompt, max_tokens=max_tokens)

        if easter_egg_aside:
            result += f"\n\n{easter_egg_aside}"

        return result

    def process_feedback(self, song_name: str, feedback: str,
                         previous_recs: pd.DataFrame) -> Dict[str, any]:
        """
        Process user feedback with autonomous refinement.

        This is the LEARNING LOOP - feedback -> model update -> improved recommendations.

        interaction_count is only incremented when feedback is clear enough to
        trigger a refinement pass, not during probing rounds.
        """
        print(f"\n[FEEDBACK PROCESSING]")

        matches = self.df[self.df['Song_Name'].str.lower() == song_name.lower()]
        if matches.empty:
            return {'error': 'Song not found'}

        song_data = matches.iloc[0].to_dict()

        print("  Updating preference model...")
        self.user_model.update_from_feedback(song_data, feedback)

        needs_probing = self._needs_clarification(feedback)

        if needs_probing:
            print("  Feedback ambiguous - probing for clarification")
            # Do not increment interaction_count during probing: the model
            # has not yet produced a new set of recommendations.
            probe_result = self._probe_preferences(previous_recs, feedback)
            return {
                'action': 'probe',
                'questions': probe_result['questions'],
                'message': 'I have some clarifying questions to better understand your preferences:'
            }
        else:
            print("  Feedback clear - refining recommendations")
            self.interaction_count += 1
            refined = self.recommend_with_learning(
                song_name,
                n_recommendations=5,
                iteration=self.interaction_count
            )

            return {
                'action': 'refine',
                'refined_recommendations': refined,
                'message': 'Based on your feedback, here are refined recommendations:'
            }

    def _needs_clarification(self, feedback: str) -> bool:
        """
        Decide if feedback needs clarification.

        Short feedback is ambiguous only when it lacks clear sentiment signal.
        Explicit positive feedback like "I liked it" should not trigger probing
        even though it is fewer than 5 words.
        """
        feedback_lower = feedback.lower()

        # Clear actionable negative feedback: act immediately
        clear_negative = [
            'too slow', 'too fast', 'too sad', 'too happy',
            'too electronic', 'too acoustic'
        ]
        if any(phrase in feedback_lower for phrase in clear_negative):
            return False

        # Clear positive feedback: no need to probe
        is_positive, is_negative = _classify_sentiment(feedback_lower)
        if is_positive and not is_negative:
            return False

        # Short feedback with no clear signal is ambiguous
        if len(feedback.split()) < 5:
            return True

        # Explicitly uncertain language
        uncertain_words = ['maybe', 'not sure', 'kind of', 'sort of', 'somewhat']
        if any(word in feedback_lower for word in uncertain_words):
            return True

        return False

    def suggest_discovery_path(self, current_song: str) -> Dict[str, any]:
        """
        Proactively suggest a discovery path through the catalog.

        All songs are selected from the dataset using the similarity matrix and
        era-diversity heuristics. No LLM is used to select songs, eliminating
        hallucination of out-of-catalog titles. The LLM only renders a grounded
        transition sentence per step, constrained to Layer 1 fields.
        """
        matches = self.df[self.df['Song_Name'].str.lower() == current_song.lower()]
        if matches.empty:
            return {'error': 'Song not found'}

        song_label = matches.index[0]
        song_pos = self.df.index.get_loc(song_label)
        song_data = self.df.loc[song_label]

        # Step 1-2: two closest neighbors (comfort zone).
        all_sorted = np.argsort(self.hybrid_sim[song_pos])[::-1]
        all_sorted = all_sorted[all_sorted != song_pos]
        comfort_indices = all_sorted[:2].tolist()

        # Step 3-4: two most-similar songs from a different era (exploration).
        current_era = song_data.get('era', '')
        other_era_mask = self.df['era'] != current_era
        other_era_positions = np.where(other_era_mask)[0]

        if len(other_era_positions) >= 2:
            other_era_scores = [
                (pos, self.hybrid_sim[song_pos, pos]) for pos in other_era_positions
            ]
            other_era_scores.sort(key=lambda x: x[1], reverse=True)
            explore_indices = [pos for pos, _ in other_era_scores[:2]]
        else:
            explore_indices = all_sorted[2:4].tolist()

        # Step 5: lowest-similarity song that still clears 0.3 (unexpected).
        used = set(comfort_indices + explore_indices + [song_pos])
        unexpected_candidates = [
            pos for pos in reversed(all_sorted.tolist())
            if pos not in used and self.hybrid_sim[song_pos, pos] >= 0.3
        ]
        unexpected_index = [unexpected_candidates[0]] if unexpected_candidates else []

        path_positions = (comfort_indices + explore_indices + unexpected_index)[:5]
        path_songs = self.df.iloc[path_positions]
        similarity_scores = [float(self.hybrid_sim[song_pos, pos]) for pos in path_positions]

        layer1 = self._compute_layer1(song_pos, song_data, path_songs, similarity_scores)
        step_labels = ["Comfort zone", "Comfort zone", "Explore", "Explore", "Unexpected"]

        steps_text = ""
        for i, (exp, label) in enumerate(zip(layer1, step_labels), 1):
            delta_str = " | ".join(
                f"{k}={v:+.2f}" for k, v in exp.feature_deltas.items()
                if abs(v) > 0.05
            ) or "minimal differences"
            steps_text += (
                f"Step {i} [{label}]: {exp.song_name} [{exp.era}]\n"
                f"  similarity={exp.similarity_score:.3f} "
                f"(lyric={exp.lyric_similarity:.3f}, audio={exp.audio_similarity:.3f})\n"
                f"  dominant signal: {exp.dominant_signal} | "
                f"shared era: {exp.shared_era} | cluster: {exp.topic_cluster_name}\n"
                f"  feature deltas from input: {delta_str}\n"
            )

        prompt = (
            f"Generate a discovery path explanation for someone who likes '{current_song}'.\n\n"
            f"STRICT RULES:\n"
            f"- Only reference the structured fields provided below.\n"
            f"- Do not describe what any song is about.\n"
            f"- Do not use prior knowledge about these songs.\n"
            f"- For each step write one sentence explaining the transition "
            f"using only the similarity signal, era, and feature deltas listed.\n\n"
            f"Starting song: {song_data['Song_Name']} | Era: {current_era} | "
            f"Energy={song_data.get('energy', 0):.2f} | "
            f"Valence={song_data.get('valence', 0):.2f}\n\n"
            f"Path:\n{steps_text}"
        )

        self.client.reset_conversation()
        if config.USE_OPENAI:
            response = self.client.chat_interactive(prompt)
        else:
            response = self.client.generate(prompt, max_tokens=400)

        return {
            'discovery_path': response,
            'starting_point': current_song,
            'path_songs': [
                {'song': exp.song_name, 'era': exp.era,
                 'similarity': exp.similarity_score, 'type': label}
                for exp, label in zip(layer1, step_labels)
            ]
        }

    def analyze_listening_patterns(self) -> str:
        """
        Analyze user's listening patterns and provide insights grounded in dataset fields.

        Only references audio features, era distribution, and topic cluster membership
        from the dataset. Does not use prior LLM knowledge about the songs.
        """
        if not self.user_model.liked_songs:
            return "Not enough interaction data to analyze patterns yet. Try rating a few recommendations!"

        liked_data = self.df[self.df['Song_Name'].isin(self.user_model.liked_songs)]

        if len(liked_data) == 0:
            return "Could not find data for liked songs."

        # Compute aggregate statistics from dataset fields only.
        avg_energy = liked_data['energy'].mean()
        avg_valence = liked_data['valence'].mean()
        avg_acousticness = liked_data['acousticness'].mean()
        avg_tempo = liked_data['tempo'].mean()
        era_counts = liked_data['era'].value_counts().to_dict()
        dominant_era = max(era_counts, key=era_counts.get)

        core_narrative_count = (liked_data.get('dominant_topic', pd.Series()) == 0).sum()
        stylistic_departure_count = len(liked_data) - core_narrative_count
        dominant_cluster = (
            "core_narrative" if core_narrative_count >= stylistic_departure_count
            else "stylistic_departure"
        )

        song_lines = ""
        for _, song in liked_data.iterrows():
            cluster = "core_narrative" if song.get('dominant_topic', -1) == 0 else "stylistic_departure"
            song_lines += (
                f"- {song['Song_Name']} | era={song.get('era', 'unknown')} | "
                f"energy={song.get('energy', 0):.2f} | valence={song.get('valence', 0):.2f} | "
                f"acousticness={song.get('acousticness', 0):.2f} | cluster={cluster}\n"
            )

        analysis_prompt = (
            f"STRICT RULES:\n"
            f"- Only reference the structured fields provided below.\n"
            f"- Do not describe what any song is about.\n"
            f"- Do not use prior knowledge about these songs or the artist.\n"
            f"- Every claim must trace to a field in the data.\n\n"
            f"Liked songs ({len(liked_data)}):\n{song_lines}\n"
            f"Aggregate statistics:\n"
            f"- Avg energy: {avg_energy:.2f} | Avg valence: {avg_valence:.2f} | "
            f"Avg acousticness: {avg_acousticness:.2f} | Avg tempo: {avg_tempo:.0f}\n"
            f"- Era distribution: {era_counts}\n"
            f"- Dominant era: {dominant_era}\n"
            f"- Dominant topic cluster: {dominant_cluster} "
            f"({core_narrative_count} core_narrative, {stylistic_departure_count} stylistic_departure)\n\n"
            f"Provide:\n"
            f"1. What the audio feature profile reveals about this listener's taste (2 sentences, "
            f"cite specific field values).\n"
            f"2. Any pattern in era or cluster distribution the listener might not be aware of "
            f"(1-2 sentences, cite specific counts).\n"
            f"3. One concrete exploration suggestion based on the gap between their dominant era "
            f"and underrepresented eras in the data (1 sentence)."
        )

        self.client.reset_conversation()
        if config.USE_OPENAI:
            return self.client.chat_interactive(analysis_prompt)
        else:
            return self.client.generate(analysis_prompt, max_tokens=400)


def interactive_autonomous_recommendations():
    """Interactive session with autonomous recommendation agent."""
    from src.data_loading import load_and_merge_data
    from src.era_analysis import define_eras
    from src.similarity_analysis import create_hybrid_similarity_system

    print("="*80)
    print("AUTONOMOUS RECOMMENDATION AGENT")
    print("="*80)
    print(f"Model: {config.MODEL}\n")

    print("Features:")
    print("  Active preference learning")
    print("  Exploration vs exploitation")
    print("  Adaptive explanations")
    print("  Clarifying questions")
    print("  Discovery paths")
    print()

    print("Loading data...")
    merged_df = load_and_merge_data(config.DATA_DIR, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    df = define_eras(merged_df)

    print("Creating similarity system...")
    sim_results = create_hybrid_similarity_system(df)

    print("Initializing agent...")
    agent = AutonomousRecommendationAgent(sim_results, sim_results['df'])

    print("\nAgent ready!")
    print("\nCommands:")
    print("  rec <song>                          - Get recommendations")
    print("  feedback <text>                     - Provide feedback on last recommendations")
    print("  discover <song>                     - Get a discovery path")
    print("  analyze                             - Analyze your listening patterns")
    print("  preferences                         - Show learned preferences")
    print("  depth [minimal/standard/detailed]   - Set explanation depth")
    print("  quit                                - Exit")
    print()

    last_song = None
    last_recs = None

    while True:
        try:
            command = input("\nYou: ").strip()

            if command.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not command:
                continue

            if command.lower().startswith('depth '):
                depth_str = command[6:].strip().lower()
                if depth_str == 'minimal':
                    agent.user_model.explanation_preference = ExplanationDepth.MINIMAL
                    print("Set to minimal explanations")
                elif depth_str == 'detailed':
                    agent.user_model.explanation_preference = ExplanationDepth.DETAILED
                    print("Set to detailed explanations")
                else:
                    agent.user_model.explanation_preference = ExplanationDepth.STANDARD
                    print("Set to standard explanations")
                continue

            if command.lower() == 'preferences':
                print("\n" + agent.user_model.get_preference_summary())
                continue

            if command.lower() == 'analyze':
                print("\nAnalyzing your listening patterns...")
                analysis = agent.analyze_listening_patterns()
                print("\n" + "="*80)
                print("LISTENING PATTERN ANALYSIS")
                print("="*80)
                print(analysis)
                continue

            if command.lower().startswith('discover '):
                song = command[9:].strip()
                print(f"\nCreating discovery path from '{song}'...")
                result = agent.suggest_discovery_path(song)

                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print("\n" + "="*80)
                    print("DISCOVERY PATH")
                    print("="*80)
                    print(result['discovery_path'])
                continue

            if command.lower().startswith('feedback '):
                if not last_song or last_recs is None:
                    print("No previous recommendations to provide feedback on.")
                    continue

                feedback_text = command[9:].strip()
                print("\nProcessing feedback...")
                result = agent.process_feedback(last_song, feedback_text, last_recs)

                if result['action'] == 'probe':
                    print("\n" + result['message'])
                    for i, q in enumerate(result['questions'], 1):
                        print(f"{i}. {q}")
                    print("\nPlease answer with: feedback <your answer>")

                elif result['action'] == 'refine':
                    print("\n" + result['message'])
                    refined = result['refined_recommendations']

                    print("\n" + "="*80)
                    print("REFINED RECOMMENDATIONS")
                    print("="*80)

                    for i, rec in enumerate(refined['recommendations'], 1):
                        print(f"\n{i}. {rec['Song_Name']}")
                        print(f"   Album: {rec['Album']} | Era: {rec['era']}")
                        print(f"   Similarity: {refined['similarity_scores'][i-1]:.3f}")

                    print("\n" + "-"*80)
                    print("EXPLANATION:")
                    print("-"*80)
                    print(refined['explanation'])

                    if refined['user_model_summary']:
                        print("\n" + "-"*80)
                        print("UPDATED PREFERENCES:")
                        print("-"*80)
                        print(refined['user_model_summary'])

                    last_recs = refined['recommendations_df']

                continue

            if command.lower().startswith('rec '):
                song = command[4:].strip()
                print(f"\nGetting recommendations for '{song}'...")
                result = agent.recommend_with_learning(song)

                if 'error' in result:
                    print(f"Error: {result['error']}")
                    continue

                print("\n" + "="*80)
                print(f"RECOMMENDATIONS FOR: {result['song']}")
                print("="*80)

                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"\n{i}. {rec['Song_Name']}")
                    print(f"   Album: {rec['Album']} | Era: {rec['era']}")
                    print(f"   Similarity: {result['similarity_scores'][i-1]:.3f}")

                print("\n" + "-"*80)
                print("EXPLANATION:")
                print("-"*80)
                print(result['explanation'])

                last_song = song
                last_recs = result['recommendations_df']

                print("\nTip: Provide feedback with 'feedback <your thoughts>'")
                continue

            print("Unknown command. Try:")
            print("  rec Blank Space")
            print("  feedback too slow")
            print("  discover Style")
            print("  analyze")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    interactive_autonomous_recommendations()
