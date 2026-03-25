"""
Autonomous Memory Agent with:
- Semantic memory retrieval (embedding-based)
- Memory consolidation and forgetting
- Proactive memory application
- Meta-learning about the user
- Memory-driven insights
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from src import config


@dataclass
class Memory:
    """Structured memory with metadata."""
    question: str
    answer: str
    timestamp: str
    topic: str
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'answer': self.answer,
            'timestamp': self.timestamp,
            'topic': self.topic,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Memory':
        embedding = np.array(data['embedding']) if data.get('embedding') else None
        return Memory(
            question=data['question'],
            answer=data['answer'],
            timestamp=data['timestamp'],
            topic=data.get('topic', 'general'),
            importance=data.get('importance', 0.5),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed'),
            embedding=embedding
        )


@dataclass
class UserProfile:
    """
    User profile learned over time.

    Captures META-KNOWLEDGE about the user - not just what they asked,
    but patterns in HOW they interact and what they are interested in.
    """
    preferred_topics: Dict[str, int] = field(default_factory=dict)
    preferred_analysis_depth: str = "standard"
    favorite_eras: List[str] = field(default_factory=list)
    question_patterns: Dict[str, int] = field(default_factory=dict)
    interaction_style: str = "balanced"
    total_interactions: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """
        Safe deserialization with explicit key handling and defaults.
        Tolerates schema changes gracefully instead of raising TypeError
        on unexpected or missing keys.
        """
        profile = cls()
        profile.preferred_topics = data.get('preferred_topics', {})
        profile.preferred_analysis_depth = data.get('preferred_analysis_depth', 'standard')
        profile.favorite_eras = data.get('favorite_eras', [])
        profile.question_patterns = data.get('question_patterns', {})
        profile.interaction_style = data.get('interaction_style', 'balanced')
        profile.total_interactions = data.get('total_interactions', 0)
        return profile

    def update(self, question: str, topic: str):
        """Update profile based on new interaction."""
        self.total_interactions += 1
        self.preferred_topics[topic] = self.preferred_topics.get(topic, 0) + 1

        question_lower = question.lower()
        if any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference']):
            pattern = 'comparison'
        elif any(word in question_lower for word in ['why', 'how', 'explain']):
            pattern = 'analytical'
        elif any(word in question_lower for word in ['recommend', 'suggest', 'find', 'similar']):
            pattern = 'exploratory'
        else:
            pattern = 'factual'

        self.question_patterns[pattern] = self.question_patterns.get(pattern, 0) + 1

        total_patterns = sum(self.question_patterns.values())
        if total_patterns > 5:
            analytical_ratio = self.question_patterns.get('analytical', 0) / total_patterns
            exploratory_ratio = self.question_patterns.get('exploratory', 0) / total_patterns

            if analytical_ratio > 0.5:
                self.interaction_style = "analytical"
            elif exploratory_ratio > 0.5:
                self.interaction_style = "exploratory"
            else:
                self.interaction_style = "conversational"

    def get_summary(self) -> str:
        if self.total_interactions == 0:
            return "No interaction history yet."

        summary = f"User Profile ({self.total_interactions} interactions):\n"
        summary += f"  Interaction style: {self.interaction_style}\n"

        if self.preferred_topics:
            top_topics = sorted(self.preferred_topics.items(), key=lambda x: x[1], reverse=True)[:3]
            summary += f"  Top interests: {', '.join([f'{t}({c})' for t, c in top_topics])}\n"

        if self.question_patterns:
            top_patterns = sorted(self.question_patterns.items(), key=lambda x: x[1], reverse=True)[:2]
            summary += f"  Question styles: {', '.join([f'{p}({c})' for p, c in top_patterns])}\n"

        return summary


def _build_data_context(df: pd.DataFrame) -> str:
    """
    Build a compact text summary of the dataset for use in the system prompt.

    This grounds the agent's answers in the actual data without injecting
    the full dataframe into every LLM call.
    """
    context = "AVAILABLE DATASET SUMMARY:\n"
    context += f"Total songs: {len(df)}\n"
    context += f"Eras: {', '.join(sorted(df['era'].unique()))}\n"
    context += f"Albums: {df['Album'].nunique()} total\n\n"

    context += "Songs per era:\n"
    for era, count in df['era'].value_counts().items():
        context += f"  {era}: {count} songs\n"

    context += "\nAudio feature ranges (min - max):\n"
    for col in ['energy', 'valence', 'danceability', 'acousticness', 'tempo']:
        if col in df.columns:
            context += f"  {col}: {df[col].min():.2f} - {df[col].max():.2f}\n"

    context += "\nEra-level averages (energy / valence / acousticness):\n"
    for era, grp in df.groupby('era'):
        e = grp['energy'].mean()
        v = grp['valence'].mean()
        a = grp['acousticness'].mean()
        context += f"  {era}: energy={e:.2f}, valence={v:.2f}, acousticness={a:.2f}\n"

    return context


class AutonomousMemoryAgent:
    """
    Memory agent with semantic retrieval and proactive capabilities.
    """

    def __init__(self, df: pd.DataFrame, model: str = config.MODEL,
                 memory_file: str = config.AGENT_MEMORY_JSON):
        if config.USE_OPENAI:
            self.client = OpenAIClient()
        else:
            self.client = OllamaClient(model=model)

        self.df = df
        self.memory_file = memory_file
        self.short_term_memory: List[Memory] = []
        self.long_term_memory: List[Memory] = []
        self.user_profile = UserProfile()
        self.consolidation_threshold = 10
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self._embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Build data context once at init from the actual dataframe.
        # This fixes the AttributeError where data_context was referenced
        # but never defined.
        self.data_context = _build_data_context(df)

        self._load_memory()

    def _load_memory(self):
        """Load long-term memory and user profile."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)

                self.long_term_memory = [Memory.from_dict(m) for m in data.get('memories', [])]

                # Drop embeddings from a previous (smaller) model so they are
                # re-created lazily at retrieval time with the current model.
                for mem in self.long_term_memory:
                    if mem.embedding is not None and len(mem.embedding) != self._embedding_dim:
                        mem.embedding = None

                profile_data = data.get('user_profile', {})
                if profile_data:
                    # Use from_dict instead of **profile_data to handle schema
                    # changes gracefully without raising TypeError.
                    self.user_profile = UserProfile.from_dict(profile_data)

                print(f"Loaded {len(self.long_term_memory)} memories")
                print(self.user_profile.get_summary())
            except Exception as e:
                print(f"Could not load memory: {e}")

    def _save_memory(self):
        """Save long-term memory and user profile."""
        try:
            data = {
                'memories': [m.to_dict() for m in self.long_term_memory],
                'user_profile': {
                    'preferred_topics': self.user_profile.preferred_topics,
                    'preferred_analysis_depth': self.user_profile.preferred_analysis_depth,
                    'favorite_eras': self.user_profile.favorite_eras,
                    'question_patterns': self.user_profile.question_patterns,
                    'interaction_style': self.user_profile.interaction_style,
                    'total_interactions': self.user_profile.total_interactions
                }
            }

            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Could not save memory: {e}")

    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create a sentence-transformer embedding for semantic similarity.

        Uses all-MiniLM-L6-v2 (384-dim) via the shared EMBEDDING_MODEL config.
        Outputs are L2-normalised by the model, so cosine similarity reduces
        to a dot product — consistent with _semantic_similarity below.
        """
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def _semantic_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))

    def _retrieve_relevant_memories(self, query: str, k: int = 3) -> List[Tuple[Memory, float]]:
        """
        Retrieve semantically similar memories.

        Finds memories based on embedding similarity rather than keyword overlap.
        """
        if not self.long_term_memory:
            return []

        query_embedding = self._create_embedding(query)

        scored_memories = []
        for memory in self.long_term_memory:
            if memory.embedding is None:
                memory.embedding = self._create_embedding(memory.question + " " + memory.answer)

            similarity = self._semantic_similarity(query_embedding, memory.embedding)

            recency_boost = 1.0
            if memory.last_accessed:
                try:
                    days_ago = (datetime.now() - datetime.fromisoformat(memory.last_accessed)).days
                    recency_boost = 1.0 / (1.0 + days_ago * 0.1)
                except ValueError:
                    pass

            final_score = similarity * memory.importance * recency_boost
            scored_memories.append((memory, final_score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)

        for memory, _ in scored_memories[:k]:
            memory.access_count += 1
            memory.last_accessed = datetime.now().isoformat()

        return scored_memories[:k]

    def _classify_topic(self, question: str, answer: str) -> str:
        """
        Classify interaction topic for memory organization.

        Priority order matters: more specific/actional topics (recommendations,
        comparisons) are checked before broad descriptive ones (musical_analysis)
        to avoid generic keywords like 'energy' swallowing intent-bearing queries.

        Word-level matching via split() prevents substring false positives,
        e.g. 'era' matching inside 'average'.
        """
        words = set((question + " " + answer).lower().split())

        # Check actional/intent topics first — these have clear user intent
        # that should not be overridden by content keywords
        if any(w in words for w in ['recommend', 'similar', 'like', 'suggestion', 'suggest']):
            return 'recommendations'
        elif any(w in words for w in ['compare', 'comparison', 'difference', 'vs', 'versus']):
            return 'comparisons'
        elif any(w in words for w in ['lyric', 'lyrics', 'theme', 'themes', 'story', 'narrative']):
            return 'lyrical_analysis'
        # career_evolution before musical_analysis: 'era/evolution/album' signal
        # temporal scope, which is more specific than generic audio feature keywords
        elif any(w in words for w in ['era', 'eras', 'evolution', 'career', 'album',
                                       'albums', 'discography']):
            return 'career_evolution'
        elif any(w in words for w in ['sound', 'production', 'tempo', 'energy', 'acoustic',
                                       'danceability', 'valence', 'bpm', 'key', 'loudness']):
            return 'musical_analysis'
        else:
            return 'general'

    def _assess_importance(self, question: str, answer: str, topic: str) -> float:
        """
        Assess interaction importance for memory consolidation.

        topic is passed in explicitly so importance scoring uses the already-
        classified topic rather than re-deriving it with a different ordering.
        """
        importance = 0.5

        if len(answer) > 500:
            importance += 0.2

        if any(word in question.lower() for word in ['why', 'how', 'analyze', 'explain']):
            importance += 0.2

        # Follow-up on same topic signals sustained engagement
        if len(self.short_term_memory) > 0:
            last_topic = self.short_term_memory[-1].topic
            if last_topic == topic:
                importance += 0.1

        return min(1.0, importance)

    def _consolidate_memories(self):
        """
        Memory consolidation: identify related memories and forget low-importance ones.

        The LLM-based merge is intentionally left as a stub with clear logging
        so the consolidation threshold has a visible effect even before full
        merge logic is implemented.
        """
        print(f"\n[MEMORY CONSOLIDATION] Processing {len(self.short_term_memory)} recent memories...")

        if len(self.short_term_memory) < 3:
            return

        consolidation_prompt = "Analyze these recent interactions and identify if any should be consolidated:\n\n"
        for i, mem in enumerate(self.short_term_memory[-5:], 1):
            consolidation_prompt += f"\n{i}. Q: {mem.question}\n   A: {mem.answer[:200]}...\n"

        consolidation_prompt += """
Should any of these be consolidated (merged) because they cover the same topic?
If yes, respond with: CONSOLIDATE: [list of numbers to merge]
If no, respond with: NO_CONSOLIDATION
"""

        if config.USE_OPENAI:
            response = self.client.chat_interactive(consolidation_prompt)
        else:
            response = self.client.generate(consolidation_prompt, max_tokens=200)

        if "CONSOLIDATE:" in response:
            print("  Consolidating related memories...")
            for mem in self.short_term_memory:
                if mem.access_count > 0 or mem.importance > 0.7:
                    print(f"    Promoting: {mem.topic}")

        # Forget low-importance, unaccessed memories
        before = len(self.short_term_memory)
        self.short_term_memory = [
            m for m in self.short_term_memory
            if not (m.importance < 0.3 and m.access_count == 0)
        ]
        forgotten = before - len(self.short_term_memory)
        if forgotten > 0:
            print(f"  Forgot {forgotten} low-importance memories")

    def _proactive_memory_check(self, question: str) -> Optional[str]:
        """
        Proactively surface past memories relevant to the current question.
        """
        relevant = self._retrieve_relevant_memories(question, k=2)

        if not relevant or relevant[0][1] < 0.3:
            return None

        memory, score = relevant[0]

        if score > 0.5 and memory.topic in self.user_profile.preferred_topics:
            note = (
                f"\nNote: This relates to something we discussed before "
                f"({memory.timestamp[:10]}). "
                f"You asked: '{memory.question[:100]}'. "
                f"Should I incorporate that context?"
            )
            return note

        return None

    def _detect_insight_opportunity(self) -> Optional[str]:
        """
        Synthesize patterns across memories to surface non-obvious insights.
        """
        if len(self.long_term_memory) < 5:
            return None

        recent_topics = [m.topic for m in self.long_term_memory[-10:]]
        topic_counts: Dict[str, int] = {}
        for topic in recent_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        if topic_counts and max(topic_counts.values()) >= 4:
            dominant_topic = max(topic_counts, key=topic_counts.get)

            insight_prompt = (
                f"The user has asked {topic_counts[dominant_topic]} questions "
                f"about {dominant_topic} recently.\n\n"
                f"Recent questions on this topic:\n"
            )

            relevant_memories = [
                m for m in self.long_term_memory[-10:] if m.topic == dominant_topic
            ]
            for mem in relevant_memories[:3]:
                insight_prompt += f"- {mem.question}\n"

            insight_prompt += (
                "\nBased on these questions, what pattern or deeper interest might they have? "
                "Suggest one insightful observation or recommendation (2-3 sentences)."
            )

            if config.USE_OPENAI:
                insight = self.client.chat_interactive(insight_prompt)
            else:
                insight = self.client.generate(insight_prompt, max_tokens=200)

            return f"\nPattern detected: {insight}"

        return None

    def ask(self, question: str, use_memory: bool = True) -> str:
        """
        Answer with autonomous memory capabilities.
        Restricted to dataset only - no external knowledge.

        Workflow:
        1. Proactive memory check
        2. Retrieve semantically similar memories
        3. Build data-grounded system prompt
        4. Generate answer
        5. Store with metadata
        6. Update user profile
        7. Check for consolidation
        8. Detect insight opportunities
        """
        print(f"\n{'='*80}")
        print("AUTONOMOUS MEMORY AGENT")
        print(f"{'='*80}")

        # STEP 1: Proactive memory check
        proactive_note = None
        if use_memory:
            proactive_note = self._proactive_memory_check(question)
            if proactive_note:
                print(proactive_note)

        # STEP 2: Retrieve relevant memories
        context = ""
        if use_memory:
            relevant_memories = self._retrieve_relevant_memories(question, k=3)

            if relevant_memories and relevant_memories[0][1] > 0.2:
                print(f"\n[MEMORY RETRIEVAL] Found {len(relevant_memories)} relevant memories")
                context = "Previous relevant context:\n\n"

                for i, (memory, score) in enumerate(relevant_memories, 1):
                    print(f"  {i}. {memory.topic} (similarity: {score:.2f}, accessed: {memory.access_count} times)")
                    context += f"{i}. You asked: {memory.question}\n"
                    context += f"   Answer summary: {memory.answer[:150]}...\n\n"

                context += "Current question:\n"

        # STEP 3: Build data-grounded system prompt
        system_prompt = f"""You are a music analysis assistant specializing in this discography dataset.

CRITICAL RESTRICTIONS:
1. Answer ONLY based on the dataset provided below
2. Do not use pre-trained knowledge about the artist
3. If the question cannot be answered from the data, say: "I don't have data to answer that."
4. Never mention biographical information, tour dates, or personal life not in the dataset
5. Never mention awards, sales figures, or external achievements

{self.data_context}

WHAT YOU CAN ANSWER:
- Questions about songs in the dataset (audio features, themes, eras)
- Comparisons between songs, albums, or eras using the data
- Statistical analysis of the dataset
- Patterns in musical features or themes
"""

        if self.user_profile.interaction_style == "analytical":
            system_prompt += "\nThe user prefers analytical, detailed responses with specific metrics."
        elif self.user_profile.interaction_style == "exploratory":
            system_prompt += "\nThe user enjoys discovering new patterns and connections in the data."

        # STEP 4: Generate answer
        full_prompt = context + question

        if config.USE_OPENAI:
            answer = self.client.chat_interactive(full_prompt, system_prompt=system_prompt)
        else:
            full_prompt = system_prompt + "\n\n" + full_prompt
            answer = self.client.generate(full_prompt, max_tokens=500)

        if "don't have data" in answer.lower() or "cannot answer" in answer.lower():
            print("[DATA RESTRICTION] Agent correctly refused out-of-scope question")

        # STEP 5: Classify topic first, then assess importance using it.
        # Ordering matters: importance scoring uses topic, so topic must
        # be derived before _assess_importance is called.
        topic = self._classify_topic(question, answer)
        importance = self._assess_importance(question, answer, topic)
        embedding = self._create_embedding(question + " " + answer)

        memory = Memory(
            question=question,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            topic=topic,
            importance=importance,
            embedding=embedding
        )

        self.short_term_memory.append(memory)
        print(f"\n[MEMORY] Stored as: {topic} (importance: {importance:.2f})")

        # STEP 6: Update user profile
        self.user_profile.update(question, topic)

        # STEP 7: Consolidation check
        if len(self.short_term_memory) >= self.consolidation_threshold:
            self._consolidate_memories()

        # STEP 8: Insight opportunity (every 3 interactions)
        if len(self.short_term_memory) % 3 == 0:
            insight = self._detect_insight_opportunity()
            if insight:
                answer += "\n\n" + insight

        return answer

    def save_session(self):
        """Save current session to long-term memory."""
        if self.short_term_memory:
            self.long_term_memory.extend(self.short_term_memory)
            self._save_memory()
            print(f"\nSaved {len(self.short_term_memory)} interactions to long-term memory")
            self.short_term_memory = []

    def get_session_summary(self) -> str:
        """Generate intelligent session summary."""
        if not self.short_term_memory:
            return "No interactions in current session."

        summary_prompt = "Summarize this conversation session intelligently:\n\nInteractions:\n"

        for i, mem in enumerate(self.short_term_memory, 1):
            summary_prompt += f"\n{i}. Topic: {mem.topic}\n"
            summary_prompt += f"   Q: {mem.question}\n"
            summary_prompt += f"   A: {mem.answer[:100]}...\n"

        summary_prompt += """
Provide:
1. Main topics discussed (1 sentence)
2. Key insights or patterns (2-3 points)
3. Suggested follow-up questions the user might find interesting (2 questions)
"""

        if config.USE_OPENAI:
            return self.client.chat_interactive(summary_prompt)
        else:
            return self.client.generate(summary_prompt, max_tokens=400)

    def suggest_next_questions(self) -> List[str]:
        """Proactively suggest next questions based on conversation history."""
        if not self.short_term_memory and not self.long_term_memory:
            return [
                "What makes the songwriting in this catalog distinctive?",
                "How has musical style evolved across eras?",
                "Which songs have the highest energy and valence combination?"
            ]

        recent_topics = [m.topic for m in (self.short_term_memory + self.long_term_memory[-5:])]

        suggestion_prompt = (
            f"Based on these recent conversation topics: {', '.join(set(recent_topics))}\n\n"
            f"And the user's interaction style: {self.user_profile.interaction_style}\n\n"
            "Suggest 3 interesting follow-up questions they might want to explore.\n"
            "Make them specific and answerable from audio feature and lyrical data.\n\n"
            "Format: One question per line."
        )

        if config.USE_OPENAI:
            response = self.client.chat_interactive(suggestion_prompt)
        else:
            response = self.client.generate(suggestion_prompt, max_tokens=200)

        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and ('?' in line or line.startswith(('-', '1.', '2.', '3.'))):
                line = line.lstrip('- 123.').strip()
                if line:
                    questions.append(line)

        return questions[:3]

    def analyze_memory_patterns(self) -> str:
        """Meta-analysis of conversation history to surface user interest patterns."""
        all_memories = self.long_term_memory + self.short_term_memory

        if len(all_memories) < 5:
            return "Not enough conversation history for pattern analysis yet."

        analysis_prompt = (
            f"Analyze patterns in this user's conversation history:\n\n"
            f"Total interactions: {len(all_memories)}\n"
            f"User profile: {self.user_profile.get_summary()}\n\n"
            "Topic distribution:\n"
        )

        topic_counts: Dict[str, int] = {}
        for mem in all_memories:
            topic_counts[mem.topic] = topic_counts.get(mem.topic, 0) + 1

        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            analysis_prompt += f"  {topic}: {count} ({count/len(all_memories)*100:.1f}%)\n"

        analysis_prompt += "\nSample questions by topic:\n"
        for topic in list(topic_counts.keys())[:3]:
            topic_memories = [m for m in all_memories if m.topic == topic]
            if topic_memories:
                analysis_prompt += f"  {topic}: {topic_memories[0].question}\n"

        analysis_prompt += """
Provide:
1. What this reveals about the user's interests (2-3 sentences)
2. What aspects of the catalog they are most curious about
3. Suggested areas they have not explored yet that might interest them
"""

        if config.USE_OPENAI:
            return self.client.chat_interactive(analysis_prompt)
        else:
            return self.client.generate(analysis_prompt, max_tokens=500)

    def clear_memory(self, clear_long_term: bool = False, clear_profile: bool = False):
        """Clear memory with options to preserve important entries."""
        if clear_long_term:
            important = [
                m for m in self.long_term_memory
                if m.importance > 0.7 or m.access_count > 2
            ]

            if important and not clear_profile:
                print(f"\n{len(important)} important memories found, preserving them.")
                self.long_term_memory = important
                print(f"Kept {len(important)} important memories")
            else:
                self.long_term_memory = []
                print("All long-term memories cleared")

            self.short_term_memory = []

            if clear_profile:
                self.user_profile = UserProfile()
                print("User profile reset")

            self._save_memory()
            print("Changes saved to disk")
        else:
            self.short_term_memory = []
            print("Session memory cleared (long-term memories preserved)")


def interactive_autonomous_memory():
    """Interactive session with autonomous memory agent."""
    from src.data_loading import load_and_merge_data
    from src.era_analysis import define_eras

    print("="*80)
    print("AUTONOMOUS MEMORY AGENT")
    print("="*80)
    print(f"Model: {config.MODEL}\n")

    print("Loading data...")
    merged_df = load_and_merge_data(config.DATA_DIR, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    df = define_eras(merged_df)

    print("Initializing agent...")
    agent = AutonomousMemoryAgent(df=df)

    print("\nAgent ready!")
    print("\nCommands:")
    print("  <question>       - Ask anything (answered from dataset only)")
    print("  summary          - Session summary")
    print("  suggest          - Suggested follow-up questions")
    print("  profile          - Learned user profile")
    print("  analyze          - Conversation pattern analysis")
    print("  data             - Show dataset context")
    print("  save             - Save session to long-term memory")
    print("  clear            - Clear session only")
    print("  clear long       - Clear all saved memories (keeps important ones)")
    print("  clear all        - Clear everything including profile")
    print("  quit             - Exit (auto-saves)")
    print()

    try:
        while True:
            question = input("\nYou: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nSaving session...")
                agent.save_session()
                print("Goodbye!")
                break

            if question.lower() == 'summary':
                print("\n" + "="*80)
                print("SESSION SUMMARY")
                print("="*80)
                print(agent.get_session_summary())
                continue

            if question.lower() == 'suggest':
                print("\nSuggested questions based on your interests:")
                suggestions = agent.suggest_next_questions()
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"{i}. {suggestion}")
                continue

            if question.lower() == 'profile':
                print("\n" + "="*80)
                print("USER PROFILE")
                print("="*80)
                print(agent.user_profile.get_summary())
                continue

            if question.lower() == 'analyze':
                print("\n" + "="*80)
                print("CONVERSATION PATTERN ANALYSIS")
                print("="*80)
                print(agent.analyze_memory_patterns())
                continue

            if question.lower() == 'data':
                print("\n" + "="*80)
                print("AVAILABLE DATA")
                print("="*80)
                print(agent.data_context)
                continue

            if question.lower() == 'save':
                agent.save_session()
                continue

            if question.lower().startswith('clear'):
                if 'all' in question.lower():
                    agent.clear_memory(clear_long_term=True, clear_profile=True)
                elif 'long' in question.lower():
                    agent.clear_memory(clear_long_term=True, clear_profile=False)
                else:
                    agent.clear_memory(clear_long_term=False, clear_profile=False)
                continue

            if not question:
                continue

            answer = agent.ask(question)
            print(f"\n{'-'*80}")
            print("Agent:")
            print(f"{'-'*80}")
            print(answer)

            if len(agent.short_term_memory) % 4 == 0 and len(agent.short_term_memory) > 0:
                print("\nYou might also want to explore:")
                suggestions = agent.suggest_next_questions()
                for suggestion in suggestions[:2]:
                    print(f"  - {suggestion}")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving session...")
        agent.save_session()
        print("Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    interactive_autonomous_memory()
