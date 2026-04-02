"""Conversational agent with Chain-of-Thought reasoning and dynamic data retrieval."""
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Any

from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from src import config


class DataDictionary:
    """Defines all available data columns and their meanings."""

    COLUMNS = {
        'Song_Name': 'Name of the song',
        'Album': 'Album name the song belongs to',
        'album_clean': 'Cleaned album name',
        'era': 'Musical era (Country, Transition, Pop, Indie, Pop Revival)',
        'Release_Date': 'Date when the song was released',
        'danceability': 'How suitable the song is for dancing (0=not danceable, 1=very danceable)',
        'energy': 'Perceptual measure of intensity and activity (0=calm, 1=energetic)',
        'valence': 'Musical positivity/happiness (0=sad/negative, 1=happy/positive)',
        'acousticness': 'Confidence the track is acoustic (0=not acoustic, 1=acoustic)',
        'instrumentalness': 'Predicts whether track has no vocals (0=vocals, 1=instrumental)',
        'liveness': 'Detects presence of audience in recording (0=studio, 1=live)',
        'speechiness': 'Detects presence of spoken words (0=music, 1=speech)',
        'loudness': 'Overall loudness in decibels (typically -60 to 0)',
        'tempo': 'Estimated tempo in beats per minute (BPM)',
        'duration_ms': 'Duration of the song in milliseconds',
        'key': 'Musical key the song is in (0-11, corresponds to pitch classes)',
        'mode': 'Modality of track (0=minor, 1=major)',
        'time_signature': 'Time signature (beats per measure)',
        'polarity': 'Sentiment polarity from lyrics (-1=negative, 1=positive)',
        'subjectivity': 'Subjectivity of lyrics (0=objective, 1=subjective)',
        'dominant_topic': 'BERTopic cluster ID (0=core narrative style, other=stylistic departure)',
        'topic_weight': 'Strength of the dominant topic assignment',
    }


class AnalysisAssistant:
    """
    Conversational analysis agent with Chain-of-Thought reasoning
    and dynamic data retrieval.

    Mirrors the recommendation agent's data loading pattern:
    loads directly from source files via load_and_merge_data + define_eras,
    no dependency on pre-saved CSV files.

    Conversation history is owned entirely by the LLM client.
    AnalysisAssistant never maintains a parallel history list.
    Data responses are injected as user-role messages with a clear label
    so the model understands their provenance.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None,
                 model: str = config.MODEL):
        if config.USE_OPENAI:
            self.client = OpenAIClient()
        else:
            self.client = OllamaClient(model=model)

        # Accept a pre-loaded dataframe (e.g. from Streamlit session state)
        # or load directly from source files if not provided.
        if df is not None:
            self.df = df
        else:
            self.df = self._load_data()

        self.system_prompt = self._build_system_prompt()
        print("Analysis assistant initialised with CoT and dynamic data retrieval")

    def _load_data(self) -> Optional[pd.DataFrame]:
        """
        Load data directly from source files.
        Mirrors recommendation agent: load_and_merge_data + define_eras.
        No dependency on pre-saved CSV files.
        """
        from src.data_loading import load_and_merge_data
        from src.era_analysis import define_eras

        try:
            merged_df = load_and_merge_data(
                config.DATA_DIR,
                config.SPOTIFY_CSV,
                config.ALBUM_SONG_CSV
            )
            df = define_eras(merged_df)
            print(f"Loaded dataset: {len(df)} rows, {df['era'].nunique()} eras")
            return df
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
            return None

    def _build_system_prompt(self) -> str:
        """Build system prompt with dataset context and reasoning instructions."""
        if self.df is not None:
            era_counts = self.df['era'].value_counts().to_dict()
            era_summary = " | ".join(f"{era}: {count}" for era, count in era_counts.items())
            dataset_info = (
                f"Dataset: {len(self.df)} songs across {self.df['era'].nunique()} eras\n"
                f"Era distribution: {era_summary}\n"
                f"Albums: {self.df['Album'].nunique()} total\n"
            )
        else:
            dataset_info = "Dataset: not loaded\n"

        columns_text = "\n".join(
            f"  {col}: {desc}" for col, desc in DataDictionary.COLUMNS.items()
        )

        return f"""You are an analytical assistant for a music dataset.

DATASET OVERVIEW:
{dataset_info}
AVAILABLE COLUMNS:
{columns_text}

CRITICAL CONSTRAINTS:
- Base ALL claims on data returned by DATA_REQUEST. Never invent statistics.
- Do NOT use pre-trained knowledge about the artist or songs.
- Do NOT reference lyrics, music videos, tours, interviews, or personal life.
- Do NOT reference anything outside the dataset columns listed above.
- If a question cannot be answered from the dataset, say so explicitly.

CHAIN-OF-THOUGHT REASONING PROTOCOL:
For any analytical question follow these steps:

1. UNDERSTAND: Rephrase what is answerable from the dataset only.
2. PLAN: List which columns and aggregations are needed.
3. REQUEST DATA: Issue exactly one DATA_REQUEST per iteration:

   DATA_REQUEST: {{"dataset": "songs_with_topics", "columns": ["col1", "col2"], "filters": {{"column": "value"}}, "aggregation": "describe aggregation"}}

   Valid aggregations: "mean grouped by era", "count grouped by album",
   "summary statistics", "max grouped by era", "min grouped by era"

4. ANALYZE: Once data is provided, cite specific numbers from it.
5. ANSWER: Provide a clear, evidence-based response.

RULES:
- Issue one DATA_REQUEST per iteration, then wait for data.
- Request ALL columns you need in a single DATA_REQUEST — do not request one column at a time.
- If after receiving data you realise you need additional columns, issue another DATA_REQUEST.
- NEVER invent or assume data values. If you do not have data for a claim, request it.
- If data is marked "(Assuming hypothetical...)" or similar, that is a hallucination — stop and request real data instead.
- Always cite specific values from retrieved data in your answer.
- Use concise analytical language.
- NEVER write "(Assuming hypothetical...)" or any variation.
  If you lack data, issue a DATA_REQUEST. Assumed values are a critical failure.
"""

    def _get_basic_context(self) -> str:
        """Compact dataset summary injected with every user message."""
        if self.df is None:
            return "Dataset not loaded.\n"

        context = f"Dataset: {len(self.df)} songs\n"
        context += f"Eras: {', '.join(self.df['era'].unique())}\n"
        context += f"Albums: {self.df['Album'].nunique()} total\n"

        numeric_cols = ['energy', 'valence', 'acousticness', 'tempo']
        available = [c for c in numeric_cols if c in self.df.columns]
        if available:
            means = self.df[available].mean()
            context += "Dataset averages: " + " | ".join(
                f"{c}={means[c]:.2f}" for c in available
            ) + "\n"
        return context

    def _extract_data_request(self, response: str) -> Optional[Dict]:
        """Extract DATA_REQUEST JSON from agent response."""
        if "DATA_REQUEST:" not in response:
            return None

        try:
            start = response.find("DATA_REQUEST:") + len("DATA_REQUEST:")
            json_start = response.find("{", start)
            if json_start == -1:
                return None

            brace_count = 0
            json_end = json_start
            for i, char in enumerate(response[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            json_str = response[json_start:json_end].replace('\n', ' ')
            # Only strip backslashes not part of valid escape sequences.
            json_str = re.sub(r'\\(?!["\\\/bfnrtu])', '', json_str)
            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parse failed: {e}, trying manual parse")
            return self._manual_parse_data_request(response)

    def _manual_parse_data_request(self, response: str) -> Optional[Dict]:
        """Fallback manual parser for malformed DATA_REQUEST JSON."""
        try:
            columns_match = re.search(r'"columns"\s*:\s*\[([^\]]+)\]', response)
            columns = []
            if columns_match:
                columns = [
                    c.strip().strip('"\'')
                    for c in columns_match.group(1).split(',')
                ]

            filters = {}
            filters_match = re.search(r'"filters"\s*:\s*\{([^\}]+)\}', response)
            if filters_match:
                for pair in filters_match.group(1).split(','):
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        filters[k.strip().strip('"')] = v.strip().strip('"')

            agg_match = re.search(r'"aggregation"\s*:\s*"([^"]+)"', response)
            aggregation = agg_match.group(1) if agg_match else ""

            result = {
                "dataset": "songs_with_topics",
                "columns": columns,
                "filters": filters,
                "aggregation": aggregation
            }
            print(f"[MANUAL PARSE] {result}")
            return result
        except Exception as e:
            print(f"Manual parse also failed: {e}")
            return None

    def _fulfill_data_request(self, request: Dict) -> str:
        """
        Execute DATA_REQUEST against the dataframe using pandas operations.
        Dataset name from LLM is ignored — there is only one dataset.
        """
        if self.df is None:
            return "ERROR: Dataset not loaded."

        columns = request.get('columns', [])
        filters = request.get('filters', {})
        aggregation = request.get('aggregation', '')

        df = self.df.copy()

        # Apply filters.
        for col, value in filters.items():
            if col not in df.columns:
                print(f"[WARNING] Column '{col}' not found, skipping filter")
                continue

            # Handle list values for IN queries (multiple songs).
            if isinstance(value, list):
                mask = df[col].astype(str).str.lower().isin(
                    [v.lower() for v in value]
                )
                if not mask.any():
                    # Fallback: partial match for any item in list.
                    mask = df[col].astype(str).apply(
                        lambda x: any(
                            v.lower() in x.lower() for v in value
                        )
                    )
            else:
                mask = df[col].astype(str).str.lower() == str(value).lower()
                if not mask.any():
                    mask = df[col].astype(str).str.contains(
                        str(value), case=False, na=False
                    )

            df = df[mask]
            print(f"[FILTER] {col}='{value}' -> {len(df)} rows")

        if len(df) == 0:
            return f"ERROR: No data matches filters: {filters}"

        # Validate and select columns.
        available = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"[WARNING] Columns not found: {missing}")
        if not available:
            # Fall back to key columns.
            available = [
                c for c in ['Song_Name', 'Album', 'era', 'energy', 'valence']
                if c in df.columns
            ]

        df = df[available]

        # Apply aggregation.
        if aggregation:
            agg_lower = aggregation.lower()

            if 'grouped by' in agg_lower or 'group by' in agg_lower:
                group_col = next(
                    (c for c in available if c.lower() in agg_lower), None
                )
                if group_col:
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        if 'mean' in agg_lower or 'average' in agg_lower:
                            result = df.groupby(group_col)[numeric_cols].mean()
                        elif 'sum' in agg_lower or 'total' in agg_lower:
                            result = df.groupby(group_col)[numeric_cols].sum()
                        elif 'count' in agg_lower:
                            result = df.groupby(group_col).size().to_frame('count')
                        elif 'max' in agg_lower:
                            result = df.groupby(group_col)[numeric_cols].max()
                        elif 'min' in agg_lower:
                            result = df.groupby(group_col)[numeric_cols].min()
                        else:
                            result = df.groupby(group_col)[numeric_cols].mean()
                        return f"DATA RETRIEVED (grouped):\n{result.to_string()}\n"

            elif any(w in agg_lower for w in ['summary', 'statistics', 'describe']):
                return f"DATA RETRIEVED (summary):\n{df.describe().to_string()}\n"

        # Return raw rows.
        if len(df) <= 20:
            return f"DATA RETRIEVED ({len(df)} rows):\n{df.to_string(index=False)}\n"
        else:
            return (
                f"DATA RETRIEVED ({len(df)} total rows):\n\n"
                f"SUMMARY:\n{df.describe().to_string()}\n\n"
                f"SAMPLE (20 rows):\n{df.sample(20).to_string(index=False)}\n"
            )

    def ask(self, question: str, max_iterations: int = 10) -> str:
        """
        Answer a question using CoT reasoning and dynamic data retrieval.

        Conversation history is owned by the LLM client, not by this class.
        Each call to client.chat_interactive appends to the client's internal
        history automatically. Data responses are injected as user-role messages
        with a clear label so the model knows their provenance.
        """
        full_question = (
            f"{question}\n\n"
            f"[CONTEXT]\n{self._get_basic_context()}"
        )

        iteration = 0
        current_message = full_question

        while iteration < max_iterations:
            response = self.client.chat_interactive(
                user_message=current_message,
                system_prompt=self.system_prompt if iteration == 0 else None,
                max_tokens=1000
            )

            data_request = self._extract_data_request(response)

            if data_request:
                print(f"[DATA REQUEST] columns={data_request.get('columns')} "
                      f"filters={data_request.get('filters')} "
                      f"agg={data_request.get('aggregation')}")
                data_response = self._fulfill_data_request(data_request)
                print(f"[DATA RESPONSE] {data_response[:100]}...")

                # Inject data back as the next user message.
                # Clear label so the model understands this is retrieved data,
                # not a new question from the user.
                current_message = (
                    f"[DATA RETRIEVED - continue your analysis using this data]\n\n"
                    f"{data_response}\n\n"
                    f"Now complete steps ANALYZE and ANSWER from your plan."
                )
                iteration += 1
            else:
                # No data request — model has produced its final answer.
                return response

        return response + "\n\n[Reached maximum iteration limit]"

    def reset(self):
        """Reset conversation history on both this agent and the LLM client."""
        if hasattr(self.client, 'reset_conversation'):
            self.client.reset_conversation()
        print("Conversation history cleared")

    def suggest_questions(self) -> str:
        """Suggest interesting analytical questions based on dataset structure."""
        self.client.reset_conversation()
        prompt = (
            "Based on the available music dataset columns "
            "(audio features: energy, valence, acousticness, tempo, danceability; "
            "lyric features: polarity, subjectivity; "
            "metadata: era, Album, dominant_topic), "
            "suggest 4 interesting analytical questions that would reveal insights "
            "about musical evolution or patterns. "
            "Format as a numbered list. Be specific about which columns to use."
        )
        return self.client.chat_interactive(
            user_message=prompt,
            system_prompt=self.system_prompt
        )

    #def suggest_insights(self) -> str:
    #    """Generate data-driven insights via the full CoT ask loop."""
    #    self.client.reset_conversation()
    #    return self.ask(
    #        "Analyse the dataset and generate 3 specific, data-driven insights. "
    #        "Each insight must include specific numbers retrieved from the data. "
    #        "Focus on patterns across eras or audio feature distributions."
    #    )

    def suggest_insights(self) -> str:
        """Generate data-driven insights via the full CoT ask loop."""
        self.client.reset_conversation()
        return self.ask(
            "Analyse the dataset and generate 3 specific, data-driven insights "
            "about musical evolution across eras. "
            "You MUST request ALL of these columns in a single DATA_REQUEST: "
            "era, energy, valence, acousticness, tempo, polarity. "
            "Use aggregation 'mean grouped by era'. "
            "Do not proceed to ANSWER until you have received real data for all columns. "
            "Never assume or invent values — if you do not have data for a column, "
            "issue another DATA_REQUEST before making any claim about it."
        )


def interactive_session():
    """Start an interactive session with the analysis assistant."""
    from src.data_loading import load_and_merge_data
    from src.era_analysis import define_eras

    print("=" * 80)
    print("ANALYSIS ASSISTANT")
    print("Chain-of-Thought Reasoning + Dynamic Data Retrieval")
    print("=" * 80)
    print(f"Model: {config.MODEL}\n")

    print("Loading data...")
    try:
        merged_df = load_and_merge_data(
            config.DATA_DIR,
            config.SPOTIFY_CSV,
            config.ALBUM_SONG_CSV
        )
        df = define_eras(merged_df)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Initialising agent...")
    try:
        agent = AnalysisAssistant(df=df)
    except Exception as e:
        print(f"Error initialising agent: {e}")
        return

    print("\nAgent ready!")
    print("\nCommands:")
    print("  - Type your question")
    print("  - 'insights' for data-driven insights")
    print("  - 'questions' for sample questions")
    print("  - 'columns' to see available data columns")
    print("  - 'reset' to clear conversation history")
    print("  - 'quit' to exit")
    print()

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not question:
                continue

            if question.lower() == 'reset':
                agent.reset()
                continue

            if question.lower() == 'columns':
                print("\nAVAILABLE COLUMNS:")
                for col, desc in DataDictionary.COLUMNS.items():
                    print(f"  {col}: {desc}")
                continue

            if question.lower() == 'insights':
                print("\nGenerating insights...\n")
                response = agent.suggest_insights()
            elif question.lower() == 'questions':
                print("\nGenerating sample questions...\n")
                response = agent.suggest_questions()
            else:
                response = agent.ask(question)

            print(f"\nAssistant:\n{response}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    interactive_session()
