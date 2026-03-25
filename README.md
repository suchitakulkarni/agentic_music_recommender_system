# Music Intelligence Multi-Agent System

A modular multi-agent framework for music analysis and recommendation, built on top of structured audio and lyrical datasets. Five architecturally distinct agents demonstrate different patterns in agentic LLM design: preference learning, tool use, chain-of-thought reasoning, semantic memory, and multi-agent orchestration with inter-agent debate.

The demo dataset is Taylor Swift's discography (Spotify audio features + lyrical sentiment + topic models). The agents are dataset-agnostic — any structured music corpus with equivalent features can be substituted.

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Orchestrator                     │
│         Dynamic assembly · Inter-agent debate · Weighted        │
│                     confidence synthesis                        │
└────────────┬──────────────┬──────────────┬──────────────────────┘
             │              │              │
     ┌───────▼──────┐ ┌─────▼──────┐ ┌────▼──────────┐
     │   Lyrical    │ │  Musical   │ │  Contextual   │
     │   Analyst    │ │  Analyst   │ │   Analyst     │
     └──────────────┘ └────────────┘ └───────────────┘

┌──────────────────────┐   ┌──────────────────────┐
│  Recommendation      │   │   Memory Agent        │
│  Agent               │   │                       │
│  · Preference model  │   │  · Semantic retrieval │
│  · Explore/exploit   │   │  · Consolidation      │
│  · Active feedback   │   │  · User profiling     │
└──────────────────────┘   └──────────────────────┘

┌──────────────────────┐   ┌──────────────────────┐
│  Tool Agent          │   │  Analysis Assistant   │
│                      │   │                       │
│  · Tool registration │   │  · CoT reasoning      │
│  · Self-correction   │   │  · Dynamic data       │
│  · Chained execution │   │    retrieval          │
└──────────────────────┘   └──────────────────────┘
```

### Agent Descriptions

| Agent | Pattern | Key Design Features |
|---|---|---|
| `RecommendationAgent` | Preference learning + explore/exploit | Learns user preference model across sessions; balances exploitation of known preferences with era-diverse exploration; feedback triggers model updates |
| `MemoryAgent` | Semantic memory with consolidation | Embedding-based retrieval across sessions; importance scoring; memory consolidation; user profile inference from interaction patterns |
| `ToolAgent` | Dynamic tool use with self-correction | LLM-driven tool selection and execution planning; retry logic with autonomous argument correction on failure; success-rate tracking per tool |
| `AnalysisAssistant` | Chain-of-thought + dynamic data retrieval | Structured CoT protocol; agent requests specific data slices mid-reasoning; pandas-backed aggregation layer |
| `MultiAgentOrchestrator` | Multi-agent debate + confidence routing | Dynamic specialist assembly per question; inter-agent contradiction detection and debate; confidence-weighted synthesis |

---

## Repository Structure

```
music_intelligence_agents/
│
├── README.md
├── main.py
├── demo_agents.py
├── requirements.txt
│
├── eval/
│   ├── eval_recommendation_agent.py
│   └── eval_memory_agent.py
│
├── results/                        # eval CSVs land here
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── era_analysis.py
│   ├── similarity_analysis.py      # fix the double-encoding bug before push
│   ├── topic_modeling.py
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── openai_client.py
│   └── ollama_client.py
│
└── src/agents/
    ├── __init__.py
    ├── recommendation_agent.py     # fixed - use output version
    ├── memory_agent.py             # fixed - use output version
    └── experimental/
        ├── tool_agent.py
        ├── analysis_assistant.py
        └── multi_agent_system.py
```


## Evaluation Results

### Recommendation Agent (no LLM required)
| Test | Result |
|---|---|
| Self-exclusion (input song never recommended to itself) | 100% (5/5) |
| Recommendations closer than random baseline | 80% (4/5) |
| Neighborhood feature homogeneity ratio | 0.66 |
| Exploration era diversity | 100% (5/5) |

Mean feature distance improvement over random: ~0.10

The 1 failure in feature distance (song: *august*) reflects a lyric-audio
mismatch — its lyrical register matches folklore/evermore but its production
is more energetic. Fixed-weight hybrid similarity resolves ambiguity toward
lyric similarity in these cases.

### Memory Agent (no LLM required)
| Test | Result |
|---|---|
| Topic classification accuracy | 100% (5/5) |
| Retrieval precision@3 | 75% (3/4) |

The 1 retrieval failure reflects a known bag-of-words limitation: lexically
distant but semantically related queries (e.g. "albums" vs "eras") score low
despite conceptual overlap. Sentence-transformer upgrade is the documented
next step.

---

## Technical Stack

- **LLM backends**: OpenAI API or local Ollama (switchable via config)
- **Data**: Spotify audio features, lyrical sentiment (TextBlob), topic models (BERTopic/LDA)
- **Similarity**: Hybrid cosine similarity over audio feature vectors and lyrical embeddings
- **Memory persistence**: JSON-backed long-term memory with numpy embedding storage
- **Dependencies**: `pandas`, `numpy`, `scikit-learn`, `bertopic`, `openai`, `ollama`

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/suchitakulkarni/DataScience.git
cd DataScience/music_intelligence_agents
pip install -r requirements.txt
```

### 2. Configure backend

```bash
# For OpenAI
export OPENAI_API_KEY="your_api_key_here"

# For local Ollama (no API key needed)
ollama pull mistral   # or whichever model is set in config.py
```

Set `USE_OPENAI = True/False` in `src/config.py`.

### 3. Run a demo

```bash
# All agents interactive demo
python demo_agents.py

# Individual agents
python -m src.agents.recommendation_agent
python -m src.agents.memory_agent
python -m src.agents.analysis_assistant
python -m src.agents.tool_agent
python -m src.agents.multi_agent_system
```

---

## Agent Interaction Examples

**Recommendation with preference learning**
```
rec All Too Well
> Recommendations: [Red, The Last Time, Come Back Be Here, ...]
> Explanation: High valence/low energy profile, folklore-era acoustic signature...

feedback too acoustic
> Preference model updated: acousticness ceiling lowered
> Refined recommendations: [...]
```

**Multi-agent analysis**
```
analyze Blank Space
> [Lyrical Analyst] HIGH confidence: narrative framing, self-aware irony...
> [Musical Analyst] MEDIUM confidence: high energy (0.73), major key, 96 BPM...
> [Contextual Analyst] HIGH confidence: 1989 era transition toward pop maximalism...
> [DEBATE] Lyrical vs Musical: valence/sentiment tension resolved...
> [SYNTHESIS] Weighted final answer...
```

**Chain-of-thought analysis**
```
You: How has emotional tone shifted from early to recent eras?

UNDERSTAND: Comparing sentiment across temporal eras
PLAN: Need valence, polarity, energy grouped by era
DATA_REQUEST: {"dataset": "songs_with_topics", "columns": ["era", "valence", "polarity"], "aggregation": "mean grouped by era"}
ANALYZE: debut/fearless avg valence 0.61 → folklore/evermore 0.38...
```

---

## Design Decisions

**Why five separate agents rather than one monolithic agent?**
Each agent isolates a distinct agentic pattern. This makes it straightforward to benchmark them against each other and swap components — e.g. replacing the bag-of-words memory embeddings with sentence-transformers without touching the recommendation logic.

**Why support both OpenAI and Ollama?**
Cost and reproducibility. Local Ollama runs allow full offline development and deterministic evals without API spend. The backend abstraction is a single config flag.

**Why a structured music dataset as the demo domain?**
Audio feature vectors (energy, valence, tempo, acousticness) provide a well-defined, numerically grounded space that makes agent reasoning auditable. When an agent claims "this recommendation matches your energy preference," the claim is directly verifiable against the data.

---

## Known Limitations and Roadmap

- Memory embeddings currently use a bag-of-words approximation; sentence-transformer upgrade is the next priority
- Tool agent has two registered tools; full tool suite (similarity search, era comparison, feature search) is in progress
- No automated evaluation harness yet; manual session testing only
- Multi-agent contradiction detection uses lexical heuristics; LLM-structured output parsing is planned

---

## License

MIT License. See `LICENSE` for details.
