# Music Intelligence Multi-Agent System — Project Context

## Who I am
Particle physics PhD with ~20 years research, funding acquisition, and team management experience.
ML/AI and this project are not my primary field — treat explanations accordingly (no basics needed).
I know Taylor Swift's catalog deeply and act as the red-team for this dataset.

---

## What this system is
A multi-agent music recommendation and analysis system built on a structured Taylor Swift corpus.
- 153 songs, 11 albums, 5 eras, Spotify audio features + local lyric embeddings + BERTopic topics
- Five agents: Recommendation, Memory, Analysis Assistant, Tool Agent, Multi-Agent Orchestrator
- LLM backend switchable between OpenAI (deployment) and Ollama (local dev) via single config flag
- Streamlit frontend, deployed on Streamlit Community Cloud

**Current state (March 2026):** foundations stable, key bugs fixed, explainability layer in progress.

---

## The product vision

### Core value proposition
**Explainable recommendation as a first-class feature, not an afterthought.**
Spotify, Netflix, YouTube are black boxes. They recommend but never explain.
This system's explanations are grounded in actual reasoning, not post-hoc labels.
The explanation IS the recommendation — proactive, specific, data-grounded.

### Target user
Swifties (Taylor Swift fan community). Analytically engaged, will immediately stress-test
explanations, will share impressive outputs. One impressed user with a large account
is the growth mechanism — no paid distribution needed.

### Differentiators vs Spotify
1. Explainability — Spotify is institutionally prevented from showing their reasoning
2. Domain depth — BERTopic on a single artist's corpus produces thematically meaningful
   clusters that a general model across millions of songs cannot achieve
3. Memory across sessions — the system knows your taste history and references it explicitly
4. Cross-song callback layer — authorial intent as a similarity dimension (unique, not in literature)
5. Privacy-first — local lyric embeddings, no lyrics transmitted via API

---

## Dataset and era structure

| Era | Albums |
|-----|--------|
| Country | Taylor Swift, Fearless, Speak Now |
| Transition | Red |
| Pop | 1989, Reputation, Lover |
| Indie | Folklore, Evermore |
| Pop Revival | Midnights, TTPD |

**Key finding:** Era structure is non-linear in time. Pop Revival is not a continuation
of Indie — it's a partial return toward Pop with retained vocabulary complexity.
Validated in audio feature space (radar chart) and confirmed stable across 5 BERTopic runs.

**BERTopic clustering (stable, n=5 runs):**
- Cluster 1: Core Narrative Style — Topic 0, 48 songs (31% of catalog), cross-era
- Cluster 2: Stylistic Departures — e.g. Red album core, 22, The Lucky One

**Vocabulary diversity finding:** Pop Era has lowest unique word ratio — deliberate
simplification of 1989/Reputation era. Indie and Pop Revival highest — Aaron Dessner
influence and TTPD density. Real result, worth foregrounding in demo.

---

## Similarity foundation

| Component | Detail |
|-----------|--------|
| Lyric similarity | SentenceTransformer (all-MiniLM-L6-v2), full lyrics, cosine similarity |
| Audio similarity | 9 Spotify features, StandardScaler, cosine similarity |
| Hybrid weight | 60% lyric + 40% audio |
| Returns | `lyric_similarity`, `audio_similarity`, `hybrid_similarity`, `df`, `available_audio` |

**Upgrade path:** all-mpnet-base-v2 or mxbai-embed-large (via Ollama locally).
8GB VRAM confirmed available. Embeddings precomputed and stored — transformer not
loaded at serving time. Lyric data never leaves local environment (legal constraint).

**Known august anomaly:** August has folklore lyrical register but more energetic
production than typical folklore songs. 60/40 hybrid resolves toward lyric similarity.
Produces thematically coherent but audio-distant recommendations. Confirmed by red-team.

**Clean / So It Goes callback:** these two songs are distant in both lyric and audio
space despite sharing a deliberate word-level callback. Confirmed evidence that the
callback graph captures connections embedding models are blind to.

---

## Explainability architecture (in progress)

### Three-layer output schema (implemented in recommendation_agent.py)

**Layer 1 — Pure data, no LLM:**
`SongRecommendationExplanation` dataclass fields:
- similarity_score, lyric_similarity, audio_similarity
- dominant_signal: "lyric" | "audio" | "balanced"
- shared_era, era, shared_topic_cluster, topic_cluster_name
- feature_deltas (per audio feature, absolute difference)
- explore_or_exploit
- memory_reference (optional)

**Layer 2 — Constrained LLM:**
Prompt explicitly forbids: describing what songs are about, themes, narratives,
cultural context, artist biography, any prior knowledge.
Every claim must trace to a Layer 1 field. Calibrated to MINIMAL/STANDARD/DETAILED depth.

**Layer 3 — Easter egg aside (optional):**
Only fires when confirmed high-confidence lyrical callbacks exist between input song
and recommendation set. Separate constrained LLM call. Silent when no callbacks found.
"By the way..." tone — insider feel, not formal annotation.

### Key constraint
The LLM cannot reach for training knowledge (e.g. "commentary on societal double standards")
because the prompt closes that door explicitly. Specificity is enforced by structure,
not by hoping the LLM stays grounded.

---

## Easter egg dataset
File: `taylor_swift_easter_eggs.csv`
Columns: album, song, easter_egg_type, easter_egg, confidence, notes
105 entries across all albums.
Types: hidden_message, numeric_reference, lyrical_callback, rerecording_hint
Confidence: high / medium / low

**Coverage gaps:** TTPD Anthology disc (The Manuscript side) not covered — lyrics
not available. Life of a Showgirl disc — same issue. Needs manual extension.

**Usage in system:** loaded at agent init, filtered to high-confidence lyrical_callbacks only.
Keyed by song name (lowercase, spaces stripped) for fast lookup.
Cross-song callback detection checks input song against recommendation set bidirectionally.

**Future: callback graph as similarity dimension**
Cross-song callbacks encode deliberate authorial intent — a third similarity dimension
alongside lyric embeddings and audio features. Not yet implemented as ranking signal.
Correct sequencing: explanation-only first (current), then ranking integration later.
The graph would surface connections (e.g. Clean / So It Goes) that no embedding model recovers.

---

## Feedback architecture

### Current (being replaced)
Free-form text feedback parsed by LLM — expensive, unnecessary, breaks at scale.

### Target: structured checkbox feedback
Maps directly to UserPreferenceModel fields, zero LLM cost:
- Too energetic / too mellow → preferred_energy_range
- Too acoustic / too electronic → preferred_acousticness
- Too fast / too slow → preferred_tempo_range
- Too upbeat / too dark → preferred_valence_range
- Wrong era (with era selector) → disliked_eras
- Not my vibe → disliked_songs

**At scale across all artists — two-tier structure:**
1. Universal tier (fixed): energy, valence, tempo, acousticness, era — always present
2. Artist tier (derived): one or two checkboxes from that artist's BERTopic cluster
   structure, generated once at model-build time

~8-10 checkboxes total, 3 categories. Manageable regardless of catalog size.
Audio feature dimensions don't explode across artists — they're bounded by Spotify's feature space.

**LLM calls at scale:** only for explanation generation, not feedback processing.
One call per recommendation request regardless of feedback volume.

---

## Persistence and privacy

### Two-mode architecture
**One-time mode (default):** Streamlit session state only, resets on close.
Zero storage, zero GDPR surface. No infrastructure needed.

**Persistent mode (opt-in):** anonymous token in URL, JSON blob stored server-side.
No email, no name, no IP. Only preference model fields stored.
Right to erasure = one delete button, one line of code.
Genuinely anonymous data is outside GDPR scope if no linkage to real person exists.

**Consent UI:** simple modal at first interaction offering the choice.
"No account required. Your preferences are yours."
This framing resonates with Swiftie audience (Taylor's ownership fight parallel).

---

## Known bugs and status

| Issue | Status |
|-------|--------|
| Double-encoding bug (transformer runs twice) | Open |
| Non-song entries (13 liner notes/poems in dataset) | Open |
| Preference filtering post-retrieval instead of pre | Open |
| Memory agent bag-of-words embeddings | Fixed March 2026 |
| Analysis assistant conversation history (two lists) | Open |
| Tool agent missing 3 of 5 tool categories | Open |
| Orchestrator hardcoded contradiction heuristic | Open |
| Song name parsing for multi-word songs | Fixed March 2026 |
| Shared conversation history across LLM calls | Fixed March 2026 |
| Discovery path hallucinating out-of-catalog songs | Fixed March 2026 |
| Unconstrained LLM explanation (training knowledge bleed) | Fixed March 2026 |

---

## Roadmap priority (agreed sequencing)

**Now (foundation):**
- Fix double-encoding bug
- Filter non-song entries at data load time
- Wire session state properly in Streamlit
- Confirm `dominant_topic` column maps correctly (0 = core narrative)

**Next (explainability — the product):**
- Structured checkbox feedback UI replacing free-form text
- Taste profile visualisation (exportable, shareable — the Wrapped moment)
- Confidence threshold on explanations — silence better than a wrong explanation

**Later (depth):**
- Upgrade to all-mpnet-base-v2 embeddings
- Callback graph as ranking signal (after explanation-only validated)
- Version differentiation: Taylor's Version vs original (test Spotify feature delta first)
- Extend Easter egg dataset: TTPD Anthology, Life of a Showgirl

**Long term (scale):**
- Multi-artist corpus using audio features only (no lyrics needed)
- Anonymous persistence layer
- Cross-catalog preference transfer testing

---

## Scaling path
**Deep mode (current):** single artist, full lyric embeddings local, Easter egg layer,
BERTopic topic clusters. Legally constrained to local deployment.

**Broad mode (future):** multi-artist, Spotify audio features only, no lyrics.
Explainability still differentiates from Spotify even without lyric dimension.
Demo strategy: show Swift as deep case, then generalise to multi-artist as separate capability.

Lyrics at scale = legal nightmare. Alternatives: co-writer/producer credit networks
(completely legal, surprisingly powerful signal), audio features only via Spotify API.

---

## Tech stack
- Python, pandas, numpy
- sentence-transformers (all-MiniLM-L6-v2, upgrade path to all-mpnet-base-v2)
- BERTopic, scikit-learn
- Streamlit (frontend + Community Cloud deployment)
- OpenAI API (deployment LLM) / Ollama (local dev)
- Matplotlib/Seaborn (visualisations)
- 8GB VRAM local machine confirmed

---

## What not to do
- Do not try to beat Spotify on volume or collaborative filtering — unwinnable
- Do not use free-form LLM feedback at scale — unnecessary token cost
- Do not let LLM explanations reach for training knowledge — enforced by prompt structure
- Do not add login/authentication — GDPR complexity far exceeds demo value
- Do not implement callback graph as ranking signal before explanation-only is validated
