"""Configuration and constants for Taylor Swift analysis."""
import os, sys
import spacy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_SCIENCE_DATA = os.getcwd()
if DATA_SCIENCE_DATA is None:
    raise ValueError("Environment variable DATA_PATH is not set")

RESULTS_DIR = os.path.join(DATA_SCIENCE_DATA, "results")
# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define file paths relative to DATA_PATH
SPOTIFY_CSV = os.path.join(DATA_SCIENCE_DATA, "data", "taylor_swift_spotify.csv")
ALBUM_SONG_CSV = os.path.join(DATA_SCIENCE_DATA, "data", "song_names_formatted.csv")
DATA_DIR = os.path.join(DATA_SCIENCE_DATA, "data")

LYRIC_EMBEDDINGS_PKL  = os.path.join(DATA_DIR, "lyric_embeddings.pkl")
LYRIC_SIMILARITY_NPY  = os.path.join(DATA_DIR, "lyric_similarity.npy")
AUDIO_SIMILARITY_NPY  = os.path.join(DATA_DIR, "audio_similarity.npy")
HYBRID_SIMILARITY_NPY = os.path.join(DATA_DIR, "hybrid_similarity.npy")
SONG_EMBEDDINGS_NPZ   = os.path.join(DATA_DIR, "song_embeddings.npz")
AGENT_MEMORY_JSON     = os.path.join(DATA_DIR, "agent_memory.json")
SONGS_WITH_ERAS_CSV   = os.path.join(DATA_DIR, "songs_with_eras.csv")
TOPIC_ERA_SUMMARY_CSV = os.path.join(DATA_DIR, "topic_era_summary.csv")
FINAL_TOPICS_CSV      = os.path.join(DATA_DIR, "final_topics_stable.csv")

#if not os.path.exists(SPOTIFY_CSV): print('no spotify data found'); sys.exit(0)
#if not os.path.exists(ALBUM_SONG_CSV): print('no Album data found'); sys.exit(0)
#if not os.path.exists(RESULTS_DIR): print('no result directory found'); sys.exit(0)
#if not os.path.exists(DATA_DIR): print('no data directory found'); sys.exit(0)

"""Configuration file for Taylor Swift analysis project."""
import os

# API Configuration
USE_OPENAI = True  # Toggle: True for OpenAI (fast), False for Ollama (local)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set as environment variable
SAFE_MODE = False
if USE_OPENAI == False:
    # Safe mode for public presentations (removes lyrics from OpenAI calls)
    SAFE_MODE = False  # Set to True when presenting publicly

# Maximum lyrics characters to process (if absolutely needed)
MAX_LYRICS_CHARS = 0 if SAFE_MODE else None

# Flag to use cached embeddings only (never reprocess lyrics)
USE_CACHED_EMBEDDINGS_ONLY = SAFE_MODE

if USE_OPENAI == True:
    # OpenAI settings
    MODEL = "gpt-4o-mini"  # Fast and cheap: $0.15/1M input tokens
    #TEMPERATURE = 0.7
    #MAX_TOKENS = 512
    FAST_MODEL = MODEL
    REASONING_MODEL = MODEL
else:
    # Ollama settings
    MODEL = "llama3.2:3b"  # For local development
    TEMPERATURE = 0.7
    MAX_TOKENS = 512

    # Alternative models for different tasks
    FAST_MODEL = "phi3:mini"  # For quick responses - ~2.3GB RAM
    REASONING_MODEL = MODEL  # Use same model for consistency

# Album filtering
KEEP_ALBUMS = [
    'THE TORTURED POETS DEPARTMENT',
    'Midnights',
    'evermore',
    'folklore',
    'Lover',
    'reputation',
    '1989',
    'Red',
    'Speak Now',
    'Fearless (Platinum Edition)',
    'Taylor Swift (Deluxe Edition)'
]

# Era definitions
ERA_DEFINITIONS = {
    'Taylor Swift': 'Country Era',
    'TaylorSwift': 'Country Era',
    'Fearless_PlatinumEdition_)': 'Country Era',
    'Fearless': 'Country Era',
    'Speak Now': 'Country Era',
    'SpeakNow': 'Country Era',
    'Red': 'Transition Era',
    '1989': 'Pop Era',
    'Reputation': 'Pop Era',
    'Lover': 'Pop Era',
    'Folklore': 'Indie Era',
    'Evermore': 'Indie Era',
    'Midnights': 'Pop Revival Era',
    'THETORTUREDPOETSDEPARTMENT': 'Pop Revival Era'
}

ERA_ORDER = ['Country Era', 'Transition Era', 'Pop Era', 'Indie Era', 'Pop Revival Era']

# Stopwords
LYRIC_STOPWORDS = {"oh", "yeah", "im", "i'm", "ha", "ah", "ooh", "woah", "la", "na",
                   "mmm", "hmm", "uh", "eh", "ay", "hey", "yo",
                   "dont", "wanna", "gonna", "outta", "gotta", "lemme",
                   "the", "you", "your", "youre", "its", "with",
                   "be", "and", "like", "that", "what", "all", "get", "but", "for",
                   "man", "make", "look", "take", "just", "could", "ill", "huh"}

seed_topic_list = {
       "Love & Romance": ["heart", "kiss", "dance", "beautiful", "stars"],
       "Heartbreak & Loss": ["tears", "cry", "hurt", "pain", "sorry", "rain"],
       "Nostalgia & Memory": ["remember", "memories", "back", "summer", "young"],
       "Revenge & Anger": ["karma", "mad", "lies", "mean", "hate"]
   }
# Audio features
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Model parameters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_ALPHA = 0.6  # Weight for lyrics in hybrid similarity

# Topic modeling
TOPIC_RANGE = [4, 5, 6, 7, 8]
LDA_MAX_FEATURES = 800
BERTOPIC_UMAP_NEIGHBORS = 10

# Random seed for reproducibility
RANDOM_SEED = 42

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# Add to config.py

# Ollama Configuration
#OLLAMA_MODEL = "llama3.1:8b"  # Default model
#OLLAMA_TEMPERATURE = 0.7
#OLLAMA_MAX_TOKENS = 1024

# Alternative models for different tasks
#OLLAMA_FAST_MODEL = "llama3.2:3b"  # For quick responses
#OLLAMA_REASONING_MODEL = "qwen2.5:7b"  # For complex analysis

# Ollama Configuration - Optimized for 16GB RAM MacBook Pro (2020, No GPU)
#OLLAMA_MODEL = "llama3.2:3b"  # Default model - ~2GB RAM
#OLLAMA_MODEL = "llama3.2:3b"  # Default model - ~2GB RAM
#OLLAMA_TEMPERATURE = 0.7
#OLLAMA_MAX_TOKENS = 512

# Alternative models for different tasks
#OLLAMA_FAST_MODEL = "phi3:mini"  # For quick responses - ~2.3GB RAM
#OLLAMA_REASONING_MODEL = "llama3.2:3b"  # Use same model for consistency
#OLLAMA_REASONING_MODEL = OLLAMA_MODEL  # Use same model for consistency
