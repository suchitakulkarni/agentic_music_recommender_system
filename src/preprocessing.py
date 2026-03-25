from src import *
import src.config as config
'''try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")'''

def clean_lyrics(text):
    """Remove structural markers from lyrics."""
    #text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"[\"'‘’“”]", "", text)
    text = re.sub(r'()\[.*?...\]', '', text)
    text = re.sub(r"[\"']", '', text)
    markers_to_remove = [
        'Lyrics', 'Embed', 'Chorus:', 'Verse:', 'Bridge:',
        'Outro:', 'Intro:', 'Pre-Chorus:', 'Post-Chorus:'
    ]
    for marker in markers_to_remove:
        text = text.replace(marker, '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_name(name: str) -> str:
    """Normalize song names: lowercase, remove spaces, underscores, apostrophes."""
    if pd.isna(name):
        return ""
    # lowercase
    #name = name.replace('(feat. Bon Iver)', '')
    #name = name.replace('(Taylor\'s Version)', '')
    name = name.replace(' - Pop Version', '').lower()
    name = name.replace('_Poem_','').lower()
    # remove underscores, apostrophes, spaces
    name = re.sub(r"[-,.!_'?\&\s]", "", name)
    return name.strip()

def remove_feat(text: str) -> str:
    # remove "(feat. ...)" including everything after
    return re.sub(r"\(feat\..*?\)", "", text, flags=re.IGNORECASE).strip()

def remove_TV(text: str) -> str:
    # remove "(feat. ...)" including everything after
    text = re.sub(r"\(Taylor’s\..*?\)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\(Taylor's\..*?\)", "", text, flags=re.IGNORECASE).strip()
    return text

def preprocess_lyrics_enhanced(text):
    """Enhanced cleaning: handle contractions, reduce repetition, keep some structure."""
    if pd.isna(text) or not text:
        return ""

    # SAFE MODE: Return empty string to prevent processing
    if getattr(config, 'SAFE_MODE', False):
        return "[PROCESSED]"  # Placeholder for cached embeddings

    # Lowercase
    text = text.lower()

    # Handle common contractions BEFORE removing punctuation
    contractions = {
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have",
        "i'll": "i will", "you'll": "you will", "he'll": "he will",
        "won't": "will not", "can't": "cannot", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "isn't": "is not",
        "wasn't": "was not", "weren't": "were not", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not",
        "shouldn't": "should not", "couldn't": "could not"
    }

    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remove excessive repetition (e.g., "baby baby baby" -> "baby")
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text)

    # Remove special characters but keep some structure
    text = re.sub(r"[^a-z\s\-']", "", text)

    # Process with spaCy
    doc = config.nlp(text)

    # Lemmatize and filter
    tokens = []
    for token in doc:
        # Skip if stopword, too short, or not alpha
        if (token.lemma_ in config.LYRIC_STOPWORDS or
                len(token.text) <= 2 or
                not token.is_alpha):
            continue
        tokens.append(token.lemma_)

    return " ".join(tokens)
