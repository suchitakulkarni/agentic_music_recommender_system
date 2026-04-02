from src import *
def calculate_lexical_metrics(text):
    """Calculate lexical complexity metrics."""
    words = text.lower().split()
    if not words:
        return {
            'avg_word_length': 0,
            'unique_word_ratio': 0,
            'total_words': 0,
            'unique_words': 0
        }
    
    unique_words = set(words)
    return {
        'avg_word_length': np.mean([len(w) for w in words]),
        'unique_word_ratio': len(unique_words) / len(words),
        'total_words': len(words),
        'unique_words': len(unique_words)
    }

def calculate_sentiment(text):
    """Calculate sentiment."""
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }


