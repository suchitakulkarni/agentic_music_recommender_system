"""Data dictionary for Taylor Swift song dataset."""


class DataDictionary:
    """Column definitions and descriptions for the dataset."""

    COLUMNS = {
        # Song identifiers
        'Song_Name': 'Name of the song',
        'Album': 'Album name the song belongs to',
        'album_clean': 'Cleaned album name',
        'era': 'Musical era (debut, fearless, speak_now, red, 1989, reputation, lover, folklore, evermore, midnights)',
        'Release_Date': 'Date when the song was released',

        # Audio features (0-1 scale)
        'danceability': 'How suitable the song is for dancing (0=not danceable, 1=very danceable)',
        'energy': 'Perceptual measure of intensity and activity (0=calm, 1=energetic)',
        'valence': 'Musical positivity/happiness (0=sad/negative, 1=happy/positive)',
        'acousticness': 'Confidence the track is acoustic (0=not acoustic, 1=acoustic)',
        'instrumentalness': 'Predicts whether track has no vocals (0=vocals, 1=instrumental)',
        'liveness': 'Detects presence of audience in recording (0=studio, 1=live)',
        'speechiness': 'Detects presence of spoken words (0=music, 1=speech)',

        # Other audio metrics
        'loudness': 'Overall loudness in decibels (typically -60 to 0)',
        'tempo': 'Estimated tempo in beats per minute (BPM)',
        'duration_ms': 'Duration of the song in milliseconds',
        'key': 'Musical key the song is in (0-11, corresponds to pitch classes)',
        'mode': 'Modality of track (0=minor, 1=major)',
        'time_signature': 'Time signature (beats per measure)',

        # Sentiment analysis
        'polarity': 'Sentiment polarity from lyrics (-1=negative, 1=positive)',
        'subjectivity': 'Subjectivity of lyrics (0=objective, 1=subjective)',

        # Topic modeling
        'dominant_topic': 'Primary topic/theme identified in the song',
        'topic_weight': 'Strength of the dominant topic assignment'
    }