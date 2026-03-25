from src import *
from src import config
from src.preprocessing import clean_lyrics, normalize_name, remove_TV, remove_feat

SAFE_MODE = getattr(config, 'SAFE_MODE', False)

def load_and_merge_data(datapath, spotify_csv, album_song_csv):
    album_song_df = pd.read_csv(album_song_csv, engine="python", quotechar='"')
    full_spotify_df = pd.read_csv(spotify_csv, engine="python", quotechar='"')

    keep_list = [
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
    spotify_df = full_spotify_df[full_spotify_df['album'].isin(keep_list)].copy()

    # Load lyrics
    lyrics_list = []
    for idx, row in album_song_df.iterrows():
        album = row['Album']
        song_name = row['Song_Name']
        album_path = os.path.join(datapath, 'Albums', album)
        song_path = os.path.join(album_path, f"{song_name}.txt")

        if os.path.exists(song_path):
            with open(song_path, 'r') as f:
                text = f.read()
                s_begin = text[text.find('Lyrics'):]
                s_end = s_begin[:s_begin.find('Embed')] if 'Embed' in s_begin else s_begin
                lyrics_list.append(clean_lyrics(s_end))
        else:
            lyrics_list.append("")

    album_song_df['lyrics'] = lyrics_list

    # SAFE MODE: Replace lyrics with placeholder for public demos
    if SAFE_MODE:
        print("⚠️  SAFE MODE: Lyrics replaced with feature-only placeholders")
        album_song_df['lyrics_available'] = album_song_df['lyrics'].apply(lambda x: len(x) > 0)
        album_song_df['lyrics'] = '[LYRICS REMOVED FOR DEMO]'
        # You can still process features from cached embeddings

    # Apply normalization
    spotify_df['song_clean'] = spotify_df['name'].apply(remove_feat).apply(remove_TV).apply(normalize_name)
    album_song_df['song_clean'] = album_song_df['Song_Name'].apply(remove_feat).apply(remove_TV).apply(normalize_name)

    spotify_df['album_clean'] = spotify_df['album'].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    album_song_df['album_clean'] = album_song_df['Album'].str.lower()

    merged_df = album_song_df.merge(
        spotify_df,
        on=['song_clean', 'album_clean'],
        how='inner',
        suffixes=('_album', '_spotify')
    )
    print('-'*60)

    # Find unmatched normalized names
    unmatched_names = set(album_song_df['song_clean']) - set(spotify_df['song_clean'])
    #unmatched_names = set(spotify_df['song_clean']) - set(album_song_df['song_clean'])

    #if unmatched_names:
    #    print("\nUnmatched after normalization (showing raw vs cleaned):")
    #    for name in unmatched_names:
    #        rows = album_song_df[album_song_df['song_clean'] == name]
    #        for _, row in rows.iterrows():
    #            print(f"Song raw: '{row['Song_Name']}' | Song clean: '{row['song_clean']}'")
    return merged_df
