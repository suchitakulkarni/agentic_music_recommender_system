from src import *
#def analyze_album_alignment(df, rep_results, favorite_songs):
#def analyze_personal_preferences(df, rep_results, favorite_songs):
from src import *
from scipy.spatial.distance import euclidean

def analyze_personal_preferences(df, rep_results, favorite_songs, k=10):
#def analyze_individual_similarity(df, rep_results, favorite_songs, k=10):
    """
    For each favorite song, find k most similar songs and analyze patterns.
    """
    print("\n" + "=" * 80)
    print("INDIVIDUAL SONG SIMILARITY ANALYSIS")
    print("=" * 80)

    # Clean song names
    df['song_lower'] = df['Song_Name'].str.lower()
    fav_lower = [s.lower().strip() for s in favorite_songs]
    fav_df = df[df['song_lower'].isin(fav_lower)].copy()

    if fav_df.empty:
        print("No favorite songs matched.")
        return None

    print(f"\nAnalyzing {len(fav_df)} favorites with k={k} nearest neighbors each:\n")

    # Get significant features to use for similarity
    if rep_results is not None and not rep_results.empty:
        sig_features = rep_results[rep_results['p_val'] < 0.05]['feature'].tolist()
    else:
        # Fallback to common audio features
        sig_features = ['acousticness', 'danceability', 'energy', 'valence', 'tempo']

    sig_features = [f for f in sig_features if f in df.columns]

    # Store all similar songs
    all_similar = []

    for idx, fav_row in fav_df.iterrows():
        fav_name = fav_row['Song_Name']
        fav_album = fav_row['Album']
        fav_vector = fav_row[sig_features].values

        # Calculate distance to all other songs
        distances = []
        for idx2, row in df.iterrows():
            if idx == idx2:  # Skip self
                continue
            other_vector = row[sig_features].values
            dist = euclidean(fav_vector, other_vector)
            distances.append({
                'song': row['Song_Name'],
                'album': row['Album'],
                'distance': dist
            })

        # Get k nearest neighbors
        distances_df = pd.DataFrame(distances).sort_values('distance').head(k)

        print(f"'{fav_name}' (from {fav_album}):")
        print(f"  Most similar songs:")
        for i, row in distances_df.iterrows():
            print(f"    {row['song']:40s} [{row['album']}]")
        print()

        # Store for aggregate analysis
        for _, row in distances_df.iterrows():
            all_similar.append({
                'favorite': fav_name,
                'similar_song': row['song'],
                'similar_album': row['album'],
                'distance': row['distance']
            })

    # Aggregate analysis
    similar_df = pd.DataFrame(all_similar)
    album_counts = similar_df['similar_album'].value_counts()

    print("\n" + "=" * 80)
    print("AGGREGATE PATTERN ANALYSIS")
    print("=" * 80)
    print(f"\nAlbum distribution across all {len(similar_df)} similar songs:")
    for album, count in album_counts.items():
        pct = 100 * count / len(similar_df)
        print(f"  {album:20s}: {count:3d} songs ({pct:.1f}%)")

    # Reputation-specific analysis
    rep_count = album_counts.get('Reputation', 0)
    rep_pct = 100 * rep_count / len(similar_df)
    print(f"\nReputation Alignment: {rep_pct:.1f}% of similar songs are from Reputation")

    # Visualization 1: Album distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#ff0000' if alb == 'Reputation' else '#1f77b4' for alb in album_counts.index]
    ax1.barh(album_counts.index, album_counts.values, color=colors)
    ax1.set_xlabel('Number of Similar Songs')
    ax1.set_title('Which Albums Are Your Favorites Similar To?')
    ax1.invert_yaxis()

    # Visualization 2: Per-favorite breakdown
    fav_album_matrix = similar_df.groupby(['favorite', 'similar_album']).size().unstack(fill_value=0)
    fav_album_matrix.plot(kind='barh', stacked=True, ax=ax2, legend=True)
    ax2.set_xlabel('Number of Similar Songs')
    ax2.set_title('Album Breakdown by Favorite Song')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/individual_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/individual_similarity.png")

    return similar_df, album_counts