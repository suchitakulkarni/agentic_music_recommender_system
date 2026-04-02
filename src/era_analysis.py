from src import *
from src.feature_extraction import calculate_lexical_metrics, calculate_sentiment
def define_eras(merged_df):
    """Define Taylor Swift's musical eras."""
    era_definitions = {
        'Taylor Swift': 'Country Era',
        'TaylorSwift': 'Country Era',
        'FearlessPlatinumEdition': 'Country Era',
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
    
    merged_df['era'] = merged_df['Album'].map(era_definitions)
    merged_df['era'] = merged_df['era'].fillna('Other')
    
    return merged_df


def analyze_era_evolution(merged_df):
    """Comprehensive era evolution analysis."""
    print("\n" + "="*80)
    print("OPTION B: ERA EVOLUTION ANALYSIS")
    print("="*80)
    
    df = merged_df[merged_df['lyrics'] != ''].copy()
    df = define_eras(df)
    
    # Calculate metrics for each song
    print("\nCalculating lyrical metrics...")
    for idx, row in df.iterrows():
        lexical = calculate_lexical_metrics(row['lyrics'])
        sentiment = calculate_sentiment(row['lyrics'])
        
        df.loc[idx, 'avg_word_length'] = lexical['avg_word_length']
        df.loc[idx, 'unique_word_ratio'] = lexical['unique_word_ratio']
        df.loc[idx, 'total_words'] = lexical['total_words']
        df.loc[idx, 'polarity'] = sentiment['polarity']
        df.loc[idx, 'subjectivity'] = sentiment['subjectivity']
    
    # Add audio features if available
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
    available_audio = [f for f in audio_features if f in df.columns]
    
    # Build aggregation dict
    agg_dict = {
        'avg_word_length': ['mean', 'std'],
        'unique_word_ratio': ['mean', 'std'],
        'total_words': ['mean', 'std'],
        'polarity': ['mean', 'std'],
        'subjectivity': ['mean', 'std']
    }
    
    # Add available audio features to aggregation
    for feat in available_audio:
        agg_dict[feat] = ['mean', 'std']
    
    # Group by era with all features at once
    era_stats = df.groupby('era').agg(agg_dict)
    
    print("\n" + "="*80)
    print("ERA STATISTICS")
    print("="*80)
    print(era_stats)
    
    # Statistical significance tests
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    eras = df['era'].unique()
    if len(eras) >= 2:
        era1_data = df[df['era'] == eras[0]]
        era2_data = df[df['era'] == eras[1]]
        
        print(f"\nComparing {eras[0]} vs {eras[1]}:")
        
        # Test word complexity
        t_stat, p_val = stats.ttest_ind(
            era1_data['avg_word_length'].dropna(),
            era2_data['avg_word_length'].dropna()
        )
        print(f"  Word complexity: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
        
        # Test sentiment
        t_stat, p_val = stats.ttest_ind(
            era1_data['polarity'].dropna(),
            era2_data['polarity'].dropna()
        )
        print(f"  Sentiment polarity: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
    
    return df, era_stats


def create_era_audio_profile(df):
    """Create audio feature profiles for each era."""
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
    available_audio = [f for f in audio_features if f in df.columns]
    
    if not available_audio:
        print("\nNo audio features available for era profiling")
        return
    
    # Filter eras with enough data
    era_counts = df['era'].value_counts()
    valid_eras = era_counts[era_counts >= 3].index
    df_filtered = df[df['era'].isin(valid_eras)].copy()
    
    era_order = ['Country Era', 'Transition Era', 'Pop Era', 'Indie Era', 'Pop Revival Era']
    #era_order = ['Country Foundations', 'Crossover / Transition', 'Mainstream Pop', 'Indie Folk / Alternative', 'Alt-Pop Maturity']
    valid_order = [e for e in era_order if e in valid_eras]
    
    era_profiles = df_filtered.groupby('era')[available_audio].mean()
    era_profiles = era_profiles.reindex(valid_order).dropna(how='all')
    
    # Normalize to 0-1 for radar chart (tempo needs special handling)
    for col in era_profiles.columns:
        if col == 'tempo':
            # Normalize tempo to 0-1 range
            era_profiles[col] = (era_profiles[col] - era_profiles[col].min()) / (era_profiles[col].max() - era_profiles[col].min())
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(available_audio), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#8B4513', '#FF8C00', '#FF1493', '#4B0082', '#9370DB']
    
    for idx, era in enumerate(era_profiles.index):
        if idx < len(colors):
            color = colors[idx]
            values = era_profiles.loc[era].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=era, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_audio)
    ax.set_ylim(0, 1)
    ax.set_title('Musical Profile by Era', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/era_audio_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: results/era_audio_profile.png")


