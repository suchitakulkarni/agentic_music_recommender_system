import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_favorite_characteristics(df, favorite_songs):
    """
    Analyze common characteristics of favorite songs with consistency tracking.
    Includes both audio features and lyrical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged records dataframe with Spotify audio features and lyrical features
    favorite_songs : list
        List of favorite song names
    
    Returns:
    --------
    dict : Analysis results with statistics and consistency metrics
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "=" * 80)
    print("FAVORITE SONGS CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    # Audio features to analyze
    audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                      'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 
                      'popularity', 'duration_ms']
    
    # Lyrical features to analyze
    lyrical_features = ['avg_word_length', 'unique_word_ratio', 'total_words', 
                       'polarity', 'subjectivity', 'strength']
    
    # Combine all features
    all_features = audio_features + lyrical_features
    
    # Clean song names and match favorites
    df['song_lower'] = df['Song_Name'].str.lower().str.strip()
    fav_lower = [s.lower().strip() for s in favorite_songs]
    
    # Split into favorites and non-favorites
    fav_df = df[df['song_lower'].isin(fav_lower)].copy()
    non_fav_df = df[~df['song_lower'].isin(fav_lower)].copy()
    
    if fav_df.empty:
        print("ERROR: No favorite songs matched in the dataset.")
        return None
    
    n_favorites = len(fav_df)
    print(f"\nFound {n_favorites} favorite songs out of {len(df)} total songs")
    print(f"Matched favorites: {', '.join(fav_df['Song_Name'].tolist())}\n")
    
    # Calculate statistics for each feature WITH consistency tracking
    results = []
    for feature in all_features:
        if feature not in df.columns:
            continue
            
        # Skip if all values are NaN
        if df[feature].isna().all() or fav_df[feature].isna().all():
            continue
            
        fav_values = fav_df[feature].dropna().values
        if len(fav_values) == 0:
            continue
            
        fav_mean = fav_df[feature].mean()
        fav_std = fav_df[feature].std()
        overall_mean = df[feature].mean()
        overall_std = df[feature].std()
        overall_median = df[feature].median()
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(fav_df) - 1) * fav_std**2 + (len(non_fav_df) - 1) * non_fav_df[feature].std()**2) / 
                            (len(fav_df) + len(non_fav_df) - 2))
        cohens_d = (fav_mean - overall_mean) / pooled_std if pooled_std > 0 else 0
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(fav_df[feature].dropna(), non_fav_df[feature].dropna())
        
        # Calculate percentile rank
        percentile = stats.percentileofscore(df[feature].dropna(), fav_mean)
        
        # CONSISTENCY ANALYSIS
        # Count how many favorites are above/below the overall median
        above_median = sum(fav_df[feature].dropna() > overall_median)
        below_median = sum(fav_df[feature].dropna() < overall_median)
        
        # Determine consistency direction and strength
        if above_median > below_median:
            consistency_direction = "HIGH"
            consistency_count = above_median
        elif below_median > above_median:
            consistency_direction = "LOW"
            consistency_count = below_median
        else:
            consistency_direction = "MIXED"
            consistency_count = 0
        
        consistency_ratio = consistency_count / n_favorites
        
        # Calculate coefficient of variation for favorites (lower = more consistent)
        cv_favorites = (fav_std / abs(fav_mean)) if fav_mean != 0 else 0
        cv_overall = (overall_std / abs(overall_mean)) if overall_mean != 0 else 0
        
        # Determine feature type
        feature_type = "Lyrical" if feature in lyrical_features else "Audio"
        
        results.append({
            'feature': feature,
            'type': feature_type,
            'fav_mean': fav_mean,
            'fav_std': fav_std,
            'overall_mean': overall_mean,
            'overall_median': overall_median,
            'overall_std': overall_std,
            'difference': fav_mean - overall_mean,
            'percent_diff': ((fav_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0,
            'cohens_d': cohens_d,
            'p_value': p_val,
            'percentile': percentile,
            'significant': p_val < 0.05,
            'consistency_direction': consistency_direction,
            'consistency_count': consistency_count,
            'consistency_ratio': consistency_ratio,
            'cv_favorites': cv_favorites,
            'cv_overall': cv_overall,
            'is_consistent': consistency_ratio >= 0.67  # At least 2/3 agree
        })
    
    results_df = pd.DataFrame(results).sort_values('consistency_ratio', ascending=False)
    
    # Print results with consistency
    print("\n" + "=" * 80)
    print("CHARACTERISTIC COMPARISON WITH CONSISTENCY")
    print("=" * 80)
    print(f"\n{'Feature':<25} {'Type':<10} {'Fav Avg':<10} {'Overall':<10} {'Direction':<10} {'Agreement':<15} {'Effect':<10}")
    print("-" * 110)
    
    for _, row in results_df.iterrows():
        agreement_str = f"{row['consistency_count']}/{n_favorites} {row['consistency_direction']}"
        consistency_marker = " [CONSISTENT]" if row['is_consistent'] else ""
        sig_marker = " *" if row['significant'] else ""
        
        print(f"{row['feature']:<25} {row['type']:<10} {row['fav_mean']:<10.3f} {row['overall_mean']:<10.3f} "
              f"{row['consistency_direction']:<10} {agreement_str:<15} {row['cohens_d']:>9.3f}{sig_marker}{consistency_marker}")
    
    # Separate analysis by feature type
    print("\n" + "=" * 80)
    print("STRONG COMMON CHARACTERISTICS")
    print("=" * 80)
    
    strong_features = results_df[results_df['is_consistent']].copy()
    
    if strong_features.empty:
        print(f"\nNo features where at least {int(0.67*n_favorites)+1}/{n_favorites} songs agree")
    else:
        # Audio features
        strong_audio = strong_features[strong_features['type'] == 'Audio']
        if not strong_audio.empty:
            print(f"\nAUDIO FEATURES (where {int(0.67*n_favorites)+1}/{n_favorites}+ songs agree):\n")
            for _, row in strong_audio.iterrows():
                direction_word = "higher" if row['consistency_direction'] == "HIGH" else "lower"
                print(f"  - {row['feature'].upper()}: {row['consistency_count']}/{n_favorites} songs have {direction_word} than median")
                print(f"    Your avg: {row['fav_mean']:.3f} | Catalog median: {row['overall_median']:.3f} | "
                      f"Effect size: {row['cohens_d']:.2f}")
                
                if row['significant']:
                    print(f"    Statistically significant (p = {row['p_value']:.4f})")
                print()
        
        # Lyrical features
        strong_lyrical = strong_features[strong_features['type'] == 'Lyrical']
        if not strong_lyrical.empty:
            print(f"\nLYRICAL FEATURES (where {int(0.67*n_favorites)+1}/{n_favorites}+ songs agree):\n")
            for _, row in strong_lyrical.iterrows():
                direction_word = "higher" if row['consistency_direction'] == "HIGH" else "lower"
                print(f"  - {row['feature'].upper()}: {row['consistency_count']}/{n_favorites} songs have {direction_word} than median")
                print(f"    Your avg: {row['fav_mean']:.3f} | Catalog median: {row['overall_median']:.3f} | "
                      f"Effect size: {row['cohens_d']:.2f}")
                
                if row['significant']:
                    print(f"    Statistically significant (p = {row['p_value']:.4f})")
                print()
    
    # Weak or mixed patterns
    weak_features = results_df[~results_df['is_consistent']].copy()
    if not weak_features.empty:
        print("\n" + "=" * 80)
        print("MIXED/WEAK PATTERNS (Not shared by most songs)")
        print("=" * 80 + "\n")
        for _, row in weak_features.head(8).iterrows():
            print(f"  - {row['feature']} [{row['type']}]: {row['consistency_count']}/{n_favorites} songs {row['consistency_direction']}")
    
    # Show individual song profiles for context
    print("\n" + "=" * 80)
    print("INDIVIDUAL SONG PROFILES")
    print("=" * 80 + "\n")
    
    for idx, song_row in fav_df.iterrows():
        print(f"'{song_row['Song_Name']}' [{song_row['Album']}]:")
        
        # Show top distinctive features for this song (audio and lyrical)
        song_features = []
        for feature in all_features:
            if feature not in df.columns or pd.isna(song_row[feature]):
                continue
            value = song_row[feature]
            percentile = stats.percentileofscore(df[feature].dropna(), value)
            feature_type = "Lyrical" if feature in lyrical_features else "Audio"
            song_features.append({
                'feature': feature,
                'type': feature_type,
                'value': value,
                'percentile': percentile
            })
        
        song_features_df = pd.DataFrame(song_features)
        # Find most extreme (closest to 0 or 100 percentile)
        song_features_df['extremeness'] = song_features_df['percentile'].apply(lambda x: abs(50 - x))
        top_features = song_features_df.nlargest(5, 'extremeness')
        
        for _, feat in top_features.iterrows():
            direction = "high" if feat['percentile'] > 50 else "low"
            print(f"  - {feat['feature']} [{feat['type']}]: {feat['value']:.3f} ({direction}, {feat['percentile']:.0f}th percentile)")
        print()
    
    # Visualization 1: Feature comparison with consistency
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Consistency ratio (separated by type)
    ax1 = axes[0, 0]
    
    # Color by type and consistency
    colors = []
    for _, row in results_df.iterrows():
        if row['is_consistent']:
            colors.append('darkgreen' if row['type'] == 'Audio' else 'darkblue')
        else:
            colors.append('lightgreen' if row['type'] == 'Audio' else 'lightblue')
    
    bars = ax1.barh(results_df['feature'], results_df['consistency_ratio'], color=colors)
    ax1.axvline(x=0.67, color='red', linestyle='--', linewidth=2, label='Consistency Threshold (2/3)')
    ax1.set_xlabel("Consistency Ratio (fraction agreeing)")
    ax1.set_title(f"Feature Consistency Across {n_favorites} Favorites\n(Green=Audio, Blue=Lyrical)")
    ax1.set_xlim(0, 1)
    ax1.legend()
    ax1.invert_yaxis()
    
    # Add agreement labels
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax1.text(row['consistency_ratio'] + 0.02, i, 
                f"{row['consistency_count']}/{n_favorites}", 
                va='center', fontsize=8)
    
    # Plot 2: Effect sizes (only for consistent features)
    ax2 = axes[0, 1]
    consistent_df = results_df[results_df['is_consistent']].sort_values('cohens_d', key=abs, ascending=False)
    if not consistent_df.empty:
        colors = []
        for _, row in consistent_df.iterrows():
            if row['significant']:
                colors.append('darkgreen' if row['type'] == 'Audio' else 'darkblue')
            else:
                colors.append('lightgreen' if row['type'] == 'Audio' else 'lightblue')
        
        ax2.barh(consistent_df['feature'], consistent_df['cohens_d'], color=colors)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.axvline(x=-0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.set_xlabel("Effect Size (Cohen's d)")
        ax2.set_title("Effect Sizes for Consistent Features\n(Dark=Significant, Green=Audio, Blue=Lyrical)")
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No consistent features found', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Effect Sizes for Consistent Features")
    
    # Plot 3: Feature type breakdown
    ax3 = axes[1, 0]
    
    type_consistency = results_df.groupby(['type', 'is_consistent']).size().unstack(fill_value=0)
    type_consistency.plot(kind='barh', stacked=True, ax=ax3, 
                         color=['lightgray', 'darkgreen'], 
                         legend=True)
    ax3.set_xlabel('Number of Features')
    ax3.set_title('Consistency by Feature Type')
    ax3.legend(['Not Consistent', 'Consistent'], loc='best')
    
    # Plot 4: Album distribution
    ax4 = axes[1, 1]
    album_counts = fav_df['Album'].value_counts()
    ax4.barh(album_counts.index, album_counts.values, color='darkgreen', alpha=0.7)
    ax4.set_xlabel('Number of Favorite Songs')
    ax4.set_title('Album Distribution of Your Favorites')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('results/favorite_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: results/favorite_characteristics.png")
    
    # Visualization 2: Detailed consistency breakdown
    if n_favorites <= 4:  # Only create this for small sets
        fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(all_features) * 0.3)))
        fig.patch.set_facecolor('#1a4d2e')  # Dark green background
        
        # Audio features consistency map
        ax1 = axes[0]
        ax1.set_facecolor('#1a4d2e')  # Dark green background
        consistency_matrix_audio = []
        audio_labels = []
        
        for feature in [f for f in audio_features if f in df.columns]:
            overall_median = df[feature].median()
            row = []
            for _, song in fav_df.iterrows():
                if pd.isna(song[feature]):
                    row.append(0)
                elif song[feature] > overall_median:
                    row.append(1)  # Above median
                elif song[feature] < overall_median:
                    row.append(-1)  # Below median
                else:
                    row.append(0)  # At median
            consistency_matrix_audio.append(row)
            audio_labels.append(feature)
        
        if consistency_matrix_audio:
            consistency_matrix_audio = np.array(consistency_matrix_audio)
            sns.heatmap(consistency_matrix_audio, 
                       xticklabels=fav_df['Song_Name'].values,
                       yticklabels=audio_labels,
                       cmap='RdYlBu_r', center=0, 
                       cbar_kws={'label': 'Relative to Median', 'ticks': [-1, 0, 1]},
                       annot=True, fmt='d', annot_kws={'color': 'white', 'fontsize': 10, 'weight': 'bold'},
                       ax=ax1, linewidths=0.5, linecolor='#1a4d2e')
            ax1.set_title('AUDIO Features: Consistency Map\n(Red=Above, Yellow=Neutral, Blue=Below)', 
                         color='white', fontsize=14, weight='bold', pad=15)
            ax1.set_xlabel('Songs', color='white', fontsize=12, weight='bold')
            ax1.set_ylabel('Features', color='white', fontsize=12, weight='bold')
            ax1.tick_params(colors='white', labelsize=10)
            # Style colorbar
            cbar1 = ax1.collections[0].colorbar
            cbar1.ax.yaxis.set_tick_params(color='white', labelcolor='white')
            cbar1.ax.set_ylabel('Relative to Median', color='white', fontsize=10, weight='bold')
            cbar1.outline.set_edgecolor('white')
        
        # Lyrical features consistency map
        ax2 = axes[1]
        ax2.set_facecolor('#1a4d2e')  # Dark green background
        consistency_matrix_lyrical = []
        lyrical_labels = []
        
        for feature in [f for f in lyrical_features if f in df.columns]:
            overall_median = df[feature].median()
            row = []
            for _, song in fav_df.iterrows():
                if pd.isna(song[feature]):
                    row.append(0)
                elif song[feature] > overall_median:
                    row.append(1)  # Above median
                elif song[feature] < overall_median:
                    row.append(-1)  # Below median
                else:
                    row.append(0)  # At median
            consistency_matrix_lyrical.append(row)
            lyrical_labels.append(feature)
        
        if consistency_matrix_lyrical:
            consistency_matrix_lyrical = np.array(consistency_matrix_lyrical)
            sns.heatmap(consistency_matrix_lyrical, 
                       xticklabels=fav_df['Song_Name'].values,
                       yticklabels=lyrical_labels,
                       cmap='RdYlBu_r', center=0, 
                       cbar_kws={'label': 'Relative to Median', 'ticks': [-1, 0, 1]},
                       annot=True, fmt='d', annot_kws={'color': 'white', 'fontsize': 10, 'weight': 'bold'},
                       ax=ax2, linewidths=0.5, linecolor='#1a4d2e')
            ax2.set_title('LYRICAL Features: Consistency Map\n(Red=Above, Yellow=Neutral, Blue=Below)', 
                         color='white', fontsize=14, weight='bold', pad=15)
            ax2.set_xlabel('Songs', color='white', fontsize=12, weight='bold')
            ax2.set_ylabel('Features', color='white', fontsize=12, weight='bold')
            ax2.tick_params(colors='white', labelsize=10)
            # Style colorbar
            cbar2 = ax2.collections[0].colorbar
            cbar2.ax.yaxis.set_tick_params(color='white', labelcolor='white')
            cbar2.ax.set_ylabel('Relative to Median', color='white', fontsize=10, weight='bold')
            cbar2.outline.set_edgecolor('white')
        
        plt.tight_layout()
        plt.savefig('results/consistency_map.png', dpi=300, bbox_inches='tight', facecolor='#1a4d2e')
        plt.close()
        print(f"Saved: results/consistency_map.png")
    
    # Return results
    return {
        'results_df': results_df,
        'favorites_df': fav_df,
        'consistent_features': strong_features,
        'consistent_audio': strong_features[strong_features['type'] == 'Audio'] if not strong_features.empty else pd.DataFrame(),
        'consistent_lyrical': strong_features[strong_features['type'] == 'Lyrical'] if not strong_features.empty else pd.DataFrame(),
        'summary': {
            'n_favorites': n_favorites,
            'n_consistent_total': len(strong_features),
            'n_consistent_audio': len(strong_features[strong_features['type'] == 'Audio']) if not strong_features.empty else 0,
            'n_consistent_lyrical': len(strong_features[strong_features['type'] == 'Lyrical']) if not strong_features.empty else 0,
            'top_characteristic': results_df.iloc[0]['feature'] if not results_df.empty else None,
            'top_consistency': results_df.iloc[0]['consistency_ratio'] if not results_df.empty else None
        }
    }


# Example usage:
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_merged_data.csv')
    
    # Define favorite songs
    favorite_songs = [
        "Don't Blame Me",
        "Getaway Car",
        "I Did Something Bad"
    ]
    
    # Run analysis
    # results = analyze_favorite_characteristics(df, favorite_songs)
    pass