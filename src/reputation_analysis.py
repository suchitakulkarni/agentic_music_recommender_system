from src import *
def analyze_reputation_vs_others(df):
    """Analyze what makes Reputation distinct compared to other albums."""
    print("\n" + "=" * 80)
    print("REPUTATION VS OTHERS ANALYSIS")
    print("=" * 80)

    # Separate Reputation songs
    rep_df = df[df['Album'] == 'Reputation'].copy()
    others_df = df[df['Album'] != 'Reputation'].copy()

    if rep_df.empty:
        print("No Reputation songs found in dataset.")
        return None

    # Features to compare
    features = [
        'avg_word_length', 'unique_word_ratio', 'total_words',
        'polarity', 'subjectivity',
        'danceability', 'energy', 'valence', 'acousticness', 'tempo', 'loudness'
    ]
    available = [f for f in features if f in df.columns]

    results = []

    for feat in available:
        rep_vals = rep_df[feat].dropna()
        other_vals = others_df[feat].dropna()

        if len(rep_vals) > 2 and len(other_vals) > 2:
            t_stat, p_val = stats.ttest_ind(rep_vals, other_vals, equal_var=False)
            rep_mean = rep_vals.mean()
            other_mean = other_vals.mean()

            results.append({
                'feature': feat,
                'rep_mean': rep_mean,
                'others_mean': other_mean,
                'difference': rep_mean - other_mean,
                't_stat': t_stat,
                'p_val': p_val
            })

    results_df = pd.DataFrame(results).sort_values(by='p_val')

    print("\nDistinctive features (sorted by significance):")
    print(results_df[['feature', 'rep_mean', 'others_mean', 'difference', 'p_val']])

    # Visualization: Feature comparisons
    if not results_df.empty:
        top_feats = results_df.head(6)['feature']
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for i, feat in enumerate(top_feats):
            sns.boxplot(data=df[df['Album'].isin(['Reputation']) | (df['Album'] != 'Reputation')],
                        x=(df['Album'] == 'Reputation').map({True: 'Reputation', False: 'Others'}),
                        y=feat, ax=axes[i], palette='Set2')
            axes[i].set_title(f"{feat} comparison")
            axes[i].set_xlabel('')

        plt.tight_layout()
        plt.savefig('results/reputation_vs_others.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: results/reputation_vs_others.png")

    return results_df
