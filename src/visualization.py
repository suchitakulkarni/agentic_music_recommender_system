from src import *
def visualize_embeddings(embeddings, labels, method="umap", filename=None):
    """Project embeddings to 2D with t-SNE or UMAP and color by era/album."""
    if filename == None: filename = os.path.join(config.DATA_SCIENCE_DATA, "Taylor_Swift_agentic", "results","embedding_projection.png")
    print("\n" + "=" * 80)
    print("VISUALIZE EMBEDDINGS")
    print("=" * 80)

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

    reduced = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", alpha=0.7)
    plt.title(f"Embedding projection with {method.upper()}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def visualize_topics_comprehensive(df, lda_model, bert_model, lda_labels, bert_labels, X_counts, filename = None):
    """Create comprehensive topic visualizations."""
    if filename == None: filename = os.path.join(config.DATA_SCIENCE_DATA, "Taylor_Swift_agentic", "results","topic_analysis_comprehensive.png")
    print("\n" + "=" * 80)
    print("TOPIC VISUALIZATION")
    print("=" * 80)

    # Filter valid eras
    era_counts = df['era'].value_counts()
    valid_eras = era_counts[era_counts >= 3].index
    df_viz = df[df['era'].isin(valid_eras)].copy()

    fig = plt.figure(figsize=(20, 12))

    # 1. LDA Topic Distribution by Era
    ax1 = plt.subplot(2, 3, 1)
    lda_topics = df_viz['dominant_topic'].values
    era_topic_matrix = pd.crosstab(df_viz['era'], lda_topics, normalize='index')
    sns.heatmap(era_topic_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('LDA Topics by Era')
    ax1.set_xlabel('Topic ID')
    ax1.set_ylabel('Era')

    # 2. BERTopic Distribution by Era
    ax2 = plt.subplot(2, 3, 2)
    bert_topics = df_viz['bertopic_id'].values
    valid_bert = bert_topics[bert_topics != -1]
    valid_eras_bert = df_viz.loc[bert_topics != -1, 'era']
    if len(valid_bert) > 0:
        era_bert_matrix = pd.crosstab(valid_eras_bert, valid_bert, normalize='index')
        sns.heatmap(era_bert_matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax2)
    ax2.set_title('BERTopic Topics by Era')
    ax2.set_xlabel('Topic ID')
    ax2.set_ylabel('Era')

    # 3. Topic Strength Distribution (LDA)
    ax3 = plt.subplot(2, 3, 3)
    df_viz.boxplot(column='topic_strength', by='dominant_topic', ax=ax3)
    ax3.set_title('LDA Topic Coherence')
    ax3.set_xlabel('Topic ID')
    ax3.set_ylabel('Topic Strength')
    plt.suptitle('')

    # 4. Album-Topic Heatmap (LDA)
    ax4 = plt.subplot(2, 3, 4)
    album_counts = df_viz['Album'].value_counts()
    valid_albums = album_counts[album_counts >= 5].index
    df_albums = df_viz[df_viz['Album'].isin(valid_albums)]
    if len(df_albums) > 0:
        album_topic_matrix = pd.crosstab(df_albums['Album'], df_albums['dominant_topic'], normalize='index')
        sns.heatmap(album_topic_matrix, annot=True, fmt='.2f', cmap='RdPu', ax=ax4)
    ax4.set_title('LDA Topics by Album')
    ax4.set_xlabel('Topic ID')
    ax4.set_ylabel('Album')

    # 5. Topic Evolution Timeline
    ax5 = plt.subplot(2, 3, 5)
    era_order = ['Country Era', 'Transition Era', 'Pop Era', 'Indie Era', 'Pop Revival Era']
    valid_order = [e for e in era_order if e in valid_eras]

    topic_counts_by_era = []
    for era in valid_order:
        era_df = df_viz[df_viz['era'] == era]
        topic_dist = era_df['dominant_topic'].value_counts(normalize=True).sort_index()
        topic_counts_by_era.append(topic_dist)

    if topic_counts_by_era:
        timeline_df = pd.DataFrame(topic_counts_by_era, index=valid_order).fillna(0)
        timeline_df.plot(kind='bar', stacked=True, ax=ax5, colormap='tab10')
        ax5.set_title('Topic Evolution Across Eras')
        ax5.set_xlabel('Era')
        ax5.set_ylabel('Topic Proportion')
        ax5.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 6. Representative Songs per Topic
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    text_content = "Top Songs by LDA Topic:\n\n"
    for topic_id in sorted(df_viz['dominant_topic'].unique()):
        topic_songs = df_viz[df_viz['dominant_topic'] == topic_id].nlargest(3, 'topic_strength')
        text_content += f"Topic {topic_id + 1} ({lda_labels[topic_id]}):\n"
        for _, song in topic_songs.iterrows():
            song_name = song.get('Formatted_name', song.get('Song_Name', 'Unknown'))
            text_content += f"  - {song_name} ({song['topic_strength']:.2f})\n"
        text_content += "\n"

    ax6.text(0.05, 0.95, text_content, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: results/topic_analysis_comprehensive.png")

    # Create topic-era summary table
    summary_data = []
    for era in valid_order:
        era_df = df_viz[df_viz['era'] == era]
        top_topic = era_df['dominant_topic'].mode()[0] if len(era_df) > 0 else -1
        avg_strength = era_df['topic_strength'].mean()
        summary_data.append({
            'era': era,
            'n_songs': len(era_df),
            'dominant_topic': int(top_topic),
            'topic_label': lda_labels[top_topic] if top_topic >= 0 else 'N/A',
            'avg_topic_strength': avg_strength
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/topic_era_summary.csv', index=False)
    print("Saved: results/topic_era_summary.csv")

    return summary_df


def visualize_era_evolution(df, filename = None):
    """Create comprehensive era evolution visualizations."""
    if filename is None: filename = os.path.join(config.DATA_SCIENCE_DATA, "Taylor_Swift_agentic", "results","era_evolution.png")
    # Filter out eras with too few songs
    era_counts = df['era'].value_counts()
    valid_eras = era_counts[era_counts >= 3].index  # At least 3 songs per era
    df_filtered = df[df['era'].isin(valid_eras)].copy()
    
    print(f"\nEras with sufficient data (n>=3): {list(valid_eras)}")
    print(f"Era counts: {era_counts.to_dict()}")
    #print(df[df['era']=='Other'])
    #sys.exit()
    
    # Define era order for plotting
    era_order = ['Country Era', 'Transition Era', 'Pop Era', 'Indie Era', 'Pop Revival Era']
    #era_order = ['Country Foundations', 'Crossover / Transition', 'Mainstream Pop', 'Indie Folk / Alternative', 'Alt-Pop Maturity']
    valid_order = [e for e in era_order if e in valid_eras]
    
    df_filtered['era'] = pd.Categorical(df_filtered['era'], categories=valid_order, ordered=True)
    df_sorted = df_filtered.sort_values('era')
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # 1. Word complexity
    sns.boxplot(data=df_sorted, x='era', y='avg_word_length', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Word Complexity Across Eras')
    axes[0, 0].set_xlabel('')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Vocabulary diversity
    sns.boxplot(data=df_sorted, x='era', y='unique_word_ratio', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Vocabulary Diversity Across Eras')
    axes[0, 1].set_xlabel('')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Song length
    sns.boxplot(data=df_sorted, x='era', y='total_words', ax=axes[0, 2], palette='Set2')
    axes[0, 2].set_title('Song Length Across Eras')
    axes[0, 2].set_xlabel('')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Sentiment polarity
    sns.boxplot(data=df_sorted, x='era', y='polarity', ax=axes[1, 0], palette='Set2')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Sentiment Polarity Across Eras')
    axes[1, 0].set_xlabel('')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Subjectivity
    sns.boxplot(data=df_sorted, x='era', y='subjectivity', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('Emotional Expression Across Eras')
    axes[1, 1].set_xlabel('')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Energy
    if 'energy' in df.columns:
        sns.boxplot(data=df_sorted, x='era', y='energy', ax=axes[1, 2], palette='Set2')
        axes[1, 2].set_title('Musical Energy Across Eras')
        axes[1, 2].set_xlabel('')
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        axes[1, 2].text(0.5, 0.5, 'No energy data', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
    
    # 7. Danceability
    if 'danceability' in df.columns:
        sns.boxplot(data=df_sorted, x='era', y='danceability', ax=axes[2, 0], palette='Set2')
        axes[2, 0].set_title('Danceability Across Eras')
        axes[2, 0].set_xlabel('')
        axes[2, 0].tick_params(axis='x', rotation=45)
    else:
        axes[2, 0].text(0.5, 0.5, 'No danceability data', 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
    
    # 8. Acousticness
    if 'acousticness' in df.columns:
        sns.boxplot(data=df_sorted, x='era', y='acousticness', ax=axes[2, 1], palette='Set2')
        axes[2, 1].set_title('Acousticness Across Eras')
        axes[2, 1].set_xlabel('')
        axes[2, 1].tick_params(axis='x', rotation=45)
    else:
        axes[2, 1].text(0.5, 0.5, 'No acousticness data', 
                       ha='center', va='center', transform=axes[2, 1].transAxes)
    
    # 9. Valence (happiness)
    if 'valence' in df.columns:
        sns.boxplot(data=df_sorted, x='era', y='valence', ax=axes[2, 2], palette='Set2')
        axes[2, 2].set_title('Valence (Musical Positivity) Across Eras')
        axes[2, 2].set_xlabel('')
        axes[2, 2].tick_params(axis='x', rotation=45)
    else:
        axes[2, 2].text(0.5, 0.5, 'No valence data', 
                       ha='center', va='center', transform=axes[2, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nSaved: results/era_evolution.png")


def create_era_audio_profile(df, filename = None):
    """Create audio feature profiles for each era."""
    if filename is None: filename = os.path.join(config.DATA_SCIENCE_DATA, "Taylor_Swift_agentic", "results",
                                                 "era_audio_profile.png")
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
    # era_order = ['Country Foundations', 'Crossover / Transition', 'Mainstream Pop', 'Indie Folk / Alternative', 'Alt-Pop Maturity']
    valid_order = [e for e in era_order if e in valid_eras]

    era_profiles = df_filtered.groupby('era')[available_audio].mean()
    era_profiles = era_profiles.reindex(valid_order).dropna(how='all')

    # Normalize to 0-1 for radar chart (tempo needs special handling)
    for col in era_profiles.columns:
        if col == 'tempo':
            # Normalize tempo to 0-1 range
            era_profiles[col] = (era_profiles[col] - era_profiles[col].min()) / (
                        era_profiles[col].max() - era_profiles[col].min())

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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: results/era_audio_profile.png")