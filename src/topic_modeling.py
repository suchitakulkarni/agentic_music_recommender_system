from src import *
from src.preprocessing import preprocess_lyrics_enhanced

def improved_lda_topic_modeling(df, n_topics_range=[4, 5, 6, 7, 8], max_features=800):
    """LDA with hyperparameter tuning and coherence scoring."""
    print("\n" + "=" * 80)
    print("IMPROVED LDA TOPIC MODELING")
    print("=" * 80)

    # Preprocess
    df["clean_lyrics"] = df["lyrics"].astype(str).apply(preprocess_lyrics_enhanced)

    # Create vectorizer with bigrams and filtering
    vectorizer = CountVectorizer(
        stop_words='english',
        max_features=max_features,
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.7,  # Word can't appear in more than 70% of documents
        ngram_range=(1, 2)  # Include bigrams
    )

    X_counts = vectorizer.fit_transform(df['clean_lyrics'])
    feature_names = vectorizer.get_feature_names_out()

    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Document-term matrix shape: {X_counts.shape}")

    # Try different numbers of topics
    best_score = -np.inf
    best_model = None
    best_n = None

    coherence_scores = []

    for n in n_topics_range:
        print(f"\nTrying {n} topics...")

        lda = LatentDirichletAllocation(
            n_components=n,
            doc_topic_prior=0.1,  # Alpha: lower = more focused topics per doc
            topic_word_prior=0.01,  # Beta: lower = more focused words per topic
            learning_method='batch',
            max_iter=50,
            random_state=42
        )

        lda.fit(X_counts)

        # Simple coherence approximation (top word overlap penalty)
        # Better: use gensim's CoherenceModel, but keeping dependencies minimal
        top_words_per_topic = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-20:][::-1]
            top_words_per_topic.append(set([feature_names[i] for i in top_indices]))

        # Penalize topic overlap
        overlap_penalty = 0
        for i in range(len(top_words_per_topic)):
            for j in range(i + 1, len(top_words_per_topic)):
                overlap = len(top_words_per_topic[i] & top_words_per_topic[j])
                overlap_penalty += overlap

        # Score = perplexity (lower is better) - overlap penalty
        perplexity = lda.perplexity(X_counts)
        score = -perplexity - overlap_penalty * 0.1
        coherence_scores.append((n, score, perplexity))

        print(f"  Perplexity: {perplexity:.2f}, Overlap penalty: {overlap_penalty}, Score: {score:.2f}")

        if score > best_score:
            best_score = score
            best_model = lda
            best_n = n

    print(f"\nBest number of topics: {best_n} (score: {best_score:.2f})")

    # Display best model topics
    print("\n" + "-" * 80)
    print(f"BEST LDA MODEL ({best_n} topics)")
    print("-" * 80)

    topic_labels = []
    for topic_idx, topic in enumerate(best_model.components_):
        top_indices = topic.argsort()[-15:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"\nTopic {topic_idx + 1}:")
        print(", ".join(top_words[:10]))
        topic_labels.append(" ".join(top_words[:3]))  # Use top 3 words as label

    # Get topic distributions for each song
    topic_distributions = best_model.transform(X_counts)
    df['dominant_topic'] = topic_distributions.argmax(axis=1)
    df['topic_strength'] = topic_distributions.max(axis=1)

    # Add all topic probabilities
    for i in range(best_n):
        df[f'topic_{i}_prob'] = topic_distributions[:, i]

    # Visualize coherence scores
    fig, ax = plt.subplots(figsize=(10, 6))
    n_values = [x[0] for x in coherence_scores]
    scores = [x[1] for x in coherence_scores]
    ax.plot(n_values, scores, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Topics')
    ax.set_ylabel('Coherence Score')
    ax.set_title('LDA Topic Model Selection')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/lda_coherence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/lda_coherence.png")

    return best_model, X_counts, vectorizer, df, topic_labels

def improved_bertopic_modeling(df, n_topics_range=[4, 5, 6, 7, 8]):
    """BERTopic with optimized parameters for small lyric datasets."""
    print("\n" + "=" * 80)
    print("IMPROVED BERTopic MODELING")
    print("=" * 80)

    # Preprocess
    df["clean_lyrics"] = df["lyrics"].astype(str).apply(preprocess_lyrics_enhanced)

    # Use sentence transformer for embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(df["clean_lyrics"].tolist(), show_progress_bar=True)

    # Custom UMAP for small datasets
    umap_model = umap.UMAP(
        n_neighbors=10,  # Lower for small datasets
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # Custom vectorizer with lyrics stopwords
    vectorizer_model = CountVectorizer(
        stop_words=list(config.LYRIC_STOPWORDS),
        min_df=2,
        ngram_range=(1, 2)
    )

    best_topics = None
    best_model = None
    best_n = None
    best_diversity = -1

    for n in n_topics_range:
        print(f"\nTrying {n} topics...")

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            vectorizer_model=vectorizer_model,
            nr_topics=n,
            top_n_words=10,
            verbose=False,
            calculate_probabilities=True,
            seed_topic_list=config.seed_topic_list
        )

        topics, probs = topic_model.fit_transform(df["clean_lyrics"], embeddings)

        # Calculate topic diversity (unique words across topics)
        all_words = set()
        topic_words = []
        for topic_id in set(topics):
            if topic_id != -1:  # Skip outlier topic
                words = topic_model.get_topic(topic_id)
                topic_words.append(set([w for w, _ in words[:20]]))
                all_words.update([w for w, _ in words[:10]])

        diversity = len(all_words) / (n * 10) if n > 0 else 0  # Proportion of unique words
        print(f"  Topic diversity: {diversity:.3f}")
        print(f"  Documents in outlier topic: {sum([1 for t in topics if t == -1])}")

        if diversity > best_diversity:
            best_diversity = diversity
            best_model = topic_model
            best_topics = topics
            best_n = n

    print(f"\nBest configuration: {best_n} topics (diversity: {best_diversity:.3f})")

    # Display best model topics
    print("\n" + "-" * 80)
    print(f"BEST BERTopic MODEL ({best_n} topics)")
    print("-" * 80)

    topic_labels = []
    for topic_id in sorted(set(best_topics)):
        if topic_id == -1:
            continue
        words = best_model.get_topic(topic_id)
        top_words = [word for word, _ in words[:10]]
        print(f"\nTopic {topic_id + 1}:")
        print(", ".join(top_words))
        topic_labels.append(" ".join([word for word, _ in words[:3]]))

    # Add topic assignments to dataframe
    df['bertopic_id'] = best_topics
    topic_probs_array = np.array([best_model.probabilities_[i] for i in range(len(df))])
    df['bertopic_strength'] = topic_probs_array.max(axis=1)

    # Save model
    best_model.save("results/bertopic_model")
    print("\nSaved: results/bertopic_model")

    return best_model, best_topics, df, topic_labels


def compare_topic_models(df, n_topics=5, max_features=5000):
    """Compare LDA and BERTopic topic modeling on the same lyrics."""

    print("\n" + "=" * 80)
    print("LDA TOPIC MODELING")
    print("=" * 80)

    # --- LDA ---
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    X_counts = vectorizer.fit_transform(df['lyrics'])

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_counts)

    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:]]
        print(f"\nLDA Topic {idx + 1}:")
        print(", ".join(top_words))

    print("\n" + "=" * 80)
    print("BERTopic TOPIC MODELING")
    print("=" * 80)

    # --- BERTopic ---
    topic_model = BERTopic(nr_topics=n_topics, verbose=False)
    topics, probs = topic_model.fit_transform(df['lyrics'])

    for idx in range(len(set(topics)) - (1 if -1 in topics else 0)):
        topic_words = topic_model.get_topic(idx)
        top_words = [word for word, _ in topic_words[:10]]
        print(f"\nBERTopic Topic {idx + 1}:")
        print(", ".join(top_words))

    return lda, topic_model, topics, probs


def topic_modeling_lyrics(df, n_topics=10):
    """BERTopic pipeline for lyrics with embeddings and preprocessing."""

    print("\n" + "=" * 80)
    print("PREPROCESSING LYRICS")
    print("=" * 80)

    df["clean_lyrics"] = df["lyrics"].astype(str).apply(preprocess_lyrics)

    print("\n" + "=" * 80)
    print("FITTING BERTopic")
    print("=" * 80)

    # Use a compact, semantic embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=n_topics,
        verbose=True,
        calculate_probabilities=True
    )

    topics, probs = topic_model.fit_transform(df["clean_lyrics"])

    # Print top words per topic
    for idx in range(len(set(topics)) - (1 if -1 in topics else 0)):
        topic_words = topic_model.get_topic(idx)
        top_words = [word for word, _ in topic_words[:10]]
        print(f"\nTopic {idx + 1}:")
        print(", ".join(top_words))

    return topic_model, topics, probs


def topic_modeling(df, n_topics=5, max_features=5000):
    """Extract common lyrical themes using LDA topic modeling."""
    print("\n" + "=" * 80)
    print("TOPIC MODELING")
    print("=" * 80)

    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    X_counts = vectorizer.fit_transform(df['lyrics'])

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_counts)

    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        print(f"\nTopic {idx + 1}:")
        top_words = [words[i] for i in topic.argsort()[-10:]]
        print(", ".join(top_words))
    return lda
