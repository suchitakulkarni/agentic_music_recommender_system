from src import *
def era_classifier(df, lyric_embeddings, audio_features=['danceability', 'energy', 'valence', 'acousticness', 'tempo']):
    """Train a simple classifier to predict song era from lyrics + audio features."""
    print("\n" + "=" * 80)
    print("ERA CLASSIFICATION")
    print("=" * 80)

    # Prepare features
    X = lyric_embeddings
    if any(f in df.columns for f in audio_features):
        X_audio = df[audio_features].fillna(0).to_numpy()
        X = np.hstack([X, X_audio])
    y = df['era']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return clf


def cluster_songs(lyric_embeddings, df, n_clusters=5):
    """Unsupervised clustering with visualization and analysis."""
    print("\n" + "=" * 80)
    print("CLUSTERING SONGS")
    print("=" * 80)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    # Try different numbers of clusters
    best_score = -1
    best_k = n_clusters
    scores = []

    for k in range(3, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lyric_embeddings)
        score = silhouette_score(lyric_embeddings, labels)
        scores.append((k, score))

        print(f"k={k}: Silhouette score = {score:.3f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n✓ Best number of clusters: {best_k} (score: {best_score:.3f})")

    # Fit final model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(lyric_embeddings)

    # Add to dataframe
    df['cluster'] = cluster_labels

    # Analyze clusters
    print("\n" + "-" * 80)
    print("CLUSTER ANALYSIS")
    print("-" * 80)

    for cluster_id in range(best_k):
        cluster_df = df[df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_df)} songs):")

        # Show sample songs
        sample_songs = cluster_df['Song_Name'].head(5).tolist()
        print(f"  Sample songs: {', '.join(sample_songs)}")

        # Show dominant era
        if 'era' in cluster_df.columns:
            era_dist = cluster_df['era'].value_counts()
            print(f"  Dominant era: {era_dist.index[0]} ({era_dist.iloc[0]} songs)")

        # Show avg audio features
        if 'energy' in cluster_df.columns:
            print(f"  Avg Energy: {cluster_df['energy'].mean():.2f}")
        if 'valence' in cluster_df.columns:
            print(f"  Avg Valence: {cluster_df['valence'].mean():.2f}")

    # Visualize clusters
    visualize_clusters(lyric_embeddings, cluster_labels, df)

    return cluster_labels, df


def visualize_clusters(embeddings, labels, df):
    """Visualize clusters in 2D."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))

    for cluster_id in range(len(set(labels))):
        mask = labels == cluster_id
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    label=f'Cluster {cluster_id}', alpha=0.6, s=50)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Song Clusters (PCA Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/clusters_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaved: results/clusters_visualization.png")
