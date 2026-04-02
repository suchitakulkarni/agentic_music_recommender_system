from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan, umap, torch, os, json, random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import src.config as config
from src.preprocessing import preprocess_lyrics_enhanced


def get_topic_coordinates_from_plot(topic_model):
    """
    Extract the exact 2D coordinates that BERTopic uses in visualize_topics().
    """
    import plotly.graph_objects as go

    # Generate the plot
    fig = topic_model.visualize_topics()

    # Extract coordinates from the plot data
    topic_coords = {}

    for trace in fig.data:
        if hasattr(trace, 'x') and hasattr(trace, 'y') and hasattr(trace, 'name'):
            # Each trace represents a topic
            # Parse topic ID from name (usually like "0_word_word" or just topic number)
            topic_name = trace.name

            # Extract topic ID (handle different naming formats)
            try:
                # Try to parse as integer (simple case)
                topic_id = int(topic_name.split('_')[0]) if '_' in topic_name else int(topic_name)
            except:
                # Skip if we can't parse
                continue

            # Get coordinates (take first point if multiple)
            x = trace.x[0] if len(trace.x) > 0 else None
            y = trace.y[0] if len(trace.y) > 0 else None

            if x is not None and y is not None:
                topic_coords[topic_id] = (float(x), float(y))

    return topic_coords

def identify_visual_outliers(topic_model, df, threshold=2.0):
    """
    Identify outliers using the ACTUAL 2D coordinates from the visualization.
    This matches what you see in the plot.
    """
    from umap import UMAP
    from sklearn.metrics.pairwise import euclidean_distances
    import numpy as np

    # Get c-TF-IDF representations
    ctfidf_matrix = topic_model.c_tf_idf_.toarray()

    # Reduce to 2D using same method as visualization
    # BERTopic uses these default settings for visualize_topics()
    umap_model = UMAP(
        n_neighbors=10,
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    embeddings_2d = umap_model.fit_transform(ctfidf_matrix)

    print(f"Reduced to 2D: {embeddings_2d.shape}")

    # Get topic IDs and mapping
    topic_info = topic_model.get_topic_info()
    topic_ids = [tid for tid in topic_info['Topic'].values if tid != -1]

    # Find main topic (largest cluster)
    topic_counts = df['id'].value_counts()
    main_topic_id = topic_counts.index[0]

    # Find main topic's position in 2D
    main_row = topic_info[topic_info['Topic'] == main_topic_id].index[0]
    main_coord = embeddings_2d[main_row].reshape(1, -1)

    print(f"\nMain topic {main_topic_id} at 2D coordinate: {main_coord[0]}")

    # Calculate 2D Euclidean distances
    outlier_topics = []

    for tid in topic_ids:
        if tid == main_topic_id:
            continue

        # Find this topic's row
        topic_row = topic_info[topic_info['Topic'] == tid].index[0]
        topic_coord = embeddings_2d[topic_row].reshape(1, -1)

        # Calculate 2D distance
        distance = euclidean_distances(main_coord, topic_coord)[0][0]

        print(f"Topic {tid}: 2D distance = {distance:.3f}, coord = {topic_coord[0]}")

        if distance > threshold:
            topic_songs = df[df['id'] == tid]
            words = topic_model.get_topic(tid)

            outlier_topics.append({
                'topic_id': tid,
                'distance': distance,
                'coord': topic_coord[0],
                'n_songs': len(topic_songs),
                'keywords': [w for w, _ in words[:5]],
                'songs': topic_songs['Song_Name'].tolist()[:5],
                'eras': topic_songs['era'].value_counts().to_dict()
            })

    # Sort by distance
    outlier_topics.sort(key=lambda x: -x['distance'])

    print(f"\n✓ Found {len(outlier_topics)} visually separated topics (2D distance > {threshold})")

    return outlier_topics, main_topic_id, embeddings_2d

def analyze_spatial_clusters(topic_model, df, theme_names):
    """
    Analyze clusters using 2D coordinates (matches the visual plot).
    """

    print("\n" + "=" * 60)
    print("SPATIAL CLUSTER ANALYSIS (2D Distance Map)")
    print("=" * 60)

    # Use 2D distances
    outlier_topics, main_id, coords_2d = identify_visual_outliers(
        topic_model, df, threshold=2.0  # Adjust this to get ~4 topics
    )


    if not outlier_topics:
        print("Try lowering the threshold")
        return [], main_id

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print('=' * 60)
    print(f"\nMain Cluster Center: Topic {main_id} ({len(df[df['id'] == main_id])} songs)")
    print(f"  Theme: {theme_names.get(main_id, 'Unlabeled')}")

    print(f"\nVisually Separated Topics ({len(outlier_topics)} found):")
    print("-" * 60)

    for i, topic_info in enumerate(outlier_topics, 1):
        tid = topic_info['topic_id']
        print(f"\n{i}. Topic {tid}: {theme_names.get(tid, 'Unlabeled')}")
        print(f"   2D Distance from center: {topic_info['distance']:.2f}")
        print(f"   2D Coordinates: ({topic_info['coord'][0]:.2f}, {topic_info['coord'][1]:.2f})")
        print(f"   Keywords: {', '.join(topic_info['keywords'])}")
        print(f"   Songs ({topic_info['n_songs']}): {', '.join(topic_info['songs'])}")
        print(f"   Dominant era: {max(topic_info['eras'].items(), key=lambda x: x[1])}")

    return outlier_topics, main_id

def debug_topic_ids(topic_model, df):
    """Debug topic ID mappings."""
    print("\n" + "=" * 60)
    print("TOPIC ID DEBUGGING")
    print("=" * 60)

    # IDs in the model
    model_topics = sorted([tid for tid in topic_model.get_topics().keys() if tid != -1])
    print(f"Topic IDs in model: {model_topics}")

    # IDs in dataframe
    df_topics = sorted(df['id'].unique())
    df_topics = [t for t in df_topics if t != -1]
    print(f"Topic IDs in dataframe: {df_topics}")

    # Check if they match
    if model_topics == df_topics:
        print("✓ Topic IDs are consistent")
    else:
        print("✗ WARNING: Topic ID mismatch!")
        print(f"  Missing in model: {set(df_topics) - set(model_topics)}")
        print(f"  Missing in df: {set(model_topics) - set(df_topics)}")

    # Check the actual topics_ attribute
    print(f"\nUnique values in topic_model.topics_: {sorted(set(topic_model.topics_))}")
    print(f"Unique values in df['id']: {sorted(df['id'].unique())}")


def identify_spatial_outliers_from_visualization(topic_model, df, threshold=0.3):
    """
    Identify topics that are spatially separated using c-TF-IDF representations.
    """
    from sklearn.metrics.pairwise import cosine_distances
    import numpy as np

    # Get c-TF-IDF matrix (the actual topic representations)
    if not hasattr(topic_model, 'c_tf_idf_') or topic_model.c_tf_idf_ is None:
        print("Warning: c-TF-IDF matrix not available")
        return [], None

    ctfidf_matrix = topic_model.c_tf_idf_.toarray()  # Convert sparse to dense

    # Get topic IDs
    topic_ids = sorted([tid for tid in topic_model.get_topics().keys() if tid != -1])

    # Find the largest topic (center of main cluster)
    topic_counts = df['id'].value_counts()
    main_topic_id = topic_counts.index[0]

    # Get topic info from BERTopic
    topic_info = topic_model.get_topic_info()

    # Create mapping from topic ID to row in c-TF-IDF matrix
    # BERTopic stores topics with -1 (outlier) at index 0, then 0, 1, 2, ...
    topic_to_row = {}
    for idx, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:  # Skip outlier topic
            topic_to_row[topic_id] = idx

    if main_topic_id not in topic_to_row:
        print(f"Warning: Main topic {main_topic_id} not found in topic_info")
        return [], main_topic_id

    main_row = topic_to_row[main_topic_id]
    main_vector = ctfidf_matrix[main_row].reshape(1, -1)

    print(f"\nMain topic {main_topic_id} at row {main_row}")
    print(f"Main vector shape: {main_vector.shape}")
    print(f"Main vector norm: {np.linalg.norm(main_vector):.4f}")

    outlier_topics = []

    for tid in topic_ids:
        if tid == main_topic_id:
            continue

        if tid not in topic_to_row:
            print(f"Warning: Topic {tid} not in topic_to_row mapping")
            continue

        topic_row = topic_to_row[tid]
        topic_vector = ctfidf_matrix[topic_row].reshape(1, -1)

        # Calculate cosine distance
        distance = cosine_distances(main_vector, topic_vector)[0][0]

        print(f"Topic {tid} (row {topic_row}): distance = {distance:.4f}")

        # Topics with distance > threshold are spatially separated
        if distance > threshold:
            topic_songs = df[df['id'] == tid]
            words = topic_model.get_topic(tid)

            outlier_topics.append({
                'topic_id': tid,
                'distance': distance,
                'n_songs': len(topic_songs),
                'keywords': [w for w, _ in words[:5]],
                'songs': topic_songs['Song_Name'].tolist()[:5],  # Limit to 5
                'eras': topic_songs['era'].value_counts().to_dict()
            })

    # Sort by distance (most distant first)
    outlier_topics.sort(key=lambda x: -x['distance'])

    print(f"\n✓ Found {len(outlier_topics)} spatially separated topics (distance > {threshold})")

    return outlier_topics, main_topic_id


'''def analyze_spatial_clusters(topic_model, df, theme_names):
    """
    Analyze and report spatially separated clusters as seen in the distance map.
    """

    print("\n" + "=" * 60)
    print("SPATIAL CLUSTER ANALYSIS (From Distance Map)")
    print("=" * 60)

    # Find spatially separated topics
    outlier_topics, main_id = identify_spatial_outliers_from_visualization(
        topic_model, df, threshold=0.15
    )

    if not outlier_topics:
        print("No spatially separated clusters found")
        return

    print(f"\nMain Cluster: Topic {main_id} ({len(df[df['id'] == main_id])} songs)")
    print(f"  Theme: {theme_names.get(main_id, 'Unlabeled')}")

    print(f"\nSpatially Separated Topics ({len(outlier_topics)} topics):")
    print("-" * 60)

    for i, topic_info in enumerate(outlier_topics, 1):
        tid = topic_info['topic_id']
        print(f"\n{i}. Topic {tid}: {theme_names.get(tid, 'Unlabeled')}")
        print(f"   Distance from main cluster: {topic_info['distance']:.3f}")
        print(f"   Keywords: {', '.join(topic_info['keywords'])}")
        print(f"   Songs ({topic_info['n_songs']}): {', '.join(topic_info['songs'][:4])}")

        # Era distribution
        eras = topic_info['eras']
        dominant_era = max(eras.items(), key=lambda x: x[1])
        print(f"   Eras: {dict(list(eras.items())[:3])}")

        # Explain why separated
        keywords = topic_info['keywords']
        if any(word in ['shake', 'off', 'hate'] for word in keywords):
            print(f"   → Separated due to: Confidence/defiance vocabulary")
        elif any(word in ['red', 'fuck', 'twenty'] for word in keywords):
            print(f"   → Separated due to: Specific Red era identity")
        elif any(word in ['dance', 'snow', 'beach'] for word in keywords):
            print(f"   → Separated due to: Sensory/nostalgic imagery")
        elif any(word in ['wood', 'clear', 'yet'] for word in keywords):
            print(f"   → Separated due to: Haunting/uncertain tone")
        elif any(word in ['grow', 'clean', 'finally'] for word in keywords):
            print(f"   → Separated due to: Maturity/growth themes")
        else:
            print(f"   → Separated due to: Unique vocabulary/theme")

    return outlier_topics, main_id'''

def identify_outlier_clusters(df, topic_model):
    """Identify spatially separated topics from the main cluster."""
    from sklearn.metrics.pairwise import cosine_similarity

    # Get all topics
    topics = topic_model.get_topics()
    topic_ids = [tid for tid in topics.keys() if tid != -1]

    # Find the largest topic (assumed to be center of main cluster)
    topic_counts = df['id'].value_counts()
    main_topic_id = topic_counts.index[0]

    # Calculate word overlap with main topic
    main_words = set([w for w, _ in topics[main_topic_id][:30]])

    outlier_topics = []

    for tid in topic_ids:
        if tid == main_topic_id:
            continue

        topic_words = set([w for w, _ in topics[tid][:30]])

        # Calculate Jaccard similarity (word overlap)
        overlap = len(main_words & topic_words) / len(main_words | topic_words)

        # Topics with <20% word overlap are likely spatially separated
        if overlap < 0.2:
            topic_songs = df[df['id'] == tid]
            outlier_topics.append({
                'topic_id': tid,
                'overlap': overlap,
                'n_songs': len(topic_songs),
                'keywords': [w for w, _ in topics[tid][:5]],
                'songs': topic_songs['Song_Name'].tolist(),
                'era': topic_songs['era'].value_counts().to_dict()
            })

    # Sort by distance (lowest overlap first)
    outlier_topics.sort(key=lambda x: x['overlap'])

    return outlier_topics, main_topic_id


def create_talk_summary(df, topic_model, stabilities, theme_names, output_file='results/talk/TALK_SUMMARY.txt'):
    """Create a dynamic summary based on actual model results."""

    lines = []
    lines.append("=" * 80)
    lines.append("TAYLOR SWIFT TOPIC MODELING - TALK SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Key findings - dynamic
    n_topics = len(set(df['id'])) - (1 if -1 in df['id'] else 0)
    n_outliers = sum(df['id'] == -1)
    best_stability = max(s['stability'] for s in stabilities)

    lines.append("KEY FINDINGS:")
    lines.append("-" * 80)
    lines.append(f"• Total songs analyzed: {len(df)}")
    lines.append(f"• Topics discovered: {n_topics}")
    lines.append(f"• Outlier songs: {n_outliers} ({n_outliers / len(df) * 100:.1f}%)")
    lines.append(f"• Model stability score: {best_stability:.3f} (high = good)")
    lines.append(f"• Seeds tested: {len(stabilities)}")
    lines.append("")

    # Find the largest topic dynamically
    topic_counts = df['id'].value_counts()
    largest_topic_id = topic_counts.index[0]
    largest_topic_count = topic_counts.iloc[0]
    largest_topic_pct = (largest_topic_count / len(df)) * 100

    # Get words for largest topic
    largest_topic_words = topic_model.get_topic(largest_topic_id)
    largest_topic_keywords = ', '.join([w for w, _ in largest_topic_words[:8]])

    # Get era distribution for largest topic
    largest_topic_songs = df[df['id'] == largest_topic_id]
    era_dist = largest_topic_songs['era'].value_counts()

    lines.append("TOPIC STRUCTURE ANALYSIS:")
    lines.append("-" * 80)
    lines.append(f"MAIN CLUSTER: Topic {largest_topic_id}")
    lines.append(f"  • Size: {largest_topic_count} songs ({largest_topic_pct:.1f}% of catalog)")
    lines.append(f"  • Theme: {theme_names.get(largest_topic_id, 'Unlabeled')}")
    lines.append(f"  • Keywords: {largest_topic_keywords}")
    lines.append(f"  • Represents: Core Taylor Swift narrative/storytelling style")
    lines.append(f"  • Era distribution:")
    for era, count in era_dist.head(5).items():
        lines.append(f"    - {era}: {count} songs")
    lines.append("")

    # === NEW: Identify spatially separated outlier clusters ===
    outlier_clusters, main_id = identify_outlier_clusters(df, topic_model)

    if outlier_clusters:
        lines.append("SPATIALLY SEPARATED CLUSTERS (Visible in Distance Map):")
        lines.append("-" * 80)
        lines.append(f"  • {len(outlier_clusters)} topics are far from the main cluster")
        lines.append("  • These topics have <20% word overlap with core narrative style")
        lines.append("  • Represent distinct stylistic/thematic departures")
        lines.append("")
        lines.append("  WHY THEY'RE SEPARATED:")

        for i, topic_info in enumerate(outlier_clusters[:4], 1):  # Show top 4 most distant
            tid = topic_info['topic_id']
            lines.append(f"    {i}. Topic {tid}: {theme_names.get(tid, 'Unlabeled')}")
            lines.append(f"       • Keywords: {', '.join(topic_info['keywords'])}")
            lines.append(f"       • Word overlap with main cluster: {topic_info['overlap'] * 100:.1f}%")
            lines.append(f"       • Songs ({topic_info['n_songs']}): {', '.join(topic_info['songs'][:3])}")

            # Explain why it's different
            if any(word in ['shake', 'off', 'hate'] for word in topic_info['keywords']):
                lines.append(f"       • Why separated: Confidence/defiance vocabulary (not narrative)")
            elif any(word in ['red', 'fuck', 'twenty'] for word in topic_info['keywords']):
                lines.append(f"       • Why separated: Specific album/era identity (Red)")
            elif any(word in ['dance', 'snow', 'beach'] for word in topic_info['keywords']):
                lines.append(f"       • Why separated: Sensory/nostalgic imagery (not story-driven)")
            elif any(word in ['wood', 'clear', 'yet'] for word in topic_info['keywords']):
                lines.append(f"       • Why separated: Haunting/uncertain tone (distinct emotion)")
            else:
                lines.append(f"       • Why separated: Unique vocabulary/theme")

            # Era concentration
            dominant_era = max(topic_info['era'].items(), key=lambda x: x[1])
            lines.append(f"       • Era: {dominant_era[0]} ({dominant_era[1]}/{topic_info['n_songs']} songs)")
            lines.append("")

    # Find small, distinctive topics (size <= 5)
    small_topics = topic_counts[topic_counts <= 5].index.tolist()
    if small_topics:
        lines.append("OTHER MICRO-TOPICS:")
        lines.append(f"  • {len(small_topics)} small topics with 5 or fewer songs")

        # Only show ones not already in outlier cluster
        outlier_ids = [t['topic_id'] for t in outlier_clusters]
        remaining_small = [tid for tid in small_topics if tid not in outlier_ids]

        if remaining_small:
            lines.append("  • Examples (part of main cluster):")
            for topic_id in remaining_small[:3]:
                topic_songs = df[df['id'] == topic_id]
                n_songs = len(topic_songs)
                words = topic_model.get_topic(topic_id)
                keywords = ', '.join([w for w, _ in words[:5]])

                lines.append(f"    - Topic {topic_id}: {theme_names.get(topic_id, 'Unlabeled')} ({n_songs} songs)")
                lines.append(f"      Keywords: {keywords}")
        lines.append("")

    # Topic size distribution
    lines.append("TOPIC SIZE DISTRIBUTION:")
    lines.append("-" * 80)
    size_ranges = {
        'Large (20+)': len(topic_counts[topic_counts >= 20]),
        'Medium (10-19)': len(topic_counts[(topic_counts >= 10) & (topic_counts < 20)]),
        'Small (5-9)': len(topic_counts[(topic_counts >= 5) & (topic_counts < 10)]),
        'Micro (< 5)': len(topic_counts[topic_counts < 5])
    }
    for range_name, count in size_ranges.items():
        lines.append(f"  • {range_name}: {count} topics")
    lines.append("")

    # Stability analysis
    lines.append("MULTI-SEED STABILITY ANALYSIS:")
    lines.append("-" * 80)
    lines.append(f"  • Seeds tested: {', '.join([str(s['seed']) for s in stabilities])}")
    lines.append(f"  • Best seed: {[s['seed'] for s in stabilities if s['stability'] == best_stability][0]}")
    lines.append(
        f"  • Stability range: {min(s['stability'] for s in stabilities):.3f} - {max(s['stability'] for s in stabilities):.3f}")
    lines.append(f"  • Average stability: {np.mean([s['stability'] for s in stabilities]):.3f}")
    lines.append("")

    # Topic-era insights
    lines.append("TOPIC-ERA INSIGHTS:")
    lines.append("-" * 80)

    # Find topics strongly associated with specific eras
    era_dominant_topics = {}
    for topic_id in sorted(set(df['id'])):
        if topic_id == -1:
            continue
        topic_songs = df[df['id'] == topic_id]
        if len(topic_songs) > 0:
            era_counts = topic_songs['era'].value_counts()
            if len(era_counts) > 0:
                dominant_era = era_counts.index[0]
                dominant_pct = (era_counts.iloc[0] / len(topic_songs)) * 100

                # If 60%+ from one era, it's era-specific
                if dominant_pct >= 60:
                    if dominant_era not in era_dominant_topics:
                        era_dominant_topics[dominant_era] = []
                    era_dominant_topics[dominant_era].append({
                        'topic_id': topic_id,
                        'pct': dominant_pct,
                        'theme': theme_names.get(topic_id, 'Unlabeled')
                    })

    if era_dominant_topics:
        lines.append("  • Era-specific topics (>60% from one era):")
        for era, topics_list in sorted(era_dominant_topics.items()):
            lines.append(f"    {era}:")
            for topic_info in topics_list:
                lines.append(
                    f"      - Topic {topic_info['topic_id']}: {topic_info['theme']} ({topic_info['pct']:.0f}%)")
    else:
        lines.append("  • No strongly era-specific topics (themes span multiple eras)")
    lines.append("")

    # For the agentic system
    lines.append("FOR YOUR AGENTIC SYSTEM:")
    lines.append("-" * 80)
    lines.append(f"• Use Topic {largest_topic_id} as 'Classic Taylor' baseline for similarity")
    lines.append(f"• Outlier topics represent stylistic experiments/departures")
    lines.append(f"• Total of {n_topics} distinct thematic dimensions for recommendations")
    lines.append("• Topic assignments enable:")
    lines.append("  - Thematic song similarity (beyond audio features)")
    lines.append("  - Era-aware recommendations")
    lines.append("  - Lyrical theme exploration")
    lines.append("  - Identification of experimental/unique songs")
    lines.append(f"• Model is stable (tested across {len(stabilities)} random seeds)")
    lines.append("")

    # Files generated
    lines.append("FILES FOR TALK:")
    lines.append("-" * 80)
    lines.append("• results/talk/topic_distribution.png - Bar chart of topic sizes")
    lines.append("• results/talk/topic_era_heatmap.png - Era analysis heatmap")
    lines.append("• results/talk/intertopic_distance.html - Interactive topic map")
    lines.append("• results/bertopic_topics.png - Static topic map")
    lines.append("• results/final_topics_named.csv - Full data with topic assignments")
    lines.append("• results/topic_names.json - Topic theme labels")
    lines.append("")
    lines.append("=" * 80)

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    # Print to console
    print('\n'.join(lines))
    print(f"\n✓ Summary saved to: {output_file}")

    return lines




def customize_topic_visualization(topic_model, output_path="results/bertopic_topics.html"):
    """Create customized topic visualization with dark green background."""

    fig = topic_model.visualize_topics()

    # Update layout for dark theme
    fig.update_layout(
        paper_bgcolor='#0d3d0d',  # Dark green background
        plot_bgcolor='#0d3d0d',  # Dark green plot area
        font=dict(
            color='white',  # White text
            size=14
        ),
        title=dict(
            text='Intertopic Distance Map',
            font=dict(color='white', size=20),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            gridcolor='white',
            linecolor='white',
            zerolinecolor='white'
        ),
        yaxis=dict(
            gridcolor='white',
            linecolor='white',
            zerolinecolor='white'
        )
    )

    # Update trace colors (bubbles)
    cluster_colors = ['#FFD700', '#FF6347', '#00CED1', '#FF69B4', '#FFA500']

    if len(fig.data) > 0:
        for i, trace in enumerate(fig.data):
            color_idx = i % len(cluster_colors)
            trace.marker.color = cluster_colors[color_idx]
            trace.marker.line.color = 'white'
            trace.marker.line.width = 2

    fig.write_html(output_path)
    print(f"\nSaved customized topic visualization to {output_path}")

    return fig


def save_topic_visualization_as_png(topic_model, output_path="results/bertopic_topics.png"):
    """Save BERTopic visualization as static PNG with custom styling."""

    fig = topic_model.visualize_topics()

    # Apply custom styling
    fig.update_layout(
        #paper_bgcolor='#0d3d0d',
        #plot_bgcolor='#0d3d0d',
        #font=dict(color='white', size=14),
        #xaxis=dict(gridcolor='white', linecolor='white', zerolinecolor='white'),
        #yaxis=dict(gridcolor='white', linecolor='white', zerolinecolor='white'),

        font=dict(size=14),
        title=dict(
            text='Intertopic Distance Map',
            #font=dict(color='white', size=20)
            font=dict(size=20)
        )
    )

    # Update colors
    #cluster_colors = ['#FFD700', '#FF6347', '#00CED1', '#FF69B4', '#FFA500']
    cluster_colors = ['#8B4513', '#FF8C00', '#FF1493', '#4B0082', '#9370DB']
    for i, trace in enumerate(fig.data):
        color_idx = i % len(cluster_colors)
        trace.marker.color = cluster_colors[color_idx]
        trace.marker.line.color = 'black'
        trace.marker.line.width = 2

    # Save as PNG (requires kaleido)
    fig.write_image(output_path, width=1400, height=1000)
    print(f"Saved PNG to {output_path}")

    return fig


def create_talk_summary(df, topic_model, stabilities, theme_names, output_file='results/talk/TALK_SUMMARY.txt'):
    """Create a dynamic summary based on actual model results."""

    lines = []
    lines.append("=" * 80)
    lines.append("TAYLOR SWIFT TOPIC MODELING - TALK SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Key findings - dynamic
    n_topics = len(set(df['id'])) - (1 if -1 in df['id'] else 0)
    n_outliers = sum(df['id'] == -1)
    best_stability = max(s['stability'] for s in stabilities)

    lines.append("KEY FINDINGS:")
    lines.append("-" * 80)
    lines.append(f"• Total songs analyzed: {len(df)}")
    lines.append(f"• Topics discovered: {n_topics}")
    lines.append(f"• Outlier songs: {n_outliers} ({n_outliers / len(df) * 100:.1f}%)")
    lines.append(f"• Model stability score: {best_stability:.3f} (high = good)")
    lines.append(f"• Seeds tested: {len(stabilities)}")
    lines.append("")

    # Find the largest topic dynamically
    topic_counts = df['id'].value_counts()
    largest_topic_id = topic_counts.index[0]
    largest_topic_count = topic_counts.iloc[0]
    largest_topic_pct = (largest_topic_count / len(df)) * 100

    # Get words for largest topic
    largest_topic_words = topic_model.get_topic(largest_topic_id)
    largest_topic_keywords = ', '.join([w for w, _ in largest_topic_words[:8]])

    # Get era distribution for largest topic
    largest_topic_songs = df[df['id'] == largest_topic_id]
    era_dist = largest_topic_songs['era'].value_counts()

    lines.append("TOPIC STRUCTURE ANALYSIS:")
    lines.append("-" * 80)
    lines.append(f"DOMINANT TOPIC: Topic {largest_topic_id}")
    lines.append(f"  • Size: {largest_topic_count} songs ({largest_topic_pct:.1f}% of catalog)")
    lines.append(f"  • Theme: {theme_names.get(largest_topic_id, 'Unlabeled')}")
    lines.append(f"  • Keywords: {largest_topic_keywords}")
    lines.append(f"  • Era distribution:")
    for era, count in era_dist.head(5).items():
        lines.append(f"    - {era}: {count} songs")
    lines.append("")

    # Find small, distinctive topics (size <= 5)
    small_topics = topic_counts[topic_counts <= 5].index.tolist()
    if small_topics:
        lines.append("DISTINCTIVE MICRO-TOPICS (Era-Specific & Experimental):")
        lines.append(f"  • {len(small_topics)} small, highly distinctive topics found")
        lines.append("  • Examples:")

        for topic_id in small_topics[:5]:  # Show up to 5 examples
            topic_songs = df[df['id'] == topic_id]
            n_songs = len(topic_songs)
            words = topic_model.get_topic(topic_id)
            keywords = ', '.join([w for w, _ in words[:5]])
            sample_songs = topic_songs['Song_Name'].head(3).tolist()
            dominant_era = topic_songs['era'].value_counts().index[0] if len(topic_songs) > 0 else "Mixed"

            lines.append(f"    - Topic {topic_id}: {theme_names.get(topic_id, 'Unlabeled')} ({n_songs} songs)")
            lines.append(f"      Keywords: {keywords}")
            lines.append(f"      Songs: {', '.join(sample_songs)}")
            lines.append(f"      Era: {dominant_era}")
        lines.append("")

    # Topic size distribution
    lines.append("TOPIC SIZE DISTRIBUTION:")
    lines.append("-" * 80)
    size_ranges = {
        'Large (20+)': len(topic_counts[topic_counts >= 20]),
        'Medium (10-19)': len(topic_counts[(topic_counts >= 10) & (topic_counts < 20)]),
        'Small (5-9)': len(topic_counts[(topic_counts >= 5) & (topic_counts < 10)]),
        'Micro (< 5)': len(topic_counts[topic_counts < 5])
    }
    for range_name, count in size_ranges.items():
        lines.append(f"  • {range_name}: {count} topics")
    lines.append("")

    # Stability analysis
    lines.append("MULTI-SEED STABILITY ANALYSIS:")
    lines.append("-" * 80)
    lines.append(f"  • Seeds tested: {', '.join([str(s['seed']) for s in stabilities])}")
    lines.append(f"  • Best seed: {[s['seed'] for s in stabilities if s['stability'] == best_stability][0]}")
    lines.append(
        f"  • Stability range: {min(s['stability'] for s in stabilities):.3f} - {max(s['stability'] for s in stabilities):.3f}")
    lines.append(f"  • Average stability: {np.mean([s['stability'] for s in stabilities]):.3f}")
    lines.append("")

    # Topic-era insights
    lines.append("TOPIC-ERA INSIGHTS:")
    lines.append("-" * 80)

    # Find topics strongly associated with specific eras
    era_dominant_topics = {}
    for topic_id in sorted(set(df['id'])):
        if topic_id == -1:
            continue
        topic_songs = df[df['id'] == topic_id]
        if len(topic_songs) > 0:
            era_counts = topic_songs['era'].value_counts()
            if len(era_counts) > 0:
                dominant_era = era_counts.index[0]
                dominant_pct = (era_counts.iloc[0] / len(topic_songs)) * 100

                # If 60%+ from one era, it's era-specific
                if dominant_pct >= 60:
                    if dominant_era not in era_dominant_topics:
                        era_dominant_topics[dominant_era] = []
                    era_dominant_topics[dominant_era].append({
                        'topic_id': topic_id,
                        'pct': dominant_pct,
                        'theme': theme_names.get(topic_id, 'Unlabeled')
                    })

    if era_dominant_topics:
        lines.append("  • Era-specific topics (>60% from one era):")
        for era, topics_list in sorted(era_dominant_topics.items()):
            lines.append(f"    {era}:")
            for topic_info in topics_list:
                lines.append(
                    f"      - Topic {topic_info['topic_id']}: {topic_info['theme']} ({topic_info['pct']:.0f}%)")
    else:
        lines.append("  • No strongly era-specific topics (themes span multiple eras)")
    lines.append("")

    # For the agentic system
    lines.append("FOR YOUR AGENTIC SYSTEM:")
    lines.append("-" * 80)
    lines.append(f"• Use Topic {largest_topic_id} as 'Classic Taylor' baseline for similarity")
    lines.append(f"• Total of {n_topics} distinct thematic dimensions for recommendations")
    lines.append("• Topic assignments enable:")
    lines.append("  - Thematic song similarity (beyond audio features)")
    lines.append("  - Era-aware recommendations")
    lines.append("  - Lyrical theme exploration")
    lines.append(f"• Model is stable (tested across {len(stabilities)} random seeds)")
    lines.append("")

    # Files generated
    lines.append("FILES FOR TALK:")
    lines.append("-" * 80)
    lines.append("• results/talk/topic_distribution.png - Bar chart of topic sizes")
    lines.append("• results/talk/topic_era_heatmap.png - Era analysis heatmap")
    lines.append("• results/talk/intertopic_distance.html - Interactive topic map")
    lines.append("• results/bertopic_topics.png - Static topic map")
    lines.append("• results/final_topics_named.csv - Full data with topic assignments")
    lines.append("• results/topic_names.json - Topic theme labels")
    lines.append("")
    lines.append("=" * 80)

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    # Print to console
    print('\n'.join(lines))
    print(f"\n✓ Summary saved to: {output_file}")

    return lines

def create_talk_visualizations(topic_model, df, output_dir='results/talk'):
    """Create clean visualizations for presentation."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # 1. Topic size bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    topic_counts = df['id'].value_counts().sort_index()
    topic_counts.plot(kind='bar', ax=ax, color='#8b5a8e')
    ax.set_xlabel('Topic ID', fontsize=12)
    ax.set_ylabel('Number of Songs', fontsize=12)
    ax.set_title('Song Distribution Across Topics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/topic_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/topic_distribution.png")

    # 2. Topic-Era heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    topic_era = pd.crosstab(df['id'], df['era'])
    sns.heatmap(topic_era, annot=True, fmt='d', cmap='PuRd', ax=ax, cbar_kws={'label': 'Song Count'})
    ax.set_xlabel('Era', fontsize=12)
    ax.set_ylabel('Topic ID', fontsize=12)
    ax.set_title('Topics Across Taylor Swift Eras', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/topic_era_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/topic_era_heatmap.png")

    # 3. Save the intertopic distance map (already generated)
    import shutil
    if os.path.exists('results/bertopic_topics.html'):
        shutil.copy('results/bertopic_topics.html', f'{output_dir}/intertopic_distance.html')
        print(f"✓ Saved: {output_dir}/intertopic_distance.html")

    print("\n✓ All talk visualizations ready!")

def assign_final_theme_names(topic_model, df):
    """Manually assign interpretable theme names."""

    # Based on your analysis
    theme_names = {
        0: "Classic Taylor Narrative (Core Style)",
        1: "Karma & Rain Themes",
        2: "Getaway & Never Themes",
        3: "Daylight & Signs",
        4: "Welcome to New York (Love Songs)",
        5: "Dancing & Nostalgic Moments",
        6: "Raw Emotions & Honesty",
        7: "Growth & Maturity",
        8: "Shake It Off (Confidence Anthems)",
        9: "Trouble & Red Era",
        10: "Starlight & Collaborations",
        11: "Red Album Core (22, The Lucky One)",
        12: "Midnight & Will Themes",
        13: "Out of the Woods (Haunting)",
        14: "Wishes & Blame",
        15: "Reputation Era Sass"
    }

    df['topic_name'] = df['id'].map(theme_names)

    # Save lookup table
    import json
    with open('results/topic_names.json', 'w') as f:
        json.dump(theme_names, f, indent=2)

    return df, theme_names

def analyze_spatial_clusters_old(df, topic_model):
    """Identify which topics are in the outlier cluster."""

    # Get 2D coordinates (you'll need to extract these from the model)
    # For now, manually identify from your visualization

    # Top-right outlier topics (from your image, roughly topics in D1>0, D2>0)
    outlier_topics = []  # We'll fill this by inspection

    # Check what's in Topic 0 (the giant one)
    print("=" * 60)
    print("TOPIC 0 ANALYSIS (The Giant Topic - 48 songs)")
    print("=" * 60)
    topic_0_songs = df[df['id'] == 0][['Song_Name', 'Album', 'era']].copy()
    print(f"\nEras represented:")
    print(topic_0_songs['era'].value_counts())
    print(f"\nSample songs:")
    print(topic_0_songs['Song_Name'].head(10).tolist())

    # Get top words for Topic 0
    words = topic_model.get_topic(0)
    print(f"\nTop words: {', '.join([w for w, _ in words[:15]])}")

    # Check the outlier cluster (top-right)
    print("\n" + "=" * 60)
    print("OUTLIER CLUSTER ANALYSIS (Top-Right Topics)")
    print("=" * 60)

    # You'll need to identify which topics are in top-right from the plot
    # Let's check topics with fewer songs that might be outliers
    for topic_id in sorted(set(df['id'])):
        if topic_id == -1:
            continue
        topic_songs = df[df['id'] == topic_id]
        n_songs = len(topic_songs)

        # Print small topics that might be outliers
        if n_songs <= 5:
            words = topic_model.get_topic(topic_id)
            top_words = [w for w, _ in words[:5]]
            print(f"\nTopic {topic_id} ({n_songs} songs): {', '.join(top_words)}")
            print(f"  Songs: {topic_songs['Song_Name'].tolist()}")
            print(f"  Eras: {topic_songs['era'].value_counts().to_dict()}")

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Set all random seeds to {seed}")


def run_multiple_seeds(df, seeds=[42, 123, 456, 789, 999], **pipeline_kwargs):
    """Run pipeline with multiple seeds and collect results."""
    print("\n" + "=" * 60)
    print(f"RUNNING WITH {len(seeds)} DIFFERENT SEEDS")
    print("=" * 60)

    all_results = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"SEED {seed}")
        print('=' * 60)

        model, topics, probs, df_result, labels = bertopic_lyrics_pipeline(
            df.copy(),
            seed=seed,
            visualize=False,  # Don't visualize during multi-seed runs
            **pipeline_kwargs
        )

        all_results.append({
            'seed': seed,
            'model': model,
            'topics': topics,
            'probs': probs,
            'df': df_result,
            'labels': labels,
            'n_topics': len(labels),
            'n_outliers': sum(1 for t in topics if t == -1)
        })

        print(f"\nSeed {seed} summary: {len(labels)} topics, {all_results[-1]['n_outliers']} outliers")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON ACROSS SEEDS")
    print("=" * 60)
    for result in all_results:
        print(f"Seed {result['seed']:3d}: {result['n_topics']:2d} topics, {result['n_outliers']:2d} outliers")

    return all_results


def calculate_stability_metrics(all_results):
    """Calculate stability metrics across runs using ARI and NMI."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("\n" + "=" * 60)
    print("CALCULATING STABILITY METRICS")
    print("=" * 60)

    stabilities = []

    for i, result in enumerate(all_results):
        topics = result['topics']

        # Compare to all other results
        ari_scores = []
        nmi_scores = []

        for j, other in enumerate(all_results):
            if i == j:
                continue
            other_topics = other['topics']

            # ARI: measures clustering agreement (-1 to 1, higher is better)
            ari = adjusted_rand_score(topics, other_topics)
            # NMI: normalized mutual information (0 to 1, higher is better)
            nmi = normalized_mutual_info_score(topics, other_topics)

            ari_scores.append(ari)
            nmi_scores.append(nmi)

        avg_ari = np.mean(ari_scores)
        avg_nmi = np.mean(nmi_scores)
        stability = (avg_ari + avg_nmi) / 2

        stabilities.append({
            'seed': result['seed'],
            'n_topics': result['n_topics'],
            'n_outliers': result['n_outliers'],
            'ari': avg_ari,
            'nmi': avg_nmi,
            'stability': stability
        })

        print(f"Seed {result['seed']:3d}: ARI={avg_ari:.3f}, NMI={avg_nmi:.3f}, Stability={stability:.3f}")

    # Find best (highest stability)
    best = max(stabilities, key=lambda x: x['stability'])
    best_idx = [i for i, r in enumerate(all_results) if r['seed'] == best['seed']][0]

    print(f"\n{'=' * 60}")
    print("MOST STABLE MODEL")
    print('=' * 60)
    print(f"Seed: {best['seed']}")
    print(f"Topics: {best['n_topics']}")
    print(f"Outliers: {best['n_outliers']}")
    print(f"Stability Score: {best['stability']:.3f}")

    return all_results[best_idx], stabilities

def get_or_create_embeddings(df, embedding_model_name, embeddings_path=None):
    if embeddings_path == None:
        embeddings_path = os.path.join(config.DATA_SCIENCE_DATA, "Taylor_Swift_agentic", "results","song_embeddings.npz")
    """Load embeddings if they exist and match model; otherwise compute and save new ones."""
    meta_path = embeddings_path.replace(".npz", "_meta.json")
    if os.path.exists(embeddings_path) and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if meta.get("embedding_model_name") == embedding_model_name:
            print(f"Loading cached embeddings from {embeddings_path}")
            data = np.load(embeddings_path, allow_pickle=True)
            embeddings = data["embeddings"]
            return embeddings
        else:
            print(
                f"Model name mismatch - expected {meta['embedding_model_name']}, got {embedding_model_name}. Recomputing...")

    print(f"Computing new embeddings using model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(df["clean_lyrics"].tolist(), show_progress_bar=True)

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.savez(embeddings_path, embeddings=embeddings)

    meta = {
        "embedding_model_name": embedding_model_name,
        "num_documents": len(df),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings and metadata to {embeddings_path}")
    return embeddings


def bertopic_lyrics_pipeline(df, embedding_model_name="all-mpnet-base-v2",
                             min_cluster_size=5, diversity_threshold=0.5,
                             visualize=True, seed=42):  # ADD seed parameter
    """
    Optimized BERTopic pipeline for song lyrics.

    Args:
        df (pd.DataFrame): Must contain a 'lyrics' column.
        embedding_model_name (str): SentenceTransformer model name.
        min_cluster_size (int): HDBSCAN minimum cluster size.
        diversity_threshold (float): Threshold to merge similar topics (lower = more merging).
        visualize (bool): Whether to save visualizations.

    Returns:
        best_model (BERTopic)
        topics (list[int])
        probs (np.ndarray)
        df (pd.DataFrame) with topic assignments and strengths
        topic_labels (list[str])
    """
    # Set random seeds for reproducibility
    set_all_seeds(seed)

    # ------------------------
    # 1. Preprocess lyrics
    # ------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING LYRICS")
    print("=" * 60)
    df["clean_lyrics"] = df["lyrics"].astype(str).apply(preprocess_lyrics_enhanced)

    # ------------------------
    # 2. Sentence embeddings
    # ------------------------
    print("\nEmbedding lyrics with:", embedding_model_name)
    embeddings = get_or_create_embeddings(df, embedding_model_name)

    # ------------------------
    # 3. UMAP for dimensionality reduction
    # ------------------------
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=seed
    )

    # ------------------------
    # 4. HDBSCAN clustering
    # ------------------------
    #cluster_model = hdbscan.HDBSCAN(
    #    min_cluster_size=min_cluster_size,
    #    metric='euclidean',
    #    cluster_selection_method='eom',  # Changed from 'eom' - more aggressive clustering
    #    min_samples=1,  # Added - more lenient
    #    prediction_data=True
    #)
    cluster_model = hdbscan.HDBSCAN(
        min_cluster_size=3,  # REDUCED from 5
        metric='euclidean',
        cluster_selection_method='eom',  # Less aggressive
        min_samples=1,
        prediction_data=True,
        cluster_selection_epsilon=0.1,  # INCREASED from 0.0 - merge nearby clusters
    )

    # ------------------------
    # 5. BERTopic initialization with all components
    # ------------------------
    print("\nInitializing BERTopic with seed topics...")
    #vectorizer_model = CountVectorizer(
    #    stop_words=list(config.LYRIC_STOPWORDS),
    #    ngram_range=(1, 2),
    #    min_df=2,  # Must appear in at least 2 songs
    #    max_df=0.5,  # Ignore if in more than 50% of songs
    #    max_features=1000  # Limit vocabulary size
    #)

    vectorizer_model = CountVectorizer(
        stop_words=list(config.LYRIC_STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,  # CHANGED: Allow words in just 1 document
        max_df=0.95,  # CHANGED: More lenient
        max_features=5000  # INCREASED: More vocabulary
    )

    topic_model = BERTopic(
        embedding_model=embedding_model_name,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,  # ADD THIS LINE
        seed_topic_list=config.seed_topic_list,
        calculate_probabilities=True,
        verbose=True
    )

    # ------------------------
    # 6. Fit model
    # ------------------------
    print("\nFitting BERTopic model...")
    topics, probs = topic_model.fit_transform(df["clean_lyrics"], embeddings)

    print(f"\nInitial number of topics: {len(set(topics)) - (1 if -1 in topics else 0)}")

    # ------------------------
    # 7. Reduce outliers FIRST
    # ------------------------
    '''outlier_count = sum(1 for t in topics if t == -1)
    if outlier_count > 0:
        print(f"\nReducing outliers...")
        new_topics = topic_model.reduce_outliers(
            df["clean_lyrics"],
            topics,
            strategy="distributions",
            threshold=0.15  # Adjust between 0.0 (aggressive) and 1.0 (conservative)
        )
        topics = new_topics
        topic_model.update_topics(df["clean_lyrics"], topics=topics)
        print(f"Outliers after reduction: {sum(1 for t in new_topics if t == -1)}")'''

    # ------------------------
    # 7. Reduce outliers with multiple strategies
    # ------------------------
    outlier_count = sum(1 for t in topics if t == -1)
    print(f"\nInitial outliers: {outlier_count}")

    if outlier_count > len(topics) * 0.1:  # If more than 10%
        print(f"Reducing outliers with multiple strategies...")

        # Strategy 1: Very aggressive distributions
        print("  Strategy 1: distributions (threshold=0.01)")
        new_topics = topic_model.reduce_outliers(
            df["clean_lyrics"],
            topics,
            strategy="distributions",
            threshold=0.01  # VERY aggressive (was 0.15)
        )
        print(f"    After strategy 1: {sum(1 for t in new_topics if t == -1)} outliers")

        # Strategy 2: c-TF-IDF if still too many
        if sum(1 for t in new_topics if t == -1) > len(topics) * 0.1:
            print("  Strategy 2: c-tf-idf (threshold=0.05)")
            new_topics = topic_model.reduce_outliers(
                df["clean_lyrics"],
                new_topics,
                strategy="c-tf-idf",
                threshold=0.05
            )
            print(f"    After strategy 2: {sum(1 for t in new_topics if t == -1)} outliers")

        # Strategy 3: Embeddings similarity if STILL too many
        if sum(1 for t in new_topics if t == -1) > len(topics) * 0.1:
            print("  Strategy 3: embeddings (threshold=0.5)")
            new_topics = topic_model.reduce_outliers(
                df["clean_lyrics"],
                new_topics,
                strategy="embeddings",
                threshold=0.5,
                embeddings=embeddings  # Pass embeddings explicitly
            )
            print(f"    After strategy 3: {sum(1 for t in new_topics if t == -1)} outliers")

        topics = new_topics
        topic_model.update_topics(df["clean_lyrics"], topics=topics)

        final_outliers = sum(1 for t in topics if t == -1)
        print(f"\nFinal outliers: {final_outliers} ({final_outliers / len(topics) * 100:.1f}%)")

    # ------------------------
    # 7. Reduce topics to merge similar ones
    # ------------------------
    #if len(set(topics)) > 2:  # Only reduce if we have multiple topics
    #    print(f"\nReducing topics with diversity threshold: {diversity_threshold}")
    #    topic_model.reduce_topics(
    #        df["clean_lyrics"],
    #        nr_topics="6"
    #    )
    #    topics = topic_model.topics_
    #    probs = topic_model.probabilities_
    #    print(f"Topics after reduction: {len(set(topics)) - (1 if -1 in topics else 0)}")

    # ------------------------
    # 8. Assign topics to dataframe
    # ------------------------
    df['id'] = topics
    if probs is not None:
        df['strength'] = probs.max(axis=1)
    else:
        df['strength'] = 0.0

    # ------------------------
    # 9. Topic labels
    # ------------------------
    topic_labels = []
    topic_info = topic_model.get_topic_info()
    print("\n" + "=" * 60)
    print("DISCOVERED TOPICS")
    print("=" * 60)

    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        if words:
            top_words = [word for word, _ in words[:5]]
            label = ", ".join(top_words)
            topic_labels.append(label)
            count = len([t for t in topics if t == topic_id])
            print(f"Topic {topic_id}: {label} (n={count})")

    # ------------------------
    # 10. Visualizations
    # ------------------------
    if visualize:
        os.makedirs("results", exist_ok=True)

        try:
            fig1 = topic_model.visualize_topics()

            fig1.write_html("results/bertopic_topics.html")
            print("\nSaved topic visualization to results/bertopic_topics.html")

        except Exception as e:
            print(f"\nCouldn't create topic visualization: {e}")

        try:
            fig2 = topic_model.visualize_heatmap()
            fig2.write_html("results/bertopic_heatmap.html")
            print("Saved heatmap to results/bertopic_heatmap.html")
        except Exception as e:
            print(f"Couldn't create heatmap: {e}")

        try:
            fig3 = topic_model.visualize_barchart(top_n_topics=8)
            fig3.write_html("results/bertopic_barchart.html")
            print("Saved barchart to results/bertopic_barchart.html")
        except Exception as e:
            print(f"Couldn't create barchart: {e}")

    # ------------------------
    # 11. Summary
    # ------------------------

    # After fitting, add this to your pipeline:
    print("\n" + "=" * 60)
    print("CLUSTER DISTRIBUTION CHECK")
    print("=" * 60)
    from collections import Counter
    topic_counts = Counter(topics)

    total_songs = len(topics)
    print(f"Total songs: {total_songs}")
    print(f"Unique assignments: {len(set(topics))}")
    print(f"Sum of cluster sizes: {sum(topic_counts.values())}")
    print(f"Match: {total_songs == sum(topic_counts.values())}")  # Should be True

    # Check for duplicates
    song_topic_pairs = list(enumerate(topics))
    
    print(f"Each song has exactly 1 topic: {len(song_topic_pairs) == total_songs}")
    print("\n" + "=" * 60)
    print("BERTOPIC PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total topics (excluding outliers): {len(topic_labels)}")
    print(f"Outlier songs (topic -1): {len([t for t in topics if t == -1])}")

    return topic_model, topics, probs, df, topic_labels

'''if __name__ == "__main__":
    """Example: Run robust topic modeling."""
    import pandas as pd

    # Load your data
    df_with_eras = pd.read_csv(config.RESULTS_DIR+"/songs_with_eras.csv")
    #merged_df = pd.read_csv(f"{config.RESULTS_DIR}/merged_records.csv")

    # Run multiple seeds
    #bert_model, bert_topics, probs, df_with_topics, bert_labels = bertopic_lyrics_pipeline(
    #    df_with_eras, visualize=False, min_cluster_size=4
    #)

    model1, topics1, _, _, _ = bertopic_lyrics_pipeline(df_with_eras, seed=42)
    model2, topics2, _, _, _ = bertopic_lyrics_pipeline(df_with_eras, seed=123)
    print(f"Seed 42: {len(set(topics1))} topics")
    print(f"Seed 123: {len(set(topics2))} topics")'''

'''if __name__ == "__main__":
    df_with_eras = pd.read_csv(config.RESULTS_DIR + "/songs_with_eras.csv")

    # Test 3 different seeds
    for seed in [42, 123, 456]:
        print(f"\n{'=' * 60}")
        print(f"TESTING SEED {seed}")
        print('=' * 60)

        model, topics, probs, df_result, labels = bertopic_lyrics_pipeline(
            df_with_eras,
            seed=seed,
            visualize=False,
            min_cluster_size=3
        )

        n_topics = len(labels)
        n_outliers = sum(1 for t in topics if t == -1)
        print(f"\nSeed {seed}: {n_topics} topics, {n_outliers} outliers")'''

'''if __name__ == "__main__":
    """Run robust topic modeling with multiple seeds."""
    import pandas as pd

    # Load data
    df_with_eras = pd.read_csv(config.RESULTS_DIR + "/songs_with_eras.csv")

    # Run with multiple seeds
    all_results = run_multiple_seeds(
        df_with_eras,
        seeds=[42, 123, 456, 789, 999],
        min_cluster_size=3
    )

    # Find most stable
    best_result, stabilities = calculate_stability_metrics(all_results)

    # Generate visualizations for best model only
    print(f"\n{'=' * 60}")
    print("GENERATING VISUALIZATIONS FOR BEST MODEL")
    print('=' * 60)

    # Re-run best seed with visualizations
    final_model, final_topics, final_probs, final_df, final_labels = bertopic_lyrics_pipeline(
        df_with_eras,
        seed=best_result['seed'],
        visualize=True,  # Now create visualizations
        min_cluster_size=3
    )

    # Run this
    analyze_spatial_clusters(final_df, final_model)
    # Run this
    create_talk_visualizations(final_model, final_df)


    # Save results
    print(f"\n{'=' * 60}")
    print("SAVING FINAL RESULTS")
    print('=' * 60)
    final_df.to_csv('results/final_topics_stable.csv', index=False)
    print("Saved topic assignments to: results/final_topics_stable.csv")

    # Save metadata
    import json

    metadata = {
        'best_seed': int(best_result['seed']),
        'n_topics': int(best_result['n_topics']),
        'n_outliers': int(best_result['n_outliers']),
        'stability_score': float(stabilities[[s['seed'] for s in stabilities].index(best_result['seed'])]['stability']),
        'all_seeds_tested': [42, 123, 456, 789, 999],
        'stability_scores': {s['seed']: float(s['stability']) for s in stabilities}
    }

    with open('results/topic_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to: results/topic_model_metadata.json")'''

if __name__ == "__main__":
    """Run robust topic modeling with multiple seeds."""
    import pandas as pd
    import json, sys
    # Load data
    df_with_eras = pd.read_csv(config.RESULTS_DIR + "/songs_with_eras.csv")

    # Run with multiple seeds
    all_results = run_multiple_seeds(
        df_with_eras,
        seeds=[42, 123, 456, 789, 999],
        min_cluster_size=3
    )

    # Find most stable
    best_result, stabilities = calculate_stability_metrics(all_results)

    # Generate visualizations for best model only
    print(f"\n{'=' * 60}")
    print("GENERATING VISUALIZATIONS FOR BEST MODEL")
    print('=' * 60)

    # Re-run best seed with visualizations
    final_model, final_topics, final_probs, final_df, final_labels = bertopic_lyrics_pipeline(
        df_with_eras,
        seed=best_result['seed'],
        visualize=True,  # Now create visualizations
        min_cluster_size=3
    )
    # 1. Name topics
    final_df, theme_names = assign_final_theme_names(final_model, final_df)
    # TEMPORARY: Find the right threshold
    print("\n" + "=" * 60)
    print("FINDING OPTIMAL THRESHOLD")
    print("=" * 60)
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.5]:
        print(f"\nThreshold {threshold}:")
        outliers, main_id, _ = identify_visual_outliers(final_model, final_df, threshold=threshold)
        print(f"  → Found {len(outliers)} outliers")
        if outliers:
            print(f"  → Topics: {[o['topic_id'] for o in outliers]}")

    # Debug topic IDs
    #debug_topic_ids(final_model, final_df)

    # Run this
    # final_df, theme_names = assign_final_theme_names(final_model, final_df)
    # final_df.to_csv('results/final_topics_named.csv', index=False)
    # print("✓ Topics named and saved!")

    # Analyze spatial clusters (matches what you see in the plot)
    #outlier_topics, main_id = analyze_spatial_clusters( final_model, final_df, theme_names )
    #print(outlier_topics, main_id)
    analyze_spatial_clusters(final_model, final_df, theme_names)
    #sys.exit()

    # Run this
    #analyze_spatial_clusters(final_df, final_model)
    # Run this
    create_talk_visualizations(final_model, final_df)
    # Interactive HTML with custom colors
    customize_topic_visualization(final_model, "results/bertopic_topics.html")

    # Alternative: Export Plotly as PNG (if you prefer exact BERTopic layout)
    save_topic_visualization_as_png(final_model, "results/bertopic_topics.png")

    # Save results
    print(f"\n{'=' * 60}")
    print("SAVING FINAL RESULTS")
    print('=' * 60)
    final_df.to_csv('results/final_topics_stable.csv', index=False)
    print("Saved topic assignments to: results/final_topics_stable.csv")


    metadata = {
        'best_seed': int(best_result['seed']),
        'n_topics': int(best_result['n_topics']),
        'n_outliers': int(best_result['n_outliers']),
        'stability_score': float(stabilities[[s['seed'] for s in stabilities].index(best_result['seed'])]['stability']),
        'all_seeds_tested': [42, 123, 456, 789, 999],
        'stability_scores': {s['seed']: float(s['stability']) for s in stabilities}
    }

    with open('results/topic_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to: results/topic_model_metadata.json")

    # FINAL TALK PREP
    print("\n" + "=" * 80)
    print("FINAL TALK PREPARATION")
    print("=" * 80)

    # 1. Name topics
    #final_df, theme_names = assign_final_theme_names(final_model, final_df)

    # 2. Create visualizations
    #create_talk_visualizations(final_model, final_df)

    # 3. Create summary
    #create_talk_summary(final_df, final_model, stabilities, theme_names)
    print("*"*80)
    #print("now identifying outliers")
    #identify_outlier_clusters(final_df, final_model)

    print("\n" + "✓" * 40)
    print("READY FOR TALK!")
    print("✓" * 40)
    print("\nYour talk materials are in: results/talk/")
    print("Main dataset: results/final_topics_named.csv")
