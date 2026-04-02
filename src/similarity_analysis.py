from src import *
import os
import pickle

def create_hybrid_similarity_system(merged_df, use_cache=True):
    import pickle
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    df = merged_df.copy()

    lyric_cache   = config.LYRIC_EMBEDDINGS_PKL
    lyric_npy     = config.LYRIC_SIMILARITY_NPY
    audio_npy     = config.AUDIO_SIMILARITY_NPY
    hybrid_npy    = config.HYBRID_SIMILARITY_NPY

    # Load precomputed similarity matrices if available.
    if (use_cache
            and os.path.exists(lyric_npy)
            and os.path.exists(audio_npy)
            and os.path.exists(hybrid_npy)):
        print("Loading precomputed similarity matrices...")
        lyric_similarity  = np.load(lyric_npy)
        audio_similarity  = np.load(audio_npy)
        hybrid_similarity = np.load(hybrid_npy)
        print("Loaded from npy cache — no lyrics processed")
        return {
            'df': df,
            'lyric_similarity': lyric_similarity,
            'audio_similarity': audio_similarity,
            'hybrid_similarity': hybrid_similarity,
            'available_audio': [c for c in config.AUDIO_FEATURES if c in df.columns]
        }

    # Fall back to computing from scratch (local dev only).
    print("Precomputed matrices not found — computing from scratch...")

    # Lyric embeddings.
    if use_cache and os.path.exists(lyric_cache):
        print("Loading cached lyric embeddings...")
        with open(lyric_cache, 'rb') as f:
            lyric_embeddings = pickle.load(f)
    else:
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        lyric_embeddings = model.encode(
            df['lyrics'].tolist(), show_progress_bar=True
        )
        with open(lyric_cache, 'wb') as f:
            pickle.dump(lyric_embeddings, f)
        print("Cached lyric embeddings")

    lyric_similarity = cosine_similarity(lyric_embeddings)

    # Audio similarity.
    available_audio = [c for c in config.AUDIO_FEATURES if c in df.columns]
    audio_features = df[available_audio].fillna(0)
    scaler = StandardScaler()
    audio_scaled = scaler.fit_transform(audio_features)
    audio_similarity = cosine_similarity(audio_scaled)

    # Hybrid.
    alpha = config.SIMILARITY_ALPHA
    hybrid_similarity = alpha * lyric_similarity + (1 - alpha) * audio_similarity

    # Save all three for future runs.
    np.save(lyric_npy,  lyric_similarity)
    np.save(audio_npy,  audio_similarity)
    np.save(hybrid_npy, hybrid_similarity)
    print("Saved similarity matrices to npy")

    return {
        'df': df,
        'lyric_similarity': lyric_similarity,
        'audio_similarity': audio_similarity,
        'hybrid_similarity': hybrid_similarity,
        'available_audio': available_audio
    }

def analyze_similarity_improvements(similarity_results):
    """Analyze how hybrid approach improves recommendations."""
    df = similarity_results['df']
    lyric_sim = similarity_results['lyric_similarity']
    hybrid_sim = similarity_results['hybrid_similarity']
    
    print("\n" + "="*80)
    print("SIMILARITY ANALYSIS: Lyrics vs Hybrid")
    print("="*80)
    
    # Find cases where recommendations differ significantly
    interesting_cases = []
    
    for i in range(len(df)):
        # Top 5 from lyrics
        lyric_top5 = np.argsort(lyric_sim[i])[-6:-1][::-1]  # Exclude self
        
        # Top 5 from hybrid
        hybrid_top5 = np.argsort(hybrid_sim[i])[-6:-1][::-1]
        
        # Check if they differ
        overlap = len(set(lyric_top5) & set(hybrid_top5))
        if overlap < 3:  # Less than 3 songs in common
            interesting_cases.append({
                'song_idx': i,
                'song_name': df.iloc[i]['Formatted_name'] if 'Formatted_name' in df.columns else df.iloc[i]['Song_Name'],
                'album': df.iloc[i]['Album'],
                'overlap': overlap,
                'lyric_top5': lyric_top5,
                'hybrid_top5': hybrid_top5
            })
    
    print(f"\nFound {len(interesting_cases)} songs where recommendations significantly differ")
    
    # Show a few examples
    if interesting_cases:
        print("\nExample cases where hybrid approach changes recommendations:")
        for case in interesting_cases[:3]:
            print(f"\n  Song: {case['song_name']} ({case['album']})")
            print(f"  Overlap: {case['overlap']}/5 songs")
    
    return interesting_cases

def visualize_similarity_comparison(similarity_results, song_idx=0):
    """Visualize how recommendations change between approaches."""
    df = similarity_results['df']
    lyric_sim = similarity_results['lyric_similarity']
    hybrid_sim = similarity_results['hybrid_similarity']
    
    song_name = df.iloc[song_idx]['Formatted_name'] if 'Formatted_name' in df.columns else df.iloc[song_idx]['Song_Name']
    
    # Get top 10 recommendations from each
    lyric_top10 = np.argsort(lyric_sim[song_idx])[-11:-1][::-1]
    hybrid_top10 = np.argsort(hybrid_sim[song_idx])[-11:-1][::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Lyrics-only recommendations
    lyric_scores = lyric_sim[song_idx][lyric_top10]
    lyric_names = [df.iloc[i]['Formatted_name'][:20] if 'Formatted_name' in df.columns 
                   else df.iloc[i]['Song_Name'][:20] for i in lyric_top10]
    
    axes[0].barh(range(10), lyric_scores, color='steelblue')
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels(lyric_names)
    axes[0].set_xlabel('Similarity Score')
    axes[0].set_title(f'Lyrics-Only Recommendations\nfor "{song_name}"')
    axes[0].invert_yaxis()

    # Hybrid recommendations
    hybrid_scores = hybrid_sim[song_idx][hybrid_top10]
    print(hybrid_top10)
    hybrid_names = [
        str(df.iloc[i]['Formatted_name'])[:20] if 'Formatted_name' in df.columns
        else str(df.iloc[i]['Song_Name'])[:20]
        for i in hybrid_top10
    ]
    
    axes[1].barh(range(10), hybrid_scores, color='mediumseagreen')
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels(hybrid_names)
    axes[1].set_xlabel('Similarity Score')
    axes[1].set_title(f'Hybrid (Lyrics + Audio) Recommendations\nfor "{song_name}"')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.DATA_SCIENCE_DATA, "Taylor_Swift_agentic", "results", "similarity_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: results/similarity_comparison.png")


def evaluate_recommender(similarity_matrix, df, ground_truth="era"):
    """Benchmark recommender using MRR and nDCG against ground truth (same era/album)."""
    print("\n" + "=" * 80)
    print("RECOMMENDER BENCHMARKING")
    print("=" * 80)

    def mean_reciprocal_rank():
        scores = []
        for i in range(len(df)):
            true_label = df.iloc[i][ground_truth]
            sims = similarity_matrix[i]
            ranking = np.argsort(sims)[::-1]
            # skip self
            ranking = ranking[ranking != i]
            for rank, j in enumerate(ranking, 1):
                if df.iloc[j][ground_truth] == true_label:
                    scores.append(1 / rank)
                    break
        return np.mean(scores)

    mrr = mean_reciprocal_rank()
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    return mrr


