"""
Quick test script to verify the new embedding explorer setup

This script:
1. Creates a small synthetic dataset
2. Runs preprocessing
3. Verifies cache files are created
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocessing import preprocess_embeddings, EmbeddingConfig, load_preprocessed_data
from utils.data_utils_v2 import parse_embedding_column


def create_synthetic_dataset(n_samples=500, embedding_dim=768):
    """Create a synthetic dataset for testing"""
    print("Creating synthetic dataset...")
    
    # Generate synthetic embeddings (3 clusters)
    np.random.seed(42)
    
    embeddings_1 = np.random.randn(n_samples // 3, embedding_dim) + [2, 0, 0]
    embeddings_2 = np.random.randn(n_samples // 3, embedding_dim) + [0, 2, 0]
    embeddings_3 = np.random.randn(n_samples - 2 * (n_samples // 3), embedding_dim) + [0, 0, 2]
    
    all_embeddings = np.vstack([embeddings_1, embeddings_2, embeddings_3])
    
    # L2 normalize (common for embeddings)
    all_embeddings = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Create dataframe
    df = pd.DataFrame({
        'id': range(n_samples),
        'title': [f"Paper {i}" for i in range(n_samples)],
        'abstract': [f"This is the abstract for paper {i} about topic {i % 3}" for i in range(n_samples)],
        'qwen_embedding': [list(emb) for emb in all_embeddings],
        'bert_embedding': [list(emb + np.random.randn(embedding_dim) * 0.1) for emb in all_embeddings],  # Slightly perturbed
    })
    
    print(f"  Created {len(df)} samples with {embedding_dim}-dim embeddings")
    return df


def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("\n" + "="*80)
    print("TESTING EMBEDDING EXPLORER v2")
    print("="*80 + "\n")
    
    # Create test data
    df = create_synthetic_dataset(n_samples=500, embedding_dim=768)
    
    # Save to CSV
    test_file = Path("test_data.csv")
    print(f"\nSaving test data to {test_file}...")
    df.to_csv(test_file, index=False)
    
    # Parse embeddings
    print("\nParsing embedding columns...")
    df = parse_embedding_column(df, 'qwen_embedding')
    df = parse_embedding_column(df, 'bert_embedding')
    print("  ✓ Parsed qwen_embedding")
    print("  ✓ Parsed bert_embedding")
    
    # Create cache directory
    cache_dir = Path("cache_test")
    cache_dir.mkdir(exist_ok=True)
    
    # Configure UMAP
    config = EmbeddingConfig(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    
    print(f"\nUMAP Configuration:")
    print(f"  n_neighbors: {config.n_neighbors}")
    print(f"  min_dist: {config.min_dist}")
    print(f"  metric: {config.metric}")
    
    # Test preprocessing for each embedding
    for emb_col in ['qwen_embedding', 'bert_embedding']:
        print(f"\n{'='*80}")
        print(f"Processing: {emb_col}")
        print(f"{'='*80}")
        
        result = preprocess_embeddings(
            df=df,
            embedding_col=emb_col,
            config=config,
            cache_dir=cache_dir
        )
        
        print(f"  ✓ UMAP projection computed: {result['umap_2d'].shape}")
        print(f"  ✓ Cache saved to: {result['cache_path']}")
        
        # Verify cache can be loaded
        loaded_result = load_preprocessed_data(Path(result['cache_path']))
        print(f"  ✓ Cache verified (loaded {len(loaded_result['df'])} records)")
        
        # Show UMAP statistics
        umap_2d = result['umap_2d']
        print(f"\n  UMAP Statistics:")
        print(f"    X range: [{umap_2d[:, 0].min():.2f}, {umap_2d[:, 0].max():.2f}]")
        print(f"    Y range: [{umap_2d[:, 1].min():.2f}, {umap_2d[:, 1].max():.2f}]")
    
    # Summary
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - Test data: {test_file}")
    print(f"  - Cache directory: {cache_dir}/")
    print(f"    • qwen_embedding.pkl")
    print(f"    • bert_embedding.pkl")
    
    print("\n✓ All tests passed!")
    print("\nYou can now test the Streamlit app with this data:")
    print(f"  streamlit run streamlit_app_v2.py")
    print(f"\nOr run preprocessing on real data:")
    print(f"  python preprocess_embeddings.py --data_file <your_file.csv>")
    
    return True


if __name__ == "__main__":
    try:
        test_preprocessing()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
