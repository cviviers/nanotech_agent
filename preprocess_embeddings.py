"""
Standalone preprocessing script for computing UMAP projections

Usage:
    python preprocess_embeddings.py --data_file <path> --embedding_cols <col1,col2,...>

Example:
    python preprocess_embeddings.py \
        --data_file papers_dataframe_full_processed_with_processed_embeddings_parsed.csv \
        --embedding_cols qwen_processed_content_embedding,bert_processed_content_embedding
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from utils.preprocessing import (
    preprocess_all_embeddings,
    EmbeddingConfig
)
from utils.data_utils_v2 import (
    load_dataframe,
    parse_embedding_column
)


def get_available_embeddings(df: pd.DataFrame) -> list:
    """Extract available embedding columns"""
    return [
        col for col in df.columns 
        if 'embedding' in col.lower() and df[col].dtype == 'object'
    ]


def main():
    parser = argparse.ArgumentParser(description="Preprocess embeddings with UMAP")
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to CSV file with embeddings'
    )
    parser.add_argument(
        '--embedding_cols',
        type=str,
        default=None,
        help='Comma-separated list of embedding columns (default: all)'
    )
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter'
    )
    parser.add_argument(
        '--min_dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean', 'manhattan'],
        help='UMAP distance metric'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache',
        help='Directory to save preprocessed results'
    )
    parser.add_argument(
        '--subset',
        type=int,
        default=None,
        help='Use only a subset of N records (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.data_file).exists():
        print(f"Error: File not found: {args.data_file}")
        sys.exit(1)
    
    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("EMBEDDING PREPROCESSING")
    print("="*80)
    print(f"Data file: {args.data_file}")
    print(f"Cache dir: {cache_dir}")
    print()
    
    # Load data
    print("Loading data...")
    df = load_dataframe(args.data_file)
    print(f"  Loaded {len(df)} records")

    # Optionally drop some rows.
    # Exclusion criteria
    keywords_exclusion = ["review", "not available"]
    for keyword in keywords_exclusion:
        df = df[~df['title'].str.lower().str.contains(keyword)]
    
    # Optionally subset
    if args.subset is not None:
        print(f"  Using subset of {args.subset} records")
        df = df.sample(n=min(args.subset, len(df)), random_state=42)
    
    # Get embedding columns
    available_embeddings = get_available_embeddings(df)
    print(f"\nAvailable embedding columns:")
    for col in available_embeddings:
        print(f"  - {col}")
    
    # Determine which columns to process
    if args.embedding_cols:
        embedding_cols = [c.strip() for c in args.embedding_cols.split(',')]
        # Validate
        invalid = set(embedding_cols) - set(available_embeddings)
        if invalid:
            print(f"\nError: Invalid embedding columns: {invalid}")
            sys.exit(1)
    else:
        embedding_cols = available_embeddings
    
    print(f"\nProcessing {len(embedding_cols)} embedding column(s):")
    for col in embedding_cols:
        print(f"  - {col}")
    
    # Parse embedding columns
    print("\nParsing embeddings...")
    for col in embedding_cols:
        df = parse_embedding_column(df, col)
        # Check embedding dimension
        sample_emb = df[col].iloc[0]
        print(f"  {col}: dimension = {len(sample_emb)}")
    
    # Create UMAP configuration
    config = EmbeddingConfig(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=42
    )
    
    print(f"\nUMAP Configuration:")
    print(f"  n_neighbors: {config.n_neighbors}")
    print(f"  min_dist: {config.min_dist}")
    print(f"  metric: {config.metric}")
    print()
    
    # Run preprocessing
    print("="*80)
    print("RUNNING UMAP PROJECTIONS")
    print("="*80)
    
    results = preprocess_all_embeddings(
        df=df,
        embedding_cols=embedding_cols,
        config=config,
        cache_dir=cache_dir
    )
    
    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Successfully processed {len(results)} embedding column(s):")
    for emb_col, result in results.items():
        print(f"\n  {emb_col}:")
        print(f"    - UMAP shape: {result['umap_2d'].shape}")
        print(f"    - Cache file: {result['cache_path']}")
    
    print(f"\nCache directory: {cache_dir.absolute()}")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run streamlit_app_v2.py")


if __name__ == "__main__":
    main()
