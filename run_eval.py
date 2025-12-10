"""
Run evaluation on Qwen embeddings
"""

import sys
import os

# Add parent directory to path to import embedding_models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding_models.eval import run_full_evaluation
import json
import numpy as np
import pandas as pd

# Paths to data files
DATA_JSON = r"data\cleaned_dataset.json"
EMBEDDINGS_NPY = r"data\bert_embeddings.npy"
OUTPUT_JSON = r"embedding_models\bert_eval_results.json"

def main():
    print("Loading data...")
    with open(DATA_JSON, 'r') as f:
        data_list = json.load(f)
    df = pd.DataFrame(data_list)
    
    print(f"Loaded {len(df)} records")
    
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_NPY)
    
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    if len(df) != embeddings.shape[0]:
        print(f"WARNING: Mismatch between records ({len(df)}) and embeddings ({embeddings.shape[0]})")
    
    print("\nRunning evaluation...")
    results = run_full_evaluation(
        df,
        embeddings=embeddings,
        keyword_col="mesh",
        min_keyword_freq=2,
        k_retrieval=100,
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
