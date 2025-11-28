"""
Data utilities for loading and parsing embeddings
"""
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load dataframe from CSV or JSON"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return df


def parse_embedding_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Parse embedding column from string representation to numpy array
    
    Handles both:
    - String representations like "[0.1, 0.2, ...]"
    - Already parsed lists/arrays
    """
    df = df.copy()
    
    def parse_single(x):
        if isinstance(x, str):
            try:
                # Try to parse as literal
                return np.array(ast.literal_eval(x), dtype=np.float32)
            except:
                return None
        elif isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=np.float32)
        else:
            return None
    
    df[col] = df[col].apply(parse_single)
    
    # Drop rows where parsing failed
    valid_mask = df[col].notna()
    if not valid_mask.all():
        print(f"Warning: Dropped {(~valid_mask).sum()} rows with invalid embeddings in {col}")
        df = df[valid_mask].reset_index(drop=True)
    
    return df


def write_df_to_excel(df: pd.DataFrame, output_dir: str = "output"):
    """Export dataframe to Excel file"""
    from datetime import datetime
    
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/export_{timestamp}.xlsx"
    
    # Drop embedding columns for export (too large)
    export_cols = [col for col in df.columns if 'embedding' not in col.lower()]
    df_export = df[export_cols].copy()
    
    df_export.to_excel(filename, index=False, engine='openpyxl')
    print(f"Exported to {filename}")
    
    return filename


def load_subset(df: pd.DataFrame, subset_size: Optional[int] = None) -> pd.DataFrame:
    """Load a random subset of the dataframe"""
    if subset_size is None or subset_size >= len(df):
        return df
    
    return df.sample(n=subset_size, random_state=42).reset_index(drop=True)
