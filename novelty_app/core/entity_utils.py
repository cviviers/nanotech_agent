"""
Entity extraction utilities for domain-specific analysis
"""
from typing import Dict, List, Any
from collections import Counter
import pandas as pd

from .constants import MATERIAL_HINTS, LIGAND_HINTS, DISEASE_HINTS, DELIVERY_HINTS, MODEL_HINTS


def simple_entity_extract(text: str) -> Dict[str, List[str]]:
    """Extract domain-specific entities from text using keyword matching."""
    text_lower = (text or '').lower()
    
    entities = {
        'materials': sorted({w for w in MATERIAL_HINTS if w in text_lower}),
        'ligands': sorted({w for w in LIGAND_HINTS if w in text_lower}),
        'diseases': sorted({w for w in DISEASE_HINTS if w in text_lower}),
        'delivery': sorted({w for w in DELIVERY_HINTS if w in text_lower}),
        'models': sorted({w for w in MODEL_HINTS if w in text_lower}),
    }
    
    return entities


def extract_entities_from_dataframe(df: pd.DataFrame, text_col: str = 'processed_content') -> pd.DataFrame:
    """Extract entities for all papers in dataframe."""
    entity_lists = {
        'materials': [],
        'ligands': [],
        'diseases': [],
        'delivery': [],
        'models': []
    }
    
    for _, row in df.iterrows():
        text = str(row.get(text_col) or row.get('abstract') or row.get('content') or '')
        entities = simple_entity_extract(text)
        
        for key in entity_lists.keys():
            entity_lists[key].append(entities.get(key, []))
    
    # Add as columns to dataframe copy
    df_with_entities = df.copy()
    for key, values in entity_lists.items():
        df_with_entities[f'entities_{key}'] = values
    
    return df_with_entities


def summarize_gap_region_entities(df: pd.DataFrame, region_indices: List[int]) -> Dict[str, Any]:
    """Summarize entity distribution in a gap region."""
    region_df = df.loc[region_indices]
    
    summary = {}
    for entity_type in ['materials', 'ligands', 'diseases', 'delivery', 'models']:
        col_name = f'entities_{entity_type}'
        if col_name in region_df.columns:
            # Flatten list of lists and count
            all_entities = []
            for entity_list in region_df[col_name]:
                if isinstance(entity_list, list):
                    all_entities.extend(entity_list)
            
            entity_counts = Counter(all_entities)
            summary[entity_type] = {
                'total_unique': len(entity_counts),
                'total_mentions': sum(entity_counts.values()),
                'top_5': entity_counts.most_common(5)
            }
        else:
            summary[entity_type] = {
                'total_unique': 0,
                'total_mentions': 0,
                'top_5': []
            }
    
    return summary
