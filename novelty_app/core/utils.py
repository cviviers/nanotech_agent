"""
Utility functions for parsing and processing data
"""
import ast
from typing import Any, Optional
import numpy as np


def parse_embedding(value: Any) -> Optional[np.ndarray]:
    """Parse embedding from various formats (list, array, string)"""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return np.asarray(parsed, dtype=float)
        except Exception:
            return None
    return None
