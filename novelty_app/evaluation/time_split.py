from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class CorpusSplit:
    df: pd.DataFrame
    embeddings: Dict[str, np.ndarray]


@dataclass
class TimeSplitResult:
    historical: CorpusSplit
    future: CorpusSplit
    publication_dates: pd.Series


def load_dataset_and_embeddings(
    data_json: str | Path,
    data_dir: str | Path,
    embedding_names: List[str],
) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    data_path = Path(data_json)
    base_dir = Path(data_dir)
    with data_path.open("r", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df = df.reset_index(drop=True).copy()
    df["source_row_index"] = np.arange(len(df), dtype=int)

    embeddings: Dict[str, np.ndarray] = {}
    for name in embedding_names:
        emb_path = base_dir / f"{name}_embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")
        arr = np.load(emb_path).astype(np.float32)
        if len(arr) != len(df):
            raise ValueError(f"Embedding length mismatch for {name}: {len(arr)} != {len(df)}")
        embeddings[name] = arr
    return df, embeddings


def _build_publication_dates(df: pd.DataFrame) -> pd.Series:
    years = pd.to_numeric(df.get("publication_year"), errors="coerce")
    months = pd.to_numeric(df.get("publication_month"), errors="coerce")
    days = pd.to_numeric(df.get("publication_day"), errors="coerce")

    values: List[pd.Timestamp] = []
    for idx in range(len(df)):
        year = years.iloc[idx]
        if pd.isna(year):
            values.append(pd.NaT)
            continue
        y = int(year)
        month = int(months.iloc[idx]) if not pd.isna(months.iloc[idx]) else 12
        month = min(max(month, 1), 12)
        if not pd.isna(days.iloc[idx]):
            day = int(days.iloc[idx])
        else:
            day = calendar.monthrange(y, month)[1]
        day = min(max(day, 1), calendar.monthrange(y, month)[1])
        values.append(pd.Timestamp(year=y, month=month, day=day))
    return pd.Series(values, index=df.index, name="publication_date")


def _subset_embeddings(embeddings: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: arr[mask].copy() for name, arr in embeddings.items()}


def split_corpus_by_time(
    df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    *,
    cutoff_date: str,
    future_window_start: str,
    future_window_end: str,
) -> TimeSplitResult:
    if not embeddings:
        raise ValueError("embeddings are required")
    if any(len(arr) != len(df) for arr in embeddings.values()):
        raise ValueError("all embeddings must align with the dataframe")

    publication_dates = _build_publication_dates(df)
    cutoff = pd.Timestamp(cutoff_date)
    future_start = pd.Timestamp(future_window_start)
    future_end = pd.Timestamp(future_window_end)

    historical_mask = (publication_dates.notna()) & (publication_dates <= cutoff)
    future_mask = (publication_dates.notna()) & (publication_dates >= future_start) & (publication_dates <= future_end)

    historical_df = df.loc[historical_mask].reset_index(drop=True).copy()
    future_df = df.loc[future_mask].reset_index(drop=True).copy()

    hist_embeddings = _subset_embeddings(embeddings, historical_mask.to_numpy())
    future_embeddings = _subset_embeddings(embeddings, future_mask.to_numpy())

    if not historical_df.empty and pd.to_datetime(historical_df["publication_year"], format="%Y", errors="coerce").dt.year.max() > cutoff.year:
        raise ValueError("historical split leaked post-cutoff papers")

    return TimeSplitResult(
        historical=CorpusSplit(df=historical_df, embeddings=hist_embeddings),
        future=CorpusSplit(df=future_df, embeddings=future_embeddings),
        publication_dates=publication_dates,
    )
