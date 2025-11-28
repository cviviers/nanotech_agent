# optimize_hdbscan.py
import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN


@dataclass
class EvalResult:
    params: Dict[str, Any]
    rel_validity: float
    assigned_ratio: float
    n_clusters: int
    silhouette: float
    score: float
    fit_time_s: float


def sample_int_log_uniform(rng: np.random.Generator, low: int, high: int) -> int:
    """Log-uniform integer sampler (inclusive bounds)."""
    x = rng.uniform(low=np.log(low), high=np.log(high))
    return int(np.clip(round(np.exp(x)), low, high))


def evaluate_once(
    X_fit: np.ndarray,
    X_sil: np.ndarray,
    params: Dict[str, Any],
    max_sil_points: int = 5000,
) -> EvalResult:
    t0 = time.time()
    clusterer = HDBSCAN(**params)
    clusterer.fit(X_fit)
    fit_time_s = time.time() - t0

    labels = clusterer.labels_
    noise_mask = labels == -1
    assigned_ratio = 1.0 - float(np.mean(noise_mask))
    n_clusters = int((np.unique(labels[~noise_mask]).size) if np.any(~noise_mask) else 0)

    # Primary quality: HDBSCAN's internal relative validity (0..1, higher is better)
    rel_validity = float(getattr(clusterer, "relative_validity_", np.nan))
    if math.isnan(rel_validity):
        rel_validity = 0.0

    # Silhouette on assigned points (sampled), only if ≥2 clusters
    silhouette = -1.0
    if n_clusters >= 2 and assigned_ratio > 0:
        # Align X_sil with current labels (recompute labels on X_sil if different set)
        # We use approximate_predict for speed if dimensions match and clusterer supports it
        try:
            from hdbscan import approximate_predict
            sil_labels, _ = approximate_predict(clusterer, X_sil)
        except Exception:
            # Fallback: reuse training labels by sampling from X_fit indices
            # (not ideal if X_sil != X_fit, but safe fallback)
            if X_sil is X_fit:
                sil_labels = labels
            else:
                sil_labels = None

        if sil_labels is None:
            silhouette = -1.0
        else:
            mask = sil_labels != -1
            if np.sum(mask) >= 10 and np.unique(sil_labels[mask]).size >= 2:
                # Sample for speed
                idx = np.where(mask)[0]
                if idx.size > max_sil_points:
                    idx = np.random.default_rng(0).choice(idx, size=max_sil_points, replace=False)
                try:
                    silhouette = float(silhouette_score(X_sil[idx], sil_labels[idx], metric=params.get("metric", "euclidean")))
                except Exception:
                    silhouette = -1.0

    # Composite score: prioritize rel_validity, reward more assigned points.
    # You can tweak the weighting if you want less noise.
    score = rel_validity * (0.50 + 0.50 * assigned_ratio)

    return EvalResult(
        params=params,
        rel_validity=rel_validity,
        assigned_ratio=assigned_ratio,
        n_clusters=n_clusters,
        silhouette=silhouette,
        score=score,
        fit_time_s=fit_time_s,
    )


def random_search_hdbscan(
    X: np.ndarray,
    n_trials: int = 60,
    pca_components: int = 50,
    seed: int = 42,
    n_jobs: int = -1,
    cluster_selection_methods=("leaf", "eom"),
    allow_metric=("euclidean",),
) -> Tuple[EvalResult, List[EvalResult]]:
    """
    Run a random search over HDBSCAN params on (optionally PCA-reduced) features.
    Returns the best result and a sorted list of all results.
    """
    rng = np.random.default_rng(seed)

    # Scale and (optionally) reduce dimensionality.
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    if pca_components and pca_components > 0 and pca_components < Xs.shape[1]:
        pca = PCA(n_components=pca_components, svd_solver="auto", random_state=seed)
        Xr = pca.fit_transform(Xs)
    else:
        Xr = Xs

    # For silhouette/approx_predict we’ll reuse the same transformed matrix
    X_fit = Xr
    X_sil = Xr

    # Build a list of parameter sets
    param_list: List[Dict[str, Any]] = []
    for _ in range(n_trials):
        min_cluster_size = sample_int_log_uniform(rng, low=5, high=500)
        min_samples = int(rng.integers(low=1, high=min(100, max(2, min_cluster_size)) + 1))
        # Small epsilon is usually best; explore a bit (relative to scaled space)
        cluster_selection_epsilon = float(rng.choice([0.0, 0.0, 0.0, rng.uniform(0.01, 0.5)]))
        csm = rng.choice(cluster_selection_methods)
        metric = rng.choice(allow_metric)

        params = dict(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=csm,
            metric=metric,
            approx_min_span_tree=True,   # speed/memory friendly
            core_dist_n_jobs=-1,         # parallel core distance
        )
        param_list.append(params)

    # Evaluate in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(evaluate_once)(X_fit, X_sil, params) for params in param_list
    )

    # Sort by composite score (desc), then by rel_validity, then by assigned_ratio
    results_sorted = sorted(results, key=lambda r: (r.score, r.rel_validity, r.assigned_ratio), reverse=True)
    best = results_sorted[0]
    return best, results_sorted


def main():
    parser = argparse.ArgumentParser(description="Random-search HDBSCAN parameters on high-dim data.")
    parser.add_argument("--npy", type=str, required=True,
                        help="Path to .npy file containing a NumPy array of shape (N, D).")
    parser.add_argument("--n-trials", type=int, default=60, help="Number of random configurations to try.")
    parser.add_argument("--pca-components", type=int, default=50, help="PCA components (0 or >=D to disable).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top configs to print.")
    args = parser.parse_args()

    X = np.load(args.npy)
    assert X.ndim == 2, "Expected 2D array (N, D)."

    best, results = random_search_hdbscan(
        X,
        n_trials=args.n_trials,
        pca_components=args.pca_components,
        seed=args.seed,
        n_jobs=args.n_jobs,
        cluster_selection_methods=("leaf", "eom"),
        allow_metric=("euclidean",),  # expand if you like (e.g., "manhattan", "cosine")
    )

    print("\n=== Top Results ===")
    for i, r in enumerate(results[:args.top_k], start=1):
        print(
            f"[{i:02d}] score={r.score:.4f}  rel_validity={r.rel_validity:.4f}  "
            f"assigned={r.assigned_ratio:.2%}  clusters={r.n_clusters:<3d}  "
            f"sil={r.silhouette:.3f}  time={r.fit_time_s:.1f}s  params={r.params}"
        )

    print("\n=== Best Configuration ===")
    r = best
    print(
        f"score={r.score:.4f}  rel_validity={r.rel_validity:.4f}  assigned={r.assigned_ratio:.2%}  "
        f"clusters={r.n_clusters}  sil={r.silhouette:.3f}  time={r.fit_time_s:.1f}s"
    )
    print("params=", r.params)

    # If you want to persist the best parameters:
    # import json, pathlib
    # pathlib.Path("best_hdbscan_params.json").write_text(json.dumps(r.params, indent=2))


if __name__ == "__main__":
    main()
