from typing import List
import numpy as np
import pandas as pd


def topsis_rank(
    df: pd.DataFrame,
    criteria_cols: List[str],
    weights: List[float],
    impacts: List[str],
) -> pd.Series:
    """
    Run TOPSIS on df[criteria_cols].

    criteria_cols: list of metric columns to use.
    weights:       importance of each metric (same length).
    impacts:       '+' if higher is better, '-' if lower is better, per metric.

    Returns a pandas Series of TOPSIS scores (0–1), indexed like df.
    """
    if len(criteria_cols) != len(weights) or len(criteria_cols) != len(impacts):
        raise ValueError("criteria_cols, weights, impacts must have same length")

    # Extract matrix
    mat = df[criteria_cols].values.astype(float)

    # 1) Normalize each column (vector norm)
    norm = np.linalg.norm(mat, axis=0)
    norm[norm == 0] = 1.0  # avoid div by zero
    mat_norm = mat / norm

    # 2) Apply weights
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    mat_weighted = mat_norm * w

    # 3) Ideal best & worst based on impact sign
    ideal_best = np.zeros(len(criteria_cols))
    ideal_worst = np.zeros(len(criteria_cols))

    for i, imp in enumerate(impacts):
        col = mat_weighted[:, i]
        if imp == '+':  # higher is better
            ideal_best[i] = col.max()
            ideal_worst[i] = col.min()
        else:  # '-' lower is better
            ideal_best[i] = col.min()
            ideal_worst[i] = col.max()

    # 4) Distances to ideal best/worst
    dist_best = np.sqrt(((mat_weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((mat_weighted - ideal_worst) ** 2).sum(axis=1))

    # 5) TOPSIS score
    scores = dist_worst / (dist_best + dist_worst + 1e-9)
    return pd.Series(scores, index=df.index, name="topsis_score")

def select_top_players_with_topsis(df: pd.DataFrame, n_candidates: int = 30) -> pd.DataFrame:
    """
    Rank players using TOPSIS on FPL-style features:

    Higher is better for: xP, xGI, threat, ict, minutes
    Lower is better for: price

    Returns top n_candidates rows, sorted by 'topsis_score' desc,
    with position-aware selection (GK/DEF/MID/FWD-like).
    """
    criteria_cols = []
    weights = []
    impacts = []

    # 1) Build criteria set
    if "xP" in df.columns:
        criteria_cols.append("xP")
        weights.append(0.25)
        impacts.append('+')

    if "xGI" in df.columns:
        criteria_cols.append("xGI")
        weights.append(0.2)
        impacts.append('+')

    if "threat" in df.columns:
        criteria_cols.append("threat")
        weights.append(0.2)
        impacts.append('+')

    if "ict" in df.columns:
        criteria_cols.append("ict")
        weights.append(0.15)
        impacts.append('+')

    if "minutes" in df.columns:
        criteria_cols.append("minutes")
        weights.append(0.1)
        impacts.append('+')

    if "value" in df.columns:
        criteria_cols.append("value")
        weights.append(0.1)
        impacts.append('-')  # cheaper is better

    # Fallback if nothing available
    if not criteria_cols:
        df = df.copy()
        df["topsis_score"] = 0.5
        return df.head(n_candidates)

    # 2) Compute TOPSIS scores
    scores = topsis_rank(df, criteria_cols, weights, impacts)
    df = df.copy()
    df["topsis_score"] = scores

    # 3) Position-aware selection
    total_slots = float(n_candidates)
    ratio = {
        "GK": 2 / 15.0,
        "DEF": 5 / 15.0,
        "MID": 5 / 15.0,
        "FWD": 3 / 15.0,
    }

    quotas = {pos: max(1, int(round(total_slots * frac))) for pos, frac in ratio.items()}
    df_sorted = df.sort_values("topsis_score", ascending=False)

    selected_indices = []

    # 3a) Per-position quotas
    for pos, quota in quotas.items():
        pos_df = df_sorted[df_sorted["position"] == pos]
        if not pos_df.empty:
            selected_indices.extend(pos_df.head(quota).index.tolist())

    # 3b) Fill remaining slots with global best
    remaining = n_candidates - len(set(selected_indices))
    if remaining > 0:
        remaining_df = df_sorted.drop(index=list(set(selected_indices)), errors="ignore")
        selected_indices.extend(remaining_df.head(remaining).index.tolist())

    # 4) Return final candidate pool
    df_selected = df.loc[list(set(selected_indices))].copy()
    df_selected = df_selected.sort_values("topsis_score", ascending=False)
    return df_selected.head(n_candidates)
