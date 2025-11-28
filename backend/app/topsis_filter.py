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
    Use a few sensible features to rank players:

    - higher expected_points is better
    - higher value_season is better
    - higher pts_per_90 is better
    - lower price is better

    Returns top n_candidates rows, sorted by 'topsis_score' desc.
    The selection is position-aware, roughly matching the FPL squad ratio:
      Final squad: 2 GK, 5 DEF, 5 MID, 3 FWD  (total 15)
      Candidate pool is scaled from this ratio.
    """
    criteria_cols = []
    weights = []
    impacts = []

    # These columns are produced in data_loader.load_aggregated_players_for_season_gw
    if "expected_points" in df.columns:
        criteria_cols.append("expected_points")
        weights.append(0.4)
        impacts.append('+')

    if "value_season" in df.columns:
        criteria_cols.append("value_season")
        weights.append(0.3)
        impacts.append('+')

    if "pts_per_90" in df.columns:
        criteria_cols.append("pts_per_90")
        weights.append(0.2)
        impacts.append('+')

    if "price" in df.columns:
        criteria_cols.append("price")
        weights.append(0.1)
        impacts.append('-')  # cheaper is better

    # If no criteria available, just fallback
    if not criteria_cols:
        df = df.copy()
        df["topsis_score"] = 0.5
        return df.head(n_candidates)

    # 1) Compute global TOPSIS scores
    scores = topsis_rank(df, criteria_cols, weights, impacts)
    df = df.copy()
    df["topsis_score"] = scores

    # 2) Position-aware selection
    # FPL final squad ratio: 2 GK, 5 DEF, 5 MID, 3 FWD (total 15)
    # Scale that ratio to n_candidates
    total_slots = float(n_candidates)
    ratio = {
        "GK": 2 / 15.0,
        "DEF": 5 / 15.0,
        "MID": 5 / 15.0,
        "FWD": 3 / 15.0,
    }

    # Compute per-position quotas (rounded)
    quotas = {
        pos: max(1, int(round(total_slots * frac)))
        for pos, frac in ratio.items()
    }
    # Make sure we don't overshoot n_candidates too much
    # (small rounding errors are ok; we clamp later)
    # Example for 30 candidates: GK=4, DEF=10, MID=10, FWD=6

    selected_indices = []

    # Helper: sort once
    df_sorted = df.sort_values("topsis_score", ascending=False)

    # 2a) Pick top players per position according to quotas
    for pos, quota in quotas.items():
        df_pos = df_sorted[df_sorted["position"] == pos]
        if df_pos.empty:
            continue
        # Take up to `quota` players for this position
        chosen = df_pos.head(quota).index.tolist()
        selected_indices.extend(chosen)

    # 2b) If we still have remaining slots, fill with best remaining players
    remaining_slots = n_candidates - len(set(selected_indices))
    if remaining_slots > 0:
        remaining_df = df_sorted.drop(index=list(set(selected_indices)), errors="ignore")
        extra = remaining_df.head(remaining_slots).index.tolist()
        selected_indices.extend(extra)

    # 2c) Build final candidate df
    df_selected = df.loc[list(set(selected_indices))].copy()
    df_selected = df_selected.sort_values("topsis_score", ascending=False)

    # Finally clamp to at most n_candidates (in case of any rounding wiggle)
    return df_selected.head(n_candidates)
