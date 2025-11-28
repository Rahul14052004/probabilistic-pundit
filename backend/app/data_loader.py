# backend/app/data_loader.py

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

# Root data dir – adjust if needed
DATA_ROOT = Path(__file__).parent.parent.parent / "data"


def load_aggregated_players_for_season_gw(season: str, gameweek: int) -> pd.DataFrame:
    """
    Aggregate player stats from Vaastav GW CSVs for a given season up to (but not including) the given GW.
    Example expected structure:
      data/vaastav/2025-26/gw1.csv
      data/vaastav/2025-26/gw2.csv
      ...

    If gameweek <= 1, we fallback to using only gw1.csv as 'history'.
    """
    if not season:
        raise ValueError("season must be provided")
    if gameweek is None:
        raise ValueError("gameweek must be provided")

    base = DATA_ROOT / season / "gws" 
    if not base.exists():
        raise RuntimeError(f"Vaastav season directory not found: {base}")

    if gameweek <= 1:
        gw_range = [1]
    else:
        gw_range = list(range(1, gameweek))

    frames = []
    for gw in gw_range:
        gw_path = base / f"gw{gw}.csv"
        if not gw_path.exists():
            logger.warning(f"{gw_path} not found, skipping.")
            continue
        df_gw = pd.read_csv(gw_path)
        df_gw["gw"] = gw
        frames.append(df_gw)

    if not frames:
        raise RuntimeError(f"No GW CSVs found for season={season}, gameweek={gameweek} under {base}")

    df_all = pd.concat(frames, ignore_index=True)

    # --- Map columns to a common schema ---

    # id
    if "element" in df_all.columns:
        df_all["player_id"] = df_all["element"]
    elif "player_id" in df_all.columns:
        df_all["player_id"] = df_all["player_id"]
    elif "id" in df_all.columns:
        df_all["player_id"] = df_all["id"]
    else:
        raise RuntimeError("No player id column found (expected element/player_id/id).")

    # name
    if "name" in df_all.columns:
        df_all["name"] = df_all["name"]
    elif "web_name" in df_all.columns:
        df_all["name"] = df_all["web_name"]
    else:
        df_all["name"] = "Unknown"

    # position
    if "position" in df_all.columns:
        df_all["position"] = df_all["position"]
    elif "element_type" in df_all.columns:
        mapping = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        df_all["position"] = df_all["element_type"].map(mapping).fillna("UNK")
    else:
        df_all["position"] = "UNK"

    # team / club
    if "team" in df_all.columns:
        df_all["club"] = df_all["team"]
    elif "team_name" in df_all.columns:
        df_all["club"] = df_all["team_name"]
    else:
        df_all["club"] = "Unknown"

    # price
    if "now_cost" in df_all.columns:
        df_all["price"] = df_all["now_cost"].astype(float) / 10.0
    elif "price" in df_all.columns:
        df_all["price"] = df_all["price"].astype(float)
    else:
        df_all["price"] = 5.0  # placeholder

    # points & other stats
    if "total_points" not in df_all.columns:
        df_all["total_points"] = df_all.get("points", 0)

    if "minutes" not in df_all.columns:
        df_all["minutes"] = 0

    if "goals_scored" not in df_all.columns:
        df_all["goals_scored"] = 0

    if "assists" not in df_all.columns:
        df_all["assists"] = 0

    # --- Aggregate per player across past GWs ---

    agg = df_all.groupby("player_id").agg(
        name=("name", "last"),
        position=("position", "last"),
        club=("club", "last"),
        price=("price", "last"),
        total_points=("total_points", "sum"),
        minutes=("minutes", "sum"),
        goals_scored=("goals_scored", "sum"),
        assists=("assists", "sum"),
        appearances=("gw", "count"),
    ).reset_index()

    # Simple EV proxies
    agg["pts_per_appearance"] = agg["total_points"] / agg["appearances"].replace(0, np.nan)
    agg["pts_per_appearance"] = agg["pts_per_appearance"].fillna(0.0)

    agg["pts_per_90"] = agg["total_points"] / (agg["minutes"] / 90.0).replace(0, np.nan)
    agg["pts_per_90"] = agg["pts_per_90"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    agg["expected_points"] = agg["pts_per_90"]

    agg["value_season"] = agg["total_points"] / agg["price"].replace(0, np.nan)
    agg["value_season"] = agg["value_season"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return agg
