# backend/app/data_loader.py

from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

DATA_ROOT = Path(__file__).parent.parent.parent / "data"


def load_aggregated_players_for_season_gw(season: str, gameweek: int) -> pd.DataFrame:
    """
    Aggregate player stats from GW CSVs up to (but not including) a target GW.
    Returns strict final schema WITHOUT player_id:

        name, position, team, value, xP, xGI, minutes,
        fixture, selected_by, threat, ict
    """
    if not season:
        raise ValueError("season must be provided")
    if gameweek is None:
        raise ValueError("gameweek must be provided")

    base = DATA_ROOT / season / "gws"
    if not base.exists():
        raise RuntimeError(f"Season directory not found: {base}")

    gw_range = [1] if gameweek <= 1 else list(range(1, gameweek))

    frames = []
    for gw in gw_range:
        gw_path = base / f"gw{gw}.csv"
        if not gw_path.exists():
            logger.warning(f"{gw_path} not found, skipping.")
            continue

        df = pd.read_csv(gw_path)
        df["gw"] = gw
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No gw CSVs found under {base}")

    df_all = pd.concat(frames, ignore_index=True)

    # --- Normalize columns we need ---
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

    # team
    if "team" in df_all.columns:
        df_all["team"] = df_all["team"]
    elif "team_name" in df_all.columns:
        df_all["team"] = df_all["team_name"]
    else:
        df_all["team"] = "Unknown"

    # metrics
    for col, default in [
        ("value", 0.0),
        ("xP", 0.0),
        ("expected_goal_involvements", 0.0),
        ("minutes", 0.0),
        ("fixture", ""),
        ("selected", 0.0),
        ("threat", 0.0),
        ("ict_index", 0.0),
    ]:
        if col not in df_all.columns:
            df_all[col] = default

    # aggregate by name + team (best unique key for now)
    agg = df_all.groupby(["name", "team"]).agg(
        position=("position", "last"),
        value=("value", "last"),
        xP=("xP", "mean"),
        xGI=("expected_goal_involvements", "mean"),
        minutes=("minutes", "sum"),
        fixture=("fixture", "last"),
        selected_by=("selected", "last"),
        threat=("threat", "mean"),
        ict=("ict_index", "mean"),
    ).reset_index()

    # enforce return ordering
    return agg[
        ["name", "position", "team", "value", "xP", "xGI",
         "minutes", "fixture", "selected_by", "threat", "ict"]
    ]
