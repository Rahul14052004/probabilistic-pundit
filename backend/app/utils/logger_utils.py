import os, json
from datetime import datetime

def save_json_log(season: str, gameweek: int, name: str, data: dict):
    """Save any object as a JSON log under /log/<season>_<gw>/<name>.json"""
    
    folder = f"log/{season}_GW{gameweek}"
    os.makedirs(folder, exist_ok=True)

    file_path = f"{folder}/{name}.json"

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[LOG] Saved → {file_path}")
