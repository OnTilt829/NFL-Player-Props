"""NFL player projections using nflverse player stats and schedules."""
import argparse
from pathlib import Path
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
STATS = {"passing_yards": ("attempts", "passing_yards"), "rushing_yards": ("carries", "rushing_yards"), "receiving_yards": ("targets", "receiving_yards"), "receptions": ("targets", "receptions")}
COLS = sorted(set(x for pair in STATS.values() for x in pair))

def download_parquet(url):
    try:
        return pd.read_parquet(url)
    except Exception as exc:
        raise RuntimeError(f"Failed to load NFL data from {url}: {exc}") from exc

def build(season, week, output):
    # nflverse player_stats release includes one parquet file per season.
    weekly = download_parquet(f"{BASE}/player_stats/player_stats_{season}.parquet")
    schedule = pd.read_csv(f"{BASE}/schedules/games.csv")
    for col in ["season", "week"]:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce")
        schedule[col] = pd.to_numeric(schedule[col], errors="coerce")
    weekly = weekly[(weekly.season == season) & (weekly.week < week)]
    if "season_type" in weekly.columns:
        weekly = weekly[weekly.season_type == "REG"]
    if weekly.empty:
        raise ValueError("No completed regular-season player stats found for selected week")
    games = schedule[(schedule.season == season) & (schedule.week == week) & (schedule.game_type == "REG")]
    opponents = {}
    for _, g in games.iterrows():
        opponents[g.home_team] = g.away_team
        opponents[g.away_team] = g.home_team
    if not opponents:
        raise ValueError("No matchups found for requested season/week")
    required = ["player_id", "player_display_name", "position", "recent_team", "opponent_team"] + COLS
    missing = [c for c in required if c not in weekly.columns]
    if missing:
        raise ValueError(f"Player stats missing columns: {missing}")
    weekly[COLS] = weekly[COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    defense = weekly.groupby(["opponent_team", "week"], as_index=False)[COLS].sum().groupby("opponent_team")[COLS].mean()
    players = weekly.groupby(["player_id", "player_display_name", "position", "recent_team"], as_index=False).agg(
        games=("week", "nunique"), **{c: (c, "sum") for c in COLS})
    teams = weekly.groupby("recent_team")[["attempts", "carries", "targets"]].sum()
    rows = []
    for _, p in players.iterrows():
        opp = opponents.get(p.recent_team)
        if not opp or opp not in defense.index or p.games <= 0:
            continue
        d, t = defense.loc[opp], teams.loc[p.recent_team]
        row = dict(player=p.player_display_name, team=p.recent_team, opponent=opp, position=p.position, games=int(p.games))
        for stat, (volume, metric) in STATS.items():
            share = p[volume] / t[volume] if t[volume] else 0
            row[stat] = round((p[metric] / p.games + d[metric] * share) / 2, 2)
        rows.append(row)
    if not rows:
        raise ValueError("No players matched scheduled teams")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["team", "position", "player"]).to_csv(output, index=False)
    print(f"Wrote {len(rows)} projections to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output", default="output/projections.csv")
    args = parser.parse_args()
    build(args.season, args.week, args.output)
