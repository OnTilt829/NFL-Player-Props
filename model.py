"""NFL opportunity x efficiency projections using nflverse weekly data."""
import argparse
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

STATS = {
    "passing_yards": ("attempts", "passing_yards"),
    "rushing_yards": ("carries", "rushing_yards"),
    "receiving_yards": ("targets", "receiving_yards"),
    "receptions": ("targets", "receptions"),
}
COLS = ["attempts", "passing_yards", "carries", "rushing_yards", "targets", "receiving_yards", "receptions"]

def build(season: int, week: int, output: str):
    weekly = nfl.import_weekly_data([season], downcast=True)
    weekly = weekly[(weekly["week"] < week) & (weekly["season_type"] == "REG")].copy()
    if weekly.empty:
        raise ValueError("No historical games before requested week")
    schedule = nfl.import_schedules([season])
    games = schedule[(schedule["week"] == week) & (schedule["game_type"] == "REG")]
    opponents = {}
    for _, g in games.iterrows():
        opponents[g.home_team] = g.away_team
        opponents[g.away_team] = g.home_team
    if not opponents:
        raise ValueError("No scheduled games for requested week")
    team_week = weekly.groupby(["season", "week", "opponent_team"], as_index=False)[COLS].sum()
    defense = team_week.groupby("opponent_team")[COLS].mean()
    players = weekly.groupby(["player_id", "player_display_name", "position", "recent_team"], as_index=False).agg(
        games=("week", "nunique"), **{c: (c, "sum") for c in COLS}
    )
    teams = weekly.groupby("recent_team")[["attempts", "carries", "targets"]].sum()
    rows = []
    for _, p in players.iterrows():
        opp = opponents.get(p.recent_team)
        if not opp or opp not in defense.index:
            continue
        d, t = defense.loc[opp], teams.loc[p.recent_team]
        row = {"player": p.player_display_name, "team": p.recent_team, "opponent": opp,
               "position": p.position, "games": int(p.games)}
        for stat, (volume, output_stat) in STATS.items():
            offensive = p[output_stat] / p.games
            share = p[volume] / t[volume] if t[volume] > 0 else 0
            defensive = d[output_stat] * share
            row[stat] = round((offensive + defensive) / 2, 2)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No players matched scheduled teams")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.sort_values(["team", "position", "player"]).to_csv(output, index=False)
    print(f"Wrote {len(result)} projections to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output", default="output/projections.csv")
    args = parser.parse_args()
    build(args.season, args.week, args.output)
