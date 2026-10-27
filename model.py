"""NFL player performance forecasts (non-wagering research baseline)."""
import argparse
from pathlib import Path
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
METRICS = {"passing_yards":"attempts","rushing_yards":"carries","receiving_yards":"targets","receptions":"targets"}
FIELDS = sorted(set(METRICS) | set(METRICS.values()))

def fetch(url, kind):
    try:
        return pd.read_parquet(url) if kind == "parquet" else pd.read_csv(url)
    except Exception as exc:
        raise RuntimeError(f"Cannot download {url}: {exc}") from exc

def build(season, week, output):
    stats_url = f"{BASE}/stats_player/stats_player_week_{season}.parquet"
    stats = fetch(stats_url, "parquet")
    schedule = fetch(f"{BASE}/schedules/games.csv", "csv")
    if "team" in stats.columns and "recent_team" not in stats.columns:
        stats = stats.rename(columns={"team":"recent_team"})
    for col in ("season","week"):
        stats[col] = pd.to_numeric(stats[col], errors="coerce")
        schedule[col] = pd.to_numeric(schedule[col], errors="coerce")
    stats = stats[(stats.season == season) & (stats.week < week)].copy()
    if "season_type" in stats.columns:
        stats = stats[stats.season_type == "REG"]
    schedule = schedule[(schedule.season == season) & (schedule.game_type == "REG")].copy()
    if stats.empty:
        raise ValueError("No completed player games available before the requested week")
    needed = ["player_id","player_display_name","position","recent_team"] + FIELDS
    absent = [x for x in needed if x not in stats.columns]
    if absent:
        raise ValueError(f"Source schema changed; missing columns: {absent}")
    for metric in FIELDS:
        stats[metric] = pd.to_numeric(stats[metric], errors="coerce").fillna(0)
    matchups = {}
    for _, game in schedule.iterrows():
        matchups[(int(game.week),game.home_team)] = game.away_team
        matchups[(int(game.week),game.away_team)] = game.home_team
    stats["opponent"] = [matchups.get((int(w),t)) for w,t in zip(stats.week,stats.recent_team)]
    if stats.opponent.notna().sum() == 0:
        raise ValueError("No historical player rows could be matched to NFL schedule")
    upcoming = schedule[schedule.week == week]
    current = {}
    for _, game in upcoming.iterrows():
        current[game.home_team] = game.away_team
        current[game.away_team] = game.home_team
    if not current:
        raise ValueError("No scheduled games in the selected week")
    # Defense totals by opposing team and week, then average across games.
    allowed = stats.dropna(subset=["opponent"]).groupby(["opponent","week"],as_index=False)[FIELDS].sum()
    defense = allowed.groupby("opponent")[FIELDS].mean()
    player = stats.groupby(["player_id","player_display_name","position","recent_team"],as_index=False).agg(
        games=("week","nunique"),**{c:(c,"sum") for c in FIELDS})
    team_totals = stats.groupby("recent_team")[list(set(METRICS.values()))].sum()
    rows = []
    for _, p in player.iterrows():
        team, opp = p.recent_team, current.get(p.recent_team)
        if opp is None or opp not in defense.index or team not in team_totals.index or p.games == 0:
            continue
        row = {"player":p.player_display_name,"team":team,"opponent":opp,"position":p.position,"games":int(p.games)}
        for metric, opportunity in METRICS.items():
            player_avg = p[metric] / p.games
            team_opportunities = team_totals.loc[team,opportunity]
            share = p[opportunity] / team_opportunities if team_opportunities else 0
            opponent_adj = defense.loc[opp,metric] * share
            row[metric] = round((player_avg + opponent_adj) / 2,2)
        rows.append(row)
    if not rows:
        raise ValueError("No projections created: check schedule and player data")
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).sort_values(["team","position","player"]).to_csv(output,index=False)
    print(f"Created {len(rows)} player performance forecasts: {output}")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    ap.add_argument("--output",default="output/projections.csv")
    a=ap.parse_args()
    build(a.season,a.week,a.output)
