"""NFL player performance forecasts (non-wagering research baseline)."""
import argparse
from pathlib import Path
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
METRICS = {"passing_yards":"attempts","rushing_yards":"carries","receiving_yards":"targets","receptions":"targets"}
TD_FIELDS = ["passing_tds", "rushing_tds", "receiving_tds"]
FIELDS = sorted(set(METRICS) | set(METRICS.values()) | set(TD_FIELDS))

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
    stats = stats[stats["position"].isin(["QB", "RB", "WR", "TE"])].copy()
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
        pass_share = p["attempts"] / team_totals.loc[team, "attempts"] if team_totals.loc[team, "attempts"] else 0
        rush_share = p["carries"] / team_totals.loc[team, "carries"] if team_totals.loc[team, "carries"] else 0
        receive_share = p["targets"] / team_totals.loc[team, "targets"] if team_totals.loc[team, "targets"] else 0
        if p.position == "QB":
            row["passing_tds"] = round((p["passing_tds"] / p.games + defense.loc[opp, "passing_tds"] * pass_share) / 2, 2)
        rush_tds = (p["rushing_tds"] / p.games + defense.loc[opp, "rushing_tds"] * rush_share) / 2
        receiving_tds = (p["receiving_tds"] / p.games + defense.loc[opp, "receiving_tds"] * receive_share) / 2
        row["rushing_receiving_tds"] = round(rush_tds + receiving_tds, 2)
        row["receiving_tds"] = round(receiving_tds, 2)
        if p.position in ("WR", "TE"):
            row["tds"] = round(receiving_tds, 2)
            row.pop("rushing_yards", None)
            row.pop("rushing_receiving_tds", None)
        rows.append(row)
    if not rows:
        raise ValueError("No projections created: check schedule and player data")
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["team", "player"])
    q = stats[stats.position.eq("QB")].copy()
    last = q.groupby("recent_team").week.transform("max")
    recent = q[q.week.eq(last)].groupby(["recent_team","player_id"]).attempts.sum().rename("recent")
    total = q.groupby(["recent_team","player_id"]).attempts.sum().rename("total")
    ranks = pd.concat([recent,total],axis=1).fillna(0).reset_index()
    best = ranks.sort_values(["recent_team","recent","total"],ascending=[True,False,False]).drop_duplicates("recent_team")
    chosen = set(zip(best.recent_team,best.player_id))
    names = set(zip(q[q.apply(lambda x:(x.recent_team,x.player_id) in chosen,axis=1)].recent_team,q[q.apply(lambda x:(x.recent_team,x.player_id) in chosen,axis=1)].player_display_name))
    result = result[(result.position != "QB") | result.apply(lambda x:(x.team,x.player) in names,axis=1)].copy()
    layouts = {
        "QB": ["player","team","opponent","games","passing_yards","passing_tds","rushing_yards","rushing_receiving_tds"],
        "RB": ["player","team","opponent","games","rushing_yards","receiving_yards","receptions","rushing_receiving_tds"],
        "WR": ["player","team","opponent","games","receptions","receiving_yards","tds"],
        "TE": ["player","team","opponent","games","receptions","receiving_yards","receiving_tds"],
    }
    csv_path = Path(output).with_suffix(".csv")
    result.to_csv(csv_path, index=False)
    print(f"Saved {len(result)} NFL performance projections: {csv_path}")
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for position, columns in layouts.items():
            subset = result.loc[result.position == position, columns]
            subset.to_excel(writer, sheet_name=position, index=False)
            sheet = writer.sheets[position]
            sheet.freeze_panes(1, 4)
            sheet.autofilter(0, 0, len(subset), len(columns)-1)
            sheet.set_column(0, 0, 27)
            sheet.set_column(1, 2, 15)
            sheet.set_column(3, len(columns)-1, 19)
            header = writer.book.add_format({"bold": True, "bg_color": "#17365D", "font_color": "#FFFFFF"})
            for idx, name in enumerate(columns):
                sheet.write(0, idx, "TDs" if name == "tds" else name.replace("_", " ").title().replace("Tds", "TDs"), header)

        breakdown = writer.book.add_worksheet("Game Breakdown")
        breakdown.set_column("A:N", 16)
        breakdown.set_column("A:A", 26)
        breakdown.set_column("F:F", 26)
        breakdown.set_column("K:K", 26)
        title = writer.book.add_format({"bold":True,"font_size":14,"bg_color":"#17365D","font_color":"white"})
        head = writer.book.add_format({"bold":True,"bg_color":"#DCE6F1"})
        for game_index, (_, game) in enumerate(upcoming.sort_values("gameday").iterrows()):
            base_row = game_index * 42
            away, home = game.away_team, game.home_team
            group = result[result.team.isin([away,home])]
            breakdown.merge_range(base_row,0,base_row,13,f"{away} @ {home} | {game.get('gameday','')}",title)
            specs = [
                ("Passing",0,group[group.position.eq("QB")],["player","team","passing_yards","passing_tds"]),
                ("Rushing",5,group[group.position.isin(["QB","RB","WR"])],["player","team","rushing_yards","rushing_receiving_tds"]),
                ("Receiving",10,group[group.position.isin(["RB","WR","TE"])],["player","team","receiving_yards","receptions"]),
            ]
            for label,col,frame,cols in specs:
                frame = frame.sort_values(cols[2],ascending=False).head(15)
                breakdown.write(base_row+2,col,label,head)
                for j,key in enumerate(cols):
                    breakdown.write(base_row+3,col+j,key.replace("_"," ").title(),head)
                for i,(_,rec) in enumerate(frame.iterrows()):
                    for j,key in enumerate(cols):
                        value=rec[key]
                        if key in ("player","team"):
                            breakdown.write_string(base_row+4+i,col+j,str(value))
                        else:
                            breakdown.write_number(base_row+4+i,col+j,float(value))
                if len(frame):
                    chart=writer.book.add_chart({"type":"column"})
                    chart.add_series({"name":cols[2],"categories":["Game Breakdown",base_row+4,col,base_row+3+len(frame),col],"values":["Game Breakdown",base_row+4,col+2,base_row+3+len(frame),col+2]})
                    chart.set_title({"name":f"{label}: {away} @ {home}"})
                    chart.set_legend({"none":True})
                    chart.set_size({"width":430,"height":250})
                    breakdown.insert_chart(base_row+21,col,chart)
    print(f"Created {len(result)} player performance forecasts: {output}")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    ap.add_argument("--output",default="output/projections.xlsx")
    a=ap.parse_args()
    build(a.season,a.week,a.output)
