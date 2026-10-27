# NFL Player Prop Projection Model

A research-oriented NFL player-stat projection pipeline using historical weekly player statistics and opponent defensive averages.

## Quick start

Install Python 3.11, then:

```bash
pip install -r requirements.txt
python model.py --season 2026 --week 9 --output output/projections.csv
```

## Automated workflow

GitHub Actions > NFL weekly projections > Run workflow. Set season and week, or use the Tuesday scheduled run. Download the generated CSV from the workflow's Artifacts section.

## Method

For passing yards, rushing yards, receiving yards and receptions, the model blends a player's historical per-game output (50%) with opponent team per-game allowed output multiplied by that player's historical share of team opportunities (50%). It does not yet account for injuries, lineup changes, recent usage, opponent position splits, or uncertainty.

## Important limitations

This is an initial baseline, not a validated forecasting system. The upstream nfl_data_py package and weekly data availability may change. The GitHub Actions workflow must complete successfully before any projection output can be considered generated. Backtesting and calibration are still required.
