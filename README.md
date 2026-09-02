# Full Court

NBA and WNBA analytics, 2003 to now. Live at <https://full-court-six.vercel.app>

![Player overview](docs/screenshot.png)

Stats side: player pages with percentiles and career trends, comparison, season
similarity, shot charts, team tables, five-man lineups, and an explorer over
every player-season. The Ask box turns a question turns natural language
questions into queries.

Predictions side: a game calendar with win probabilities and projected player
lines, Elo ratings, and a search for any player's next season.

| | NBA | WNBA |
|---|---|---|
| Game accuracy since 2015 | 65.5% | 65.8% |
| Always picking home | 56.5% | 55.3% |
| Points off by | 2.42 | 2.50 |
| Repeating last season, off by | 2.53 | 2.66 |

Season averages don't add up to a real game, so lines get refit to one: 240 team
minutes (200 in the WNBA) split by what each player earned, totals pulled toward
what teams score, anyone ruled out left out. That buys a team total that adds
up and costs about 0.2 points of per-player accuracy, which the game backtest
reports rather than hides.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-local.txt
python etl/sdv_etl.py --league nba
python etl/sdv_etl.py --league wnba
python etl/schedule_etl.py
python etl/lineup_etl.py --league nba
python etl/lineup_etl.py --league wnba
cd frontend && npm install && cd ..

uvicorn backend.main:app --reload --port 8000   # one terminal
cd frontend && npm run dev                      # another, opens :5173
```

Data is ESPN's via hoopR and wehoop, about 17 MB of Parquet in `data/`, and it's
committed, so the deployed API calls nothing.

## Worth knowing

- NBA seasons are labelled by the year they end, WNBA by their own: `2024` is
  2023-24 in one, 2024 in the other.
- ESPN publishes no possession data, so ortg/drtg/pace are estimated with
  Oliver's formula rather than counted.
- It publishes no lineup data either, but it does publish substitutions, so
  lineups are rebuilt by walking them. `--validate` checks the rebuilt minutes
  against the box score: they land within a minute for 99.5% of NBA
  player-games in 2025, falling to 92% in 2015 as the older feeds log a third
  fewer subs. The WNBA feed is unusable before 2020 and near-perfect after, so
  the two leagues start in different years. Fives that played under five
  minutes together aren't stored.
- The injury feed is a snapshot with no history, so it decides who plays but
  can't be backtested, and rookies have nothing to project from.
- Similarity scores cluster in the high 90s: trust the order, not the number.
