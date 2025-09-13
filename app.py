import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
try:
    from nba_api.stats.endpoints import shotchartdetail
    from nba_api.stats.static import players as players_static
except Exception:
    shotchartdetail = None
    players_static = None

st.set_page_config(page_title="NBA Data Explorer", layout="wide")
st.title("NBA Data Explorer")

DATA_DIR = "data"
LEAGUE_ID = "00"  # only want NBA data for now


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        st.error(f"Missing {name}. Run the ETL first:  `python etl/nba_etl.py`")
        st.stop()
    return pd.read_parquet(path)

players = load_parquet("players.parquet")
teams   = load_parquet("teams.parquet")
shots_path = os.path.join(DATA_DIR, "shots.parquet")
shots   = pd.read_parquet(shots_path) if os.path.exists(shots_path) else pd.DataFrame()

for col in ["season"]:
    if col in players.columns: players[col] = players[col].astype(str)
    if col in teams.columns:   teams[col]   = teams[col].astype(str)
    if "season" in shots.columns: shots["season"] = shots["season"].astype(str)

player_names = sorted(players.get("player_name", pd.Series(dtype=str)).dropna().unique().tolist())
team_names   = sorted(teams.get("team_name", pd.Series(dtype=str)).dropna().unique().tolist())
seasons_all  = sorted(players.get("season", pd.Series(dtype=str)).dropna().unique().tolist())

def compute_season_range(seasons):
    if len(seasons) >= 2: return (seasons[0], seasons[-1])
    if len(seasons) == 1: return (seasons[0], seasons[0])
    return (None, None)

def allowed_seasons(all_seasons, lo, hi):
    if not all_seasons or lo is None or hi is None: return []
    try:
        i, j = all_seasons.index(lo), all_seasons.index(hi)
        if i > j: i, j = j, i
        return all_seasons[i:j+1]
    except ValueError:
        return all_seasons

class ShotFetchError(Exception): pass

@retry(reraise=True, retry=retry_if_exception_type(Exception),
       stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.8, min=1, max=8))
def _first_df(ep):
    dfs = ep.get_data_frames()
    if not dfs: raise ShotFetchError("NBA API returned no data")
    return dfs[0]

def find_player_id_by_name(name: str):
    if players_static is None: return None
    hits = players_static.find_players_by_full_name(name) or []
    return hits[0]["id"] if hits else None

def fetch_player_shots_df(player_id: int, season: str):
    if shotchartdetail is None:
        raise ShotFetchError("nba_api not installed in this environment.")
    try:
        ep = shotchartdetail.ShotChartDetail(
            team_id=0, player_id=player_id,
            season_nullable=season, context_measure_simple="FGA",
            league_id_nullable=LEAGUE_ID,
        )
    except TypeError:
        ep = shotchartdetail.ShotChartDetail(
            team_id=0, player_id=player_id,
            season_nullable=season, context_measure_simple="FGA",
        )
    raw = _first_df(ep)
    rename = {
        "PLAYER_ID":"player_id","PLAYER_NAME":"player_name","TEAM_ID":"team_id",
        "GAME_ID":"game_id","GAME_DATE":"game_date","LOC_X":"x","LOC_Y":"y",
        "SHOT_MADE_FLAG":"made","SHOT_ZONE_BASIC":"zone","SHOT_DISTANCE":"distance",
        "PERIOD":"period","SHOT_CLOCK":"shot_clock",
    }
    have = [c for c in rename if c in raw.columns]
    df = raw[have].rename(columns={k:v for k,v in rename.items() if k in have})
    for want in rename.values():
        if want not in df.columns:
            df[want] = pd.NA
    df["season"] = str(season)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["made"] = pd.to_numeric(df["made"], errors="coerce").fillna(0).astype(int)
    return df

def append_shots_to_parquet(new_df: pd.DataFrame, path: str):
    if not len(new_df): return
    if os.path.exists(path):
        old = pd.read_parquet(path)
        combo = pd.concat([old, new_df], ignore_index=True)
        combo = combo.drop_duplicates(subset=["player_id","game_id","x","y","season"])
        combo.to_parquet(path, index=False)
    else:
        new_df.to_parquet(path, index=False)


def draw_plotly_court(fig, fig_width=600, margins=10):

    import numpy as np
        
    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

    threept_break_y = 89.47765084
    three_line_col = "#777777"
    main_line_col = "#777777"

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=[
            dict(type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5, line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,   line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,   line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="circle", x0=-60, y0=77.5, x1=60, y1=197.5,  xref="x", yref="y", line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=-60, y0=137.5, x1=60, y1=137.5,   line=dict(color=main_line_col, width=1), layer='below'),

            dict(type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5, line=dict(color="#ec7607", width=1), fillcolor='#ec7607'),
            dict(type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y", line=dict(color="#ec7607", width=1)),
            dict(type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5, line=dict(color="#ec7607", width=1)),

            dict(type="path", path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi), line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="path", path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101), line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y, line=dict(color=three_line_col, width=1), layer='below'),
            dict(type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y, line=dict(color=three_line_col, width=1), layer='below'),
            dict(type="line", x0=220,  y0=-52.5, x1=220,  y1=threept_break_y, line=dict(color=three_line_col, width=1), layer='below'),

            dict(type="line", x0=-250, y0=227.5, x1=-220, y1=227.5, line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=250,  y0=227.5, x1=220,  y1=227.5, line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=-90,  y0=17.5,  x1=-80,  y1=17.5,  line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=-90,  y0=27.5,  x1=-80,  y1=27.5,  line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=-90,  y0=57.5,  x1=-80,  y1=57.5,  line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=-90,  y0=87.5,  x1=-80,  y1=87.5,  line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=90,   y0=17.5,  x1=80,   y1=17.5,  line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=90,   y0=27.5,  x1=80,   y1=27.5,  line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=90,   y0=57.5,  x1=80,   y1=57.5,  line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=90,   y0=87.5,  x1=80,   y1=87.5,  line=dict(color=main_line_col, width=1), layer='below'),

            dict(type="path", path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
        ]
    )
    return True

st.sidebar.header("Global Filters")
if len(seasons_all) >= 2:
    season_min, season_max = compute_season_range(seasons_all)
    season_range = st.sidebar.select_slider(
        "Season range (for player aggregation)",
        options=seasons_all, value=(season_min, season_max)
    )
elif len(seasons_all) == 1:
    season_range = (seasons_all[0], seasons_all[0])
    st.sidebar.info(f"Only one season available: {seasons_all[0]}")
else:
    season_range = (None, None)
    st.sidebar.warning("No seasons found in players.parquet")
if len(seasons_all) <= 1:
    st.caption("ℹ️ Only one season detected. Add more seasons in ETL for richer trends.")

# Tabs
tab_compare, tab_trends, tab_similar, tab_teams, tab_shots = st.tabs(
    ["Player Comparison", "Trends", "Similar Players", "Teams", "Shot Charts"]
)

# Player Comparison
with tab_compare:
    st.subheader("Compare players on advanced metrics")
    candidate_metrics = ["ts_pct","usg_pct","ortg","drtg","pts","ast","reb","stl","blk","tov","fg_pct","three_pct","ft_pct","per","bpm"]
    available_metrics = [m for m in candidate_metrics if m in players.columns]
    names = st.multiselect("Players", player_names, max_selections=5)
    default_metrics = [m for m in ["ts_pct", "ortg", "drtg"] if m in available_metrics]
    metrics = st.multiselect("Metrics to plot", available_metrics, default=default_metrics)
    if not available_metrics:
        st.info("No comparable metrics found yet.")
    elif not names or not metrics:
        st.info("Select at least one player and one metric.")
    else:
        df = players[players["player_name"].isin(names)].copy()
        allowed = allowed_seasons(seasons_all, *season_range)
        if allowed: df = df[df["season"].isin(allowed)]
        if df.empty:
            st.info("No data for selected players/seasons.")
        else:
            df = df[["player_name","season"] + metrics].sort_values(["player_name","season"])
            if df["season"].nunique() == 1:
                season_label = df["season"].iloc[0]
                melted = df.melt(id_vars=["player_name"], value_vars=metrics, var_name="metric", value_name="value")
                fig = px.bar(melted, x="player_name", y="value", color="metric",
                             barmode="group", title=f"Comparison ({season_label})")
            else:
                melted = df.melt(id_vars=["player_name","season"], value_vars=metrics, var_name="metric", value_name="value")
                fig = px.line(melted, x="season", y="value", color="player_name",
                              facet_row="metric", markers=True, height=min(900, 260*len(metrics)))
            st.plotly_chart(fig, use_container_width=True)

with tab_trends:
    st.subheader("Player trendlines across seasons (PTS / AST / REB / TS%)")
    if not player_names:
        st.info("No players found.")
    else:
        name = st.selectbox("Player", player_names, key="trend_player")
        trend_cols = [c for c in ["pts","ast","reb","ts_pct"] if c in players.columns]
        if not trend_cols:
            st.info("Trend metrics not available yet.")
        else:
            pdf = players[players["player_name"] == name][["season"] + trend_cols].sort_values("season")
            if pdf.empty:
                st.info("No data for that player.")
            else:
                melted = pdf.melt(id_vars="season", value_vars=trend_cols, var_name="metric", value_name="value")
                fig = px.line(melted, x="season", y="value", color="metric", markers=True, title=name)
                st.plotly_chart(fig, use_container_width=True)

with tab_similar:
    st.subheader("Statistically similar players (cosine similarity)")
    desired = ["pts","ast","reb","tov","ts_pct","usg_pct","ortg","drtg"]
    feats = [f for f in desired if f in players.columns]
    st.caption(f"Features used: {', '.join(feats) if feats else 'None available'}")
    if not player_names:
        st.info("No players found.")
    else:
        name_sim = st.selectbox("Anchor player", player_names, key="sim_player")
        k = st.slider("Top-K", 3, 10, 5)
        if feats:
            df = players.copy()
            allowed = allowed_seasons(seasons_all, *season_range)
            if allowed: df = df[df["season"].isin(allowed)]
            agg = df.groupby("player_name", as_index=False)[feats].mean(numeric_only=True)
            agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if name_sim not in agg["player_name"].values:
                st.info("Selected player has no stats in the chosen range.")
            else:
                X = agg[feats].to_numpy(dtype=float)
                Xz = StandardScaler().fit_transform(X)
                sim = cosine_similarity(Xz)
                idx = agg.index[agg["player_name"] == name_sim][0]
                scores = sim[idx]; order = np.argsort(-scores)
                rows = []
                for i in order:
                    if i == idx: continue
                    rows.append({"player_name": agg.iloc[i]["player_name"], "similarity": float(scores[i])})
                    if len(rows) >= k: break
                st.dataframe(pd.DataFrame(rows))
        else:
            st.info("Not enough features yet for similarity.")

# Teams
with tab_teams:
    st.subheader("Team stats dashboard")
    if not team_names:
        st.info("No teams in teams.parquet yet.")
    else:
        tname = st.selectbox("Team", team_names)
        tdf = teams[teams["team_name"] == tname].copy()
        for c in ["wins","losses","FG_PCT","FT_PCT","FG3_PCT","ortg","drtg","pace","PTS"]:
            if c in tdf.columns: tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
        tdf = tdf.sort_values("season")
        c1, c2 = st.columns(2)
        with c1:
            if {"wins","losses"}.issubset(tdf.columns):
                st.plotly_chart(px.line(tdf, x="season", y=["wins","losses"], markers=True,
                                        title=f"{tname} — Wins/Losses"), use_container_width=True)
            else:
                st.info("Wins/Losses not available in teams.parquet")
        with c2:
            ycols = [c for c in ["ortg","drtg","pace"] if c in tdf.columns]
            if ycols:
                st.plotly_chart(px.line(tdf, x="season", y=ycols, markers=True,
                                        title=f"{tname} — ORtg/DRtg/Pace"), use_container_width=True)
            else:
                st.info("Ratings/Pace not available.")

with tab_shots:
    st.subheader("Shot chart (live from NBA API)")

    q = st.text_input("Search player (full or partial name)", value="Stephen Curry")
    season_options = sorted(players["season"].dropna().unique().tolist())
    s_sel = st.selectbox("Season", season_options, index=len(season_options)-1 if season_options else 0)


    def resolve_player_id(query: str):
        pid = find_player_id_by_name(query)
        if pid or players_static is None:
            return pid, query  
        pool = players_static.get_players()
        matches = [p for p in pool if query.lower() in p["full_name"].lower()]
        if len(matches) == 1:
            return matches[0]["id"], matches[0]["full_name"]
        elif len(matches) > 1:
            choice = st.selectbox("Multiple matches found — pick one:", [m["full_name"] for m in matches])
            if choice:
                chosen = next(m for m in matches if m["full_name"] == choice)
                return chosen["id"], chosen["full_name"]
        return None, query

    go_btn = st.button("Fetch & Plot", type="primary")

    if go_btn:
        if not q or not s_sel:
            st.warning("Enter a player name and select a season.")
        else:
            pid, resolved_name = resolve_player_id(q)
            if not pid:
                st.error("Player not found. Try a more complete name (e.g., 'Nikola Jokic').")
            else:
                with st.spinner(f"Fetching shots for {resolved_name} — {s_sel} …"):
                    try:
                        sdf = fetch_player_shots_df(pid, s_sel)

                        for c in ["x", "y", "made"]:
                            if c in sdf.columns:
                                sdf[c] = pd.to_numeric(sdf[c], errors="coerce")
                        sdf = sdf.dropna(subset=["x", "y"])

                        if not sdf.empty:
                            xmax, ymax = float(sdf["x"].abs().max()), float(sdf["y"].abs().max())
                            if xmax <= 60 and ymax <= 120:
                                sdf["x"] *= 10.0
                                sdf["y"] *= 10.0

                        miss = sdf[sdf["made"] == 0]
                        make = sdf[sdf["made"] == 1]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=miss["x"], y=miss["y"], mode="markers",
                            name="Miss", marker=dict(symbol="x", size=6, opacity=0.55)
                        ))
                        fig.add_trace(go.Scatter(
                            x=make["x"], y=make["y"], mode="markers",
                            name="Make", marker=dict(size=6, opacity=0.85)
                        ))

                        draw_plotly_court(fig, fig_width=900, margins=10)
                        fig.update_layout(title=f"{resolved_name} — {s_sel} Shot Chart")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to fetch or plot shots: {e}")