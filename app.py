import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import binned_statistic_2d

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
try:
    from nba_api.stats.endpoints import shotchartdetail, playergamelog, commonplayerinfo
    from nba_api.stats.static import players as players_static
except Exception:
    shotchartdetail = None
    playergamelog = None
    commonplayerinfo = None
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

CANDIDATE_METRICS = ["ts_pct","usg_pct","ortg","drtg","pts","ast","reb","stl","blk","tov","fg_pct","three_pct","ft_pct","per","bpm"]
AVAILABLE_METRICS = [m for m in CANDIDATE_METRICS if m in players.columns]
INVERT_METRICS = {"drtg", "tov"}

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

NBA_TIMEOUT = 75  # stats.nba.com is flaky; default 30s often trips

@retry(reraise=True, retry=retry_if_exception_type(Exception),
       stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1.0, min=2, max=20))
def _call_endpoint(endpoint_cls, **kwargs):
    """Instantiate an nba_api endpoint and return its first data frame.
    Both construction and get_data_frames can raise; we retry both together."""
    kwargs.setdefault("timeout", NBA_TIMEOUT)
    try:
        ep = endpoint_cls(**kwargs)
    except TypeError:
        # some older endpoints don't accept `timeout` kwarg
        kwargs.pop("timeout", None)
        ep = endpoint_cls(**kwargs)
    dfs = ep.get_data_frames()
    if not dfs: raise ShotFetchError("NBA API returned no data")
    return dfs[0]

def find_player_id_by_name(name: str):
    if players_static is None: return None
    hits = players_static.find_players_by_full_name(name) or []
    return hits[0]["id"] if hits else None

@st.cache_data(show_spinner=False)
def fetch_player_shots_df(player_id: int, season: str):
    if shotchartdetail is None:
        raise ShotFetchError("nba_api not installed in this environment.")
    try:
        raw = _call_endpoint(
            shotchartdetail.ShotChartDetail,
            team_id=0, player_id=player_id,
            season_nullable=season, context_measure_simple="FGA",
            league_id_nullable=LEAGUE_ID,
        )
    except TypeError:
        raw = _call_endpoint(
            shotchartdetail.ShotChartDetail,
            team_id=0, player_id=player_id,
            season_nullable=season, context_measure_simple="FGA",
        )
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


@st.cache_data(show_spinner=False)
def fetch_player_gamelog(player_id: int, season: str) -> pd.DataFrame:
    if playergamelog is None:
        raise ShotFetchError("nba_api not installed.")
    raw = _call_endpoint(playergamelog.PlayerGameLog, player_id=player_id, season=season)
    if "GAME_DATE" in raw.columns:
        raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"], errors="coerce")
        raw = raw.sort_values("GAME_DATE").reset_index(drop=True)
    return raw


@st.cache_data(show_spinner=False)
def fetch_player_birthdate(player_id: int):
    if commonplayerinfo is None: return None
    try:
        df = _call_endpoint(commonplayerinfo.CommonPlayerInfo, player_id=player_id)
        return pd.to_datetime(df.iloc[0].get("BIRTHDATE"), errors="coerce")
    except Exception:
        return None


def season_start_year(season: str):
    try:
        return int(str(season).split("-")[0])
    except Exception:
        return None


def age_for_season(birthdate, season: str):
    y = season_start_year(season)
    if birthdate is None or pd.isna(birthdate) or y is None: return None
    ref = pd.Timestamp(year=y, month=10, day=1)
    return round((ref - birthdate).days / 365.25, 1)


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

tab_compare, tab_trends, tab_pct, tab_similar, tab_gamelog, tab_age, tab_teams, tab_shots = st.tabs(
    ["Compare", "Trends", "Percentiles", "Similar Players", "Game Log", "Age Curves", "Teams", "Shot Charts"]
)

# Player Comparison
with tab_compare:
    st.subheader("Compare players on advanced metrics")
    names = st.multiselect("Players", player_names, max_selections=5)
    default_metrics = [m for m in ["ts_pct", "ortg", "drtg"] if m in AVAILABLE_METRICS]
    metrics = st.multiselect("Metrics to plot", AVAILABLE_METRICS, default=default_metrics)
    if not AVAILABLE_METRICS:
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
                view = st.radio("View", ["Radar", "Bar"], horizontal=True, key="cmp_view")
                season_label = df["season"].iloc[0]
                if view == "Bar":
                    melted = df.melt(id_vars=["player_name"], value_vars=metrics, var_name="metric", value_name="value")
                    fig = px.bar(melted, x="player_name", y="value", color="metric",
                                 barmode="group", title=f"Comparison ({season_label})")
                else:
                    pool = players[players["season"] == season_label]
                    scaled = df.set_index("player_name")[metrics].copy()
                    for m in metrics:
                        col = pd.to_numeric(pool[m], errors="coerce")
                        lo, hi = float(col.min()), float(col.max())
                        if hi > lo:
                            v = (pd.to_numeric(scaled[m], errors="coerce") - lo) / (hi - lo)
                            if m in INVERT_METRICS: v = 1 - v
                            scaled[m] = v
                        else:
                            scaled[m] = 0.5
                    fig = go.Figure()
                    for pname, row in scaled.iterrows():
                        vals = row.tolist()
                        fig.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]],
                            theta=metrics + [metrics[0]],
                            fill="toself", name=str(pname), opacity=0.55,
                        ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                        title=f"Comparison — {season_label} (league-normalized; drtg/tov inverted)",
                        height=560,
                    )
                st.plotly_chart(fig, use_container_width=True)
            else:
                melted = df.melt(id_vars=["player_name","season"], value_vars=metrics, var_name="metric", value_name="value")
                fig = px.line(melted, x="season", y="value", color="player_name",
                              facet_row="metric", markers=True, height=min(900, 260*len(metrics)))
                fig.update_yaxes(matches=None)
                st.plotly_chart(fig, use_container_width=True)

with tab_trends:
    st.subheader("Player trendlines across seasons")
    if not player_names:
        st.info("No players found.")
    elif not AVAILABLE_METRICS:
        st.info("No metrics available yet.")
    else:
        name = st.selectbox("Player", player_names, key="trend_player")
        default_trend = [m for m in ["pts","ast","reb","ts_pct"] if m in AVAILABLE_METRICS]
        trend_cols = st.multiselect("Metrics", AVAILABLE_METRICS, default=default_trend, key="trend_metrics")
        show_league = st.checkbox("Overlay league average", value=True, key="trend_league")
        if not trend_cols:
            st.info("Pick at least one metric.")
        else:
            pdf = players[players["player_name"] == name][["season"] + trend_cols].sort_values("season")
            if pdf.empty:
                st.info("No data for that player.")
            else:
                melted = pdf.melt(id_vars="season", value_vars=trend_cols, var_name="metric", value_name="value")
                melted["series"] = name
                parts = [melted]
                if show_league:
                    lg = players.groupby("season")[trend_cols].mean(numeric_only=True).reset_index()
                    lg_m = lg.melt(id_vars="season", value_vars=trend_cols, var_name="metric", value_name="value")
                    lg_m["series"] = "League avg"
                    parts.append(lg_m)
                plot_df = pd.concat(parts, ignore_index=True)
                fig = px.line(plot_df, x="season", y="value", color="series",
                              facet_row="metric", markers=True,
                              height=min(900, 260*len(trend_cols)), title=name)
                fig.update_yaxes(matches=None)
                st.plotly_chart(fig, use_container_width=True)

with tab_pct:
    st.subheader("League percentile rankings")
    if not AVAILABLE_METRICS:
        st.info("No metrics available.")
    elif not player_names:
        st.info("No players found.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            pname = st.selectbox("Player", player_names, key="pct_player")
        player_seasons = sorted(players[players["player_name"] == pname]["season"].unique())
        with col2:
            pseason = st.selectbox("Season", player_seasons, key="pct_season") if player_seasons else None
        if not pseason:
            st.info("No seasons for that player.")
        else:
            sub = players[(players["player_name"] == pname) & (players["season"] == pseason)]
            if sub.empty:
                st.info("Player has no row for that season.")
            else:
                row = sub.iloc[0]
                pool = players[players["season"] == pseason].copy()
                rows = []
                for m in AVAILABLE_METRICS:
                    col = pd.to_numeric(pool[m], errors="coerce")
                    if col.notna().sum() < 10: continue
                    # pct=True returns rank/N in (0,1]; ascending=True means largest value -> 1.0
                    # For invert metrics (lower is better) we want smallest value -> 1.0, so ascending=False.
                    rank = col.rank(pct=True, ascending=(m not in INVERT_METRICS))
                    idx = pool.index[pool["player_name"] == pname]
                    if not len(idx): continue
                    pct = float(rank.loc[idx[0]])
                    val = float(row[m]) if pd.notna(row[m]) else None
                    rows.append({"metric": m, "value": val, "percentile": round(pct*100, 1)})
                pdf = pd.DataFrame(rows).dropna(subset=["percentile"])
                if pdf.empty:
                    st.info("No percentile data.")
                else:
                    pdf = pdf.sort_values("percentile", ascending=True)
                    labels = [f"{v:.2f}" if (v is not None and abs(v) < 10) else (f"{v:.1f}" if v is not None else "")
                              for v in pdf["value"]]
                    fig = go.Figure(go.Bar(
                        x=pdf["percentile"], y=pdf["metric"],
                        orientation="h", text=labels,
                        textposition="outside",
                        marker=dict(
                            color=pdf["percentile"],
                            colorscale=[[0,"#d73027"],[0.5,"#ffffbf"],[1,"#1a9850"]],
                            cmin=0, cmax=100,
                            colorbar=dict(title="pct"),
                        ),
                    ))
                    fig.update_layout(
                        title=f"{pname} — {pseason} vs. all NBA",
                        xaxis=dict(range=[0,110], title="Percentile (100 = best)"),
                        height=max(340, 32*len(pdf)+100),
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("drtg and tov are inverted (lower = better, higher percentile).")

with tab_similar:
    st.subheader("Statistically similar players (weighted cosine similarity)")
    desired = ["pts","ast","reb","tov","ts_pct","usg_pct","ortg","drtg"]
    feats = [f for f in desired if f in players.columns]
    st.caption(f"Features used: {', '.join(feats) if feats else 'None available'}")
    if not player_names:
        st.info("No players found.")
    elif not feats:
        st.info("Not enough features yet for similarity.")
    else:
        name_sim = st.selectbox("Anchor player", player_names, key="sim_player")
        k = st.slider("Top-K", 3, 10, 5)
        with st.expander("Feature weights", expanded=False):
            st.caption("Drag to emphasize metrics in the similarity calculation (0 = ignore).")
            wcols = st.columns(min(4, len(feats)))
            weights = {}
            for i, f in enumerate(feats):
                with wcols[i % len(wcols)]:
                    weights[f] = st.slider(f, 0.0, 2.0, 1.0, 0.1, key=f"w_{f}")

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
            w = np.array([weights[f] for f in feats], dtype=float)
            Xz = Xz * w
            sim = cosine_similarity(Xz)
            idx = agg.index[agg["player_name"] == name_sim][0]
            scores = sim[idx]; order = np.argsort(-scores)
            rows = []
            for i in order:
                if i == idx: continue
                rows.append({"player_name": agg.iloc[i]["player_name"], "similarity": float(scores[i])})
                if len(rows) >= k: break
            sim_df = pd.DataFrame(rows)
            st.dataframe(sim_df, use_container_width=True)

            # Radar overlay
            pool_min = agg[feats].min()
            pool_max = agg[feats].max()
            def norm_row(pname):
                r = agg.loc[agg["player_name"] == pname, feats].iloc[0]
                out = []
                for f in feats:
                    lo, hi = float(pool_min[f]), float(pool_max[f])
                    v = (float(r[f]) - lo) / (hi - lo) if hi > lo else 0.5
                    if f in INVERT_METRICS: v = 1 - v
                    out.append(v)
                return out

            fig = go.Figure()
            for pname in [name_sim] + [r["player_name"] for r in rows]:
                vals = norm_row(pname)
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=feats + [feats[0]],
                    fill="toself", name=pname,
                    opacity=0.65 if pname == name_sim else 0.35,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                title=f"{name_sim} vs. top-{k} matches (league-normalized; drtg/tov inverted)",
                height=580,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_gamelog:
    st.subheader("Game log explorer (live from NBA API)")
    if players_static is None or playergamelog is None:
        st.info("nba_api not available.")
    else:
        gq = st.text_input("Player", value="Stephen Curry", key="gl_q")
        g_season = st.selectbox("Season", seasons_all,
                                index=len(seasons_all)-1 if seasons_all else 0, key="gl_season")
        gl_stat = st.selectbox("Stat", ["PTS","REB","AST","STL","BLK","TOV","MIN","FG_PCT","FG3_PCT"], key="gl_stat")
        window = st.slider("Rolling window (games)", 3, 20, 10, key="gl_window")
        if st.button("Fetch game log", key="gl_go"):
            pid = find_player_id_by_name(gq)
            if not pid:
                st.error("Player not found.")
            else:
                try:
                    with st.spinner(f"Fetching {gq} — {g_season}…"):
                        glog = fetch_player_gamelog(pid, g_season)
                    if glog.empty or gl_stat not in glog.columns:
                        st.info("No games or stat unavailable.")
                    else:
                        glog = glog.copy()
                        glog[gl_stat] = pd.to_numeric(glog[gl_stat], errors="coerce")
                        glog["rolling"] = glog[gl_stat].rolling(window, min_periods=1).mean()
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=glog["GAME_DATE"], y=glog[gl_stat],
                            name=f"Per-game {gl_stat}", opacity=0.55,
                        ))
                        fig.add_trace(go.Scatter(
                            x=glog["GAME_DATE"], y=glog["rolling"],
                            name=f"{window}-game avg", mode="lines", line=dict(width=3),
                        ))
                        fig.update_layout(
                            title=f"{gq} — {g_season} — {gl_stat}",
                            xaxis_title="Game date", yaxis_title=gl_stat,
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"{len(glog)} games • season avg {glog[gl_stat].mean():.2f}")
                except Exception as e:
                    st.error(f"Failed: {e}")

with tab_age:
    st.subheader("Development curves by age")
    if commonplayerinfo is None or players_static is None:
        st.info("nba_api unavailable — cannot look up birthdates.")
    elif not AVAILABLE_METRICS:
        st.info("No metrics available.")
    else:
        names_age = st.multiselect("Players", player_names, max_selections=5, key="age_players")
        default_age = "ts_pct" if "ts_pct" in AVAILABLE_METRICS else AVAILABLE_METRICS[0]
        stat_age = st.selectbox("Metric", AVAILABLE_METRICS,
                                index=AVAILABLE_METRICS.index(default_age), key="age_stat")
        if not names_age:
            st.info("Pick 1–5 players to overlay.")
        else:
            with st.spinner("Looking up birthdates…"):
                fig = go.Figure()
                any_points = False
                for pname in names_age:
                    sub = players[players["player_name"] == pname].copy()
                    if sub.empty: continue
                    pid = int(sub["player_id"].iloc[0])
                    bd = fetch_player_birthdate(pid)
                    if bd is None or pd.isna(bd):
                        st.caption(f"(no birthdate for {pname})")
                        continue
                    sub["age"] = sub["season"].map(lambda s: age_for_season(bd, s))
                    sub = sub.dropna(subset=["age", stat_age]).sort_values("age")
                    if sub.empty: continue
                    any_points = True
                    fig.add_trace(go.Scatter(
                        x=sub["age"], y=sub[stat_age],
                        mode="lines+markers", name=pname,
                    ))
                if any_points:
                    fig.update_layout(
                        xaxis_title="Age at season start", yaxis_title=stat_age,
                        title=f"{stat_age} vs. age", hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No age-aligned data available.")

# Teams
with tab_teams:
    st.subheader("Team stats dashboard")
    if not team_names:
        st.info("No teams in teams.parquet yet.")
    else:
        tname = st.selectbox("Team", team_names)
        tdf = teams[teams["team_name"] == tname].copy()
        numcols = ["wins","losses","FG_PCT","FT_PCT","FG3_PCT","ortg","drtg","pace",
                   "PTS","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","REB","TOV","AST"]
        for c in numcols:
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
                shoot = [c for c in ["FG_PCT","FG3_PCT","FT_PCT"] if c in tdf.columns]
                if shoot:
                    st.plotly_chart(px.line(tdf, x="season", y=shoot, markers=True,
                                            title=f"{tname} — Shooting %"), use_container_width=True)
                else:
                    st.info("Ratings/Pace not available.")

        st.markdown("### Four Factors (Dean Oliver)")
        needed = {"FGM","FGA","FG3M","FTA","TOV","OREB","DREB"}
        if not needed.issubset(tdf.columns):
            st.info(f"Missing columns for Four Factors: {sorted(needed - set(tdf.columns))}")
        else:
            season_opts = tdf["season"].tolist()
            ff_season = st.selectbox("Season", season_opts,
                                     index=len(season_opts)-1, key="ff_season")
            team_row = tdf[tdf["season"] == ff_season].iloc[0]

            def four_factors(row):
                fga = row.get("FGA",0) or 0
                fta = row.get("FTA",0) or 0
                fgm = row.get("FGM",0) or 0
                fg3m = row.get("FG3M",0) or 0
                tov = row.get("TOV",0) or 0
                oreb = row.get("OREB",0) or 0
                dreb = row.get("DREB",0) or 0
                efg  = ((fgm + 0.5 * fg3m) / fga) if fga else np.nan
                poss_proxy = fga + 0.44 * fta + tov
                tovp = (tov / poss_proxy) if poss_proxy else np.nan
                orb  = (oreb / (oreb + dreb)) if (oreb + dreb) else np.nan
                ftr  = (fta / fga) if fga else np.nan
                return {"eFG%": efg, "TOV%": tovp, "ORB%": orb, "FT rate": ftr}

            team_ff = four_factors(team_row)
            lg_season = teams[teams["season"] == ff_season].copy()
            for c in numcols:
                if c in lg_season.columns:
                    lg_season[c] = pd.to_numeric(lg_season[c], errors="coerce")
            lg_ff_rows = lg_season.apply(four_factors, axis=1, result_type="expand")
            lg_avg = lg_ff_rows.mean(numeric_only=True).to_dict()

            ff_df = pd.DataFrame([
                {"factor": k, "team": team_ff[k], "league avg": lg_avg.get(k, np.nan)}
                for k in ["eFG%","TOV%","ORB%","FT rate"]
            ])
            fig_ff = go.Figure()
            fig_ff.add_trace(go.Bar(x=ff_df["factor"], y=ff_df["team"],
                                    name=tname, marker_color="#1f77b4",
                                    text=[f"{v:.3f}" if pd.notna(v) else "" for v in ff_df["team"]],
                                    textposition="outside"))
            fig_ff.add_trace(go.Bar(x=ff_df["factor"], y=ff_df["league avg"],
                                    name="League avg", marker_color="#aaaaaa",
                                    text=[f"{v:.3f}" if pd.notna(v) else "" for v in ff_df["league avg"]],
                                    textposition="outside"))
            fig_ff.update_layout(barmode="group",
                                 title=f"Four Factors — {tname} {ff_season}",
                                 yaxis_title="Value")
            st.plotly_chart(fig_ff, use_container_width=True)
            st.caption("ORB% uses OREB / (OREB+DREB) as a simple proxy — the full formula requires opponent DREB.")

with tab_shots:
    st.subheader("Shot chart (live from NBA API)")

    q = st.text_input("Search player (full or partial name)", value="Stephen Curry")
    season_options = sorted(players["season"].dropna().unique().tolist())
    s_sel = st.selectbox("Season", season_options,
                         index=len(season_options)-1 if season_options else 0)
    mode = st.radio("View", ["Scatter", "Hexbin (FG%)"], horizontal=True, key="shot_mode")

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

                        fig = go.Figure()

                        if mode == "Scatter" or sdf.empty:
                            miss = sdf[sdf["made"] == 0]
                            make = sdf[sdf["made"] == 1]
                            fig.add_trace(go.Scatter(
                                x=miss["x"], y=miss["y"], mode="markers",
                                name="Miss", marker=dict(symbol="x", size=6, opacity=0.55)
                            ))
                            fig.add_trace(go.Scatter(
                                x=make["x"], y=make["y"], mode="markers",
                                name="Make", marker=dict(size=6, opacity=0.85)
                            ))
                        else:
                            x = sdf["x"].to_numpy()
                            y = sdf["y"].to_numpy()
                            m = sdf["made"].to_numpy()
                            xb = np.linspace(-250, 250, 26)
                            yb = np.linspace(-52.5, 417.5, 24)
                            cnt = binned_statistic_2d(x, y, np.ones_like(x), statistic="count", bins=[xb, yb])
                            pct = binned_statistic_2d(x, y, m, statistic="mean", bins=[xb, yb])
                            cx = 0.5 * (xb[:-1] + xb[1:])
                            cy = 0.5 * (yb[:-1] + yb[1:])
                            XC, YC = np.meshgrid(cx, cy, indexing="ij")
                            count = cnt.statistic
                            fgp = pct.statistic
                            mask = count >= 2
                            xs = XC[mask]; ys = YC[mask]
                            cs = count[mask]; fs = fgp[mask]
                            max_c = max(cs.max(), 1.0)
                            size = 8 + 22 * np.sqrt(cs / max_c)
                            player_fg = float(np.nanmean(m)) if len(m) else 0.45
                            fig.add_trace(go.Scatter(
                                x=xs, y=ys, mode="markers",
                                marker=dict(
                                    symbol="hexagon",
                                    size=size,
                                    color=fs,
                                    colorscale=[[0,"#2166ac"],[0.5,"#f7f7f7"],[1,"#b2182b"]],
                                    cmin=max(0.0, player_fg - 0.25),
                                    cmax=min(1.0, player_fg + 0.25),
                                    colorbar=dict(title="FG%", tickformat=".0%"),
                                    line=dict(width=0.5, color="#444"),
                                ),
                                text=[f"FG% {fv:.0%} • {int(c)} shots" for fv, c in zip(fs, cs)],
                                hovertemplate="%{text}<extra></extra>",
                                name="FG% by zone",
                                showlegend=False,
                            ))
                            st.caption(
                                f"Player FG%: {player_fg:.1%} — color diverges around that value "
                                f"(red = above, blue = below). Hex size = shot volume."
                            )

                        draw_plotly_court(fig, fig_width=900, margins=10)
                        fig.update_layout(title=f"{resolved_name} — {s_sel} Shot Chart")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to fetch or plot shots: {e}")
