# app.py
import sys, os
print("Python:", sys.executable)
print("CWD:", os.getcwd())

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import plotly.express as px

seasons = [2024]

# -------- Load weekly data --------
weekly = nfl.import_weekly_data(seasons).copy()

# -------- Robust PPR fantasy scoring --------
SCORING = {
    "pass_yd": 1/25,
    "pass_td": 4,
    "pass_int": -2,
    "rush_yd": 1/10,
    "rush_td": 6,
    "rec": 1,          # set to 0 for standard
    "rec_yd": 1/10,
    "rec_td": 6,
    "fumbles_lost": -2,
}

def col(df, *names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").fillna(0)
    return 0

weekly["fantasy_pts"] = (
    col(weekly, "passing_yards")      * SCORING["pass_yd"] +
    col(weekly, "passing_tds")        * SCORING["pass_td"] +
    col(weekly, "interceptions")      * SCORING["pass_int"] +
    col(weekly, "rushing_yards")      * SCORING["rush_yd"] +
    col(weekly, "rushing_tds")        * SCORING["rush_td"] +
    col(weekly, "receptions", "rec")  * SCORING["rec"] +
    col(weekly, "receiving_yards")    * SCORING["rec_yd"] +
    col(weekly, "receiving_tds")      * SCORING["rec_td"] +
    col(weekly, "fumbles_lost", "fumbles", "fumbles_lost_total") * SCORING["fumbles_lost"]
)

# -------- Derive position via per-player season MODE from weekly --------
def mode_or_nan(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]

# Aggregate season totals (and pick most common position)
season_totals = (
    weekly.groupby(["season", "player_name"], as_index=False)
    .agg(
        receiving_yards=("receiving_yards", "sum"),
        fantasy_pts=("fantasy_pts", "sum"),
        position=("position", mode_or_nan)
    )
)

# # ======== LIVE ADP vs 2024 Fantasy Points (Top 25 WRs by ADP) ========
# import os, json, requests, certifi
# from rapidfuzz import process, fuzz

# # Pull WR-only to reduce noise (you can switch back to position=all if you want)
# ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?position=wr&teams=12&year=2025"

# def write_placeholder(msg: str, fname="adp_vs_points.html"):
#     with open(fname, "w") as f:
#         f.write(f"<h3>{msg}</h3><p>See terminal logs and CSVs for details.</p>")
#     print(f"Wrote {fname} (placeholder)")

# # ===== FETCH CURRENT ADP (robust) =====
# import json, requests, os

# UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
#                     "AppleWebKit/537.36 (KHTML, like Gecko) "
#                     "Chrome/124.0 Safari/537.36"}

# def fetch_ffc_adp(position_param: str) -> dict | None:
#     url = f"https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year=2025&position={position_param}"
#     try:
#         r = requests.get(url, headers=UA, timeout=20)
#         print(f"GET {url} ->", r.status_code)
#         if not r.ok:
#             return None
#         data = r.json()
#         # persist for debugging
#         with open(f"raw_adp_{position_param}.json", "w") as f:
#             json.dump(data, f)  # so you can open it if needed
#         return data
#     except Exception as e:
#         print("ADP fetch error:", repr(e))
#         return None

# # 1) Try WR only (uppercase); 2) fallback to ALL positions
# adp_json = fetch_ffc_adp("WR")
# if not adp_json or not isinstance(adp_json.get("players", []), list) or len(adp_json["players"]) == 0:
#     print("WR-only ADP empty; falling back to ALL positions and filtering locally…")
#     adp_json = fetch_ffc_adp("ALL")

# # ===== BUILD ADP DATAFRAME =====
# adp_df = pd.DataFrame()
# if adp_json and isinstance(adp_json.get("players", []), list):
#     adp_df = pd.DataFrame(adp_json["players"])
# print("ADP df shape (pre-clean):", adp_df.shape)

# if adp_df.empty:
#     # keep your placeholder writer
#     write_placeholder("No ADP rows returned (even after ALL fallback). "
#                       "Check raw_adp_WR.json / raw_adp_ALL.json.")
# else:
#     # if we fetched ALL, filter WR here
#     if "position" in adp_df.columns:
#         # some feeds use uppercased "WR", some "Wr" — normalize then filter
#         adp_df["position"] = adp_df["position"].astype(str).str.upper()
#         adp_df = adp_df[adp_df["position"] == "WR"].copy()

#     # standardize columns
#     adp_df = adp_df.rename(columns={
#         "name": "player_name",
#         "position": "pos_adp",
#         "avg": "adp",  # some historical examples use 'avg'
#         "average_draft_position": "adp"
#     })
#     # expected set (keep what exists)
#     keep_cols = [c for c in ["player_name","pos_adp","team","adp","adp_formatted","times_drafted"] if c in adp_df.columns]
#     adp_df = adp_df[keep_cols].copy()
#     if "adp" in adp_df.columns:
#         adp_df["adp"] = pd.to_numeric(adp_df["adp"], errors="coerce")

#     print("ADP df shape (WR, post-clean):", adp_df.shape)
#     print("ADP sample:\n", adp_df.head(10))


#     # ---------- Strong normalization helpers ----------
#     SUFFIXES = {"jr","sr","ii","iii","iv","v"}
#     def normalize_name(s: pd.Series) -> pd.Series:
#         s = s.fillna("").str.lower()
#         s = s.str.replace(r"[^a-z0-9\s]", "", regex=True)         # strip punctuation
#         s = s.str.replace(r"\s+", " ", regex=True).str.strip()    # collapse spaces
#         # remove suffix at end
#         def strip_suffix(x):
#             parts = x.split()
#             if parts and parts[-1] in SUFFIXES:
#                 parts = parts[:-1]
#             return " ".join(parts)
#         s = s.apply(strip_suffix)
#         # fix common token
#         s = s.str.replace(r"\bst\b", "st", regex=True)            # "st." -> "st"
#         return s

#     # Common alias fixes (expand as needed)
#     alias_map = {
#         "dk metcalf": "dk metcalf",
#         "dj moore": "dj moore",
#         "aj brown": "aj brown",
#         "tj hockenson": "tj hockenson",
#         "ceedee lamb": "ceedee lamb",
#         "ken walker": "ken walker",
#         "kenneth walker": "ken walker",
#         "amon ra st brown": "amon ra st brown",
#         "devon achane": "devon achane",
#         "pat mahomes": "patrick mahomes",
#         "gabe davis": "gabriel davis",
#         "tank dell": "nathaniel dell",
#     }
#     def apply_alias(s: pd.Series) -> pd.Series:
#         return s.replace(alias_map)

#     # Build normalized keys
#     adp_df["_name_key"] = apply_alias(normalize_name(adp_df.get("player_name", pd.Series(dtype=str))))
#     season_totals["_name_key"] = apply_alias(normalize_name(season_totals["player_name"]))

#     # Normalize positions (we want WRs)
#     def norm_pos(x):
#         if pd.isna(x): return x
#         x = str(x).upper()
#         if x.startswith("RB"): return "RB"
#         if x.startswith("WR"): return "WR"
#         if x.startswith("TE"): return "TE"
#         if x.startswith("QB"): return "QB"
#         return x
#     adp_df["pos_adp"] = adp_df.get("pos_adp", "").apply(norm_pos)
#     season_totals["position"] = season_totals["position"].apply(norm_pos)

#     # ---------- Build Top-25 WRs by ADP from the ADP feed ----------
#     if "adp" not in adp_df.columns:
#         write_placeholder("ADP feed missing 'adp' field.")
#     else:
#         adp_wr = adp_df.copy()
#         if "pos_adp" in adp_wr.columns:
#             adp_wr = adp_wr[adp_wr["pos_adp"] == "WR"]
#         adp_wr = adp_wr.dropna(subset=["adp"])
#         adp_wr["adp"] = pd.to_numeric(adp_wr["adp"], errors="coerce")
#         adp_wr = adp_wr.dropna(subset=["adp"])
#         adp_wr_top25 = adp_wr.sort_values("adp", ascending=True).head(25).copy()

#         if adp_wr_top25.empty:
#             write_placeholder("ADP feed returned no WRs for Top-25.")
#         else:
#             # Normalize names for fuzzy match into your 2024 totals
#             def norm(s):
#                 return (s.fillna("")
#                          .str.lower()
#                          .str.replace(r"[^a-z0-9\s]", "", regex=True)
#                          .str.replace(r"\s+", " ", regex=True)
#                          .str.strip())

#             st_24 = season_totals[season_totals["season"] == 2024].copy()
#             st_24["_name_key"] = norm(st_24["player_name"])
#             st_keys = st_24["_name_key"].tolist()

#             adp_wr_top25["_name_key"] = norm(adp_wr_top25["player_name"])

#             matched_rows = []
#             for _, r in adp_wr_top25.iterrows():
#                 best = process.extractOne(r["_name_key"], st_keys, scorer=fuzz.WRatio, score_cutoff=82)
#                 if best:
#                     best_key, score, _ = best
#                     st_row = st_24.loc[st_24["_name_key"] == best_key].iloc[0]
#                     matched_rows.append({
#                         "player_name": r["player_name"],
#                         "team": r.get("team", None),
#                         "adp": r["adp"],
#                         "fantasy_pts": st_row.get("fantasy_pts", None),
#                         "position": st_row.get("position", "WR"),
#                     })
#                 else:
#                     # keep ADP row in chart even if no stats match
#                     matched_rows.append({
#                         "player_name": r["player_name"],
#                         "team": r.get("team", None),
#                         "adp": r["adp"],
#                         "fantasy_pts": None,
#                         "position": "WR",
#                     })

#             plot_df = pd.DataFrame(matched_rows)
#             plot_df.to_csv("top25_wr_adp_points.csv", index=False)
#             print("Wrote top25_wr_adp_points.csv")
#             print("Top-25 WRs matched:", plot_df.shape[0],
#                   "with fantasy_pts present:", plot_df["fantasy_pts"].notna().sum())

#             # -------- Plot Top 25 WRs by ADP vs 2024 Fantasy Points --------
#             if plot_df.empty:
#                 write_placeholder("No rows to plot after matching.")
#             else:
#                 fig_adp = px.scatter(
#                     plot_df,
#                     x="adp",
#                     y="fantasy_pts",
#                     hover_data=["player_name","team"],
#                     title="Top 25 WRs by Draft Position — ADP vs 2024 Fantasy Points",
#                     labels={"adp":"ADP (lower = earlier pick)","fantasy_pts":"2024 Fantasy Points"}
#                 )
#                 out = os.path.join(os.getcwd(), "adp_vs_points.html")
#                 fig_adp.write_html(out, auto_open=True)
#                 print("Wrote:", out)
# # ===============================================================






# # --- Peek in terminal ---
# print("\n=== season_totals sample ===")
# print(season_totals.sort_values(["season","fantasy_pts"], ascending=[True,False]).head(15))

# # --- Chart: Top 10 WRs by receiving yards (2024) ---
# top_wr_2024 = (
#     season_totals[(season_totals["season"] == 2024) & (season_totals["position"] == "WR")]
#     .nlargest(10, "receiving_yards")
# )
# fig1 = px.bar(top_wr_2024, x="player_name", y="receiving_yards",
#               title="Top 10 WRs by Receiving Yards (2024)")
# fig1.write_html("top_wr_2024.html", auto_open=True)

# # --- Chart: Top 15 by PPR Fantasy Points (2024) ---
# leaders_2024 = season_totals[season_totals["season"] == 2024].nlargest(15, "fantasy_pts")
# fig2 = px.bar(leaders_2024, x="player_name", y="fantasy_pts", color="position",
#               title="Top 15 Fantasy (PPR) — 2024")
# fig2.write_html("leaders_2024.html", auto_open=True)








##

# print("weekly has 'targets' column? ->", "targets" in weekly.columns)
# print("Number of WRs in season_totals 2024:",
#       season_totals[(season_totals["season"] == 2024) & (season_totals["position"] == "WR")].shape[0])

# # Get top 25 WRs by yards
# top_wr_2024 = (
#     season_totals[(season_totals["season"] == 2024) & (season_totals["position"] == "WR")]
#     .nlargest(25, "receiving_yards")
# )

# if "targets" in weekly.columns:
#     targets_totals = (
#         weekly.groupby(["season", "player_name"], as_index=False)
#         .agg(targets=("targets", "sum"))
#     )
#     top_wr_2024 = top_wr_2024.merge(targets_totals, on=["season", "player_name"], how="left")
# else:
#     print("No 'targets' column — using receptions as proxy.")
#     rec_totals = (
#         weekly.groupby(["season", "player_name"], as_index=False)
#         .agg(targets=("receptions", "sum"))
#     )
#     top_wr_2024 = top_wr_2024.merge(rec_totals, on=["season", "player_name"], how="left")

# if not top_wr_2024.empty:
#     fig3 = px.scatter(
#         top_wr_2024,
#         x="targets",
#         y="receiving_yards",
#         text="player_name",
#         size="receiving_yards",
#         color="fantasy_pts",
#         title="Top 25 WRs: Receiving Yards vs Targets (2024)",
#         labels={"targets": "Total Targets (or receptions proxy)", "receiving_yards": "Total Receiving Yards"}
#     )
#     fig3.update_traces(textposition="top center")
#     fig3.write_html("wr_yards_vs_targets.html", auto_open=True)
#     print("Scatter plot written to wr_yards_vs_targets.html")
# else:
#     print("No WR data available for 2024 scatter plot.")    






# ======== Rookie WR Usage Potential (Top-25 by Draft Order) — auto-detect & always write ========
# import os, re
# import numpy as np

# OUT_HTML = "rookie_wr_usage.html"

# def write_placeholder(msg: str, fname=OUT_HTML):
#     with open(fname, "w") as f:
#         f.write(f"<h3>{msg}</h3><p>See terminal logs/CSVs for details.</p>")
#     print(f"Wrote {fname} (placeholder)")

# def pick_col_by_patterns(cols, patterns, exclude=None):
#     """Return first column whose lowercase name contains any of the given substrings (ordered)."""
#     low = {c.lower(): c for c in cols}
#     exclude = set(exclude or [])
#     for pat in patterns:
#         for k, orig in low.items():
#             if any(x in k for x in exclude):
#                 continue
#             if pat in k:
#                 return orig
#     return None

# # 0) Load draft data
# try:
#     draft = nfl.import_draft_picks([2025]).copy()
# except Exception as e:
#     print("Draft fetch failed:", repr(e))
#     draft = pd.DataFrame()

# print("Draft shape:", draft.shape)
# if not draft.empty:
#     print("Draft columns:", draft.columns.tolist())
#     print(draft.head(10).to_string(index=False))

# if draft.empty:
#     write_placeholder("No draft data for 2025 from nfl_data_py.")
# else:
#     cols = list(draft.columns)

#     # Heuristic column detection (substring search)
#     # Overall pick: prefer 'overall', then 'pick_overall', then plain 'pick' (but avoid 'round')
#     pick_col  = pick_col_by_patterns(cols, ["overall", "ovr", "pick_overall", "overall_pick", "selection"], exclude=["round"])
#     if pick_col is None:
#         # fallback: plain 'pick' but not containing 'round'
#         pick_col = pick_col_by_patterns(cols, ["pick"], exclude=["round"])

#     # Round
#     round_col = pick_col_by_patterns(cols, ["round", "rnd"])

#     # Position
#     pos_col   = pick_col_by_patterns(cols, ["position", "pos"])

#     # Player name (avoid team names)
#     name_col  = pick_col_by_patterns(cols, ["player_name", "player", "name"])

#     # Team (abbr or name)
#     team_col  = pick_col_by_patterns(cols, ["team_abbr", "team_code", "posteam", "club_code", "tm", "team"])

#     print("Detected columns ->",
#           {"overall_pick": pick_col, "round": round_col, "pos": pos_col, "player": name_col, "team": team_col})

#     if any(x is None for x in [pick_col, round_col, pos_col, name_col, team_col]):
#         write_placeholder("Draft columns not found by auto-detect. See terminal for detected set & sample.")
#     else:
#         # Types
#         draft[pick_col]  = pd.to_numeric(draft[pick_col], errors="coerce")
#         draft[round_col] = pd.to_numeric(draft[round_col], errors="coerce")

#         # Team abbr mapping if team looks like full name
#         TEAM_MAP = {
#             "arizona cardinals":"ARI","atlanta falcons":"ATL","baltimore ravens":"BAL","buffalo bills":"BUF",
#             "carolina panthers":"CAR","chicago bears":"CHI","cincinnati bengals":"CIN","cleveland browns":"CLE",
#             "dallas cowboys":"DAL","denver broncos":"DEN","detroit lions":"DET","green bay packers":"GB",
#             "houston texans":"HOU","indianapolis colts":"IND","jacksonville jaguars":"JAX","kansas city chiefs":"KC",
#             "las vegas raiders":"LV","los angeles chargers":"LAC","los angeles rams":"LA","miami dolphins":"MIA",
#             "minnesota vikings":"MIN","new england patriots":"NE","new orleans saints":"NO","new york giants":"NYG",
#             "new york jets":"NYJ","philadelphia eagles":"PHI","pittsburgh steelers":"PIT","san francisco 49ers":"SF",
#             "seattle seahawks":"SEA","tampa bay buccaneers":"TB","tennessee titans":"TEN","washington commanders":"WAS"
#         }
#         def to_abbr(x):
#             if pd.isna(x): return x
#             s = str(x).strip()
#             # already an abbr?
#             if s.isupper() and 2 <= len(s) <= 3:
#                 return s
#             return TEAM_MAP.get(s.lower(), s)

#         draft["team_abbr"] = draft[team_col].apply(to_abbr)
#         draft["pos_norm"]  = draft[pos_col].astype(str).str.upper()

#         # WRs by earliest overall pick
#         wr_draft = draft[draft["pos_norm"] == "WR"].sort_values(pick_col, ascending=True).copy()

#         if wr_draft.empty:
#             print("No WR rows after filtering. First few draft rows:\n", draft.head(10).to_string(index=False))
#             write_placeholder("No WR rows in draft data (check position column).")
#         else:
#             wr_top25 = wr_draft.head(25)[[name_col, "team_abbr", round_col, pick_col]].rename(columns={
#                 name_col: "player_name",
#                 round_col:"draft_round",
#                 pick_col: "overall_pick"
#             })

#             # Detect team column in weekly
#             wk_team_col = None
#             for cand in ["team","recent_team","posteam","team_abbr","club","club_code","Tm"]:
#                 if cand in weekly.columns:
#                     wk_team_col = cand
#                     break
#             print("Detected weekly team column:", wk_team_col)
#             if wk_team_col is None:
#                 write_placeholder("Could not find a team column in weekly data. Check weekly.columns().")
#             else:
#                 # Team receiving yards (2024)
#                 rec_by_team = (
#                     weekly[weekly["season"] == 2024]
#                     .groupby(wk_team_col, as_index=False)
#                     .agg(team_rec_yards=("receiving_yards","sum"))
#                 ).rename(columns={wk_team_col:"team_abbr"})
#                 rec_by_team["team_rec_rank"] = rec_by_team["team_rec_yards"].rank(method="min", ascending=False).astype(int)

#                 merged = wr_top25.merge(rec_by_team, on="team_abbr", how="left")
#                 merged.to_csv("rookie_wr_top25_usage.csv", index=False)
#                 print("Wrote rookie_wr_top25_usage.csv")
#                 print(merged[["player_name","team_abbr","draft_round","overall_pick","team_rec_yards","team_rec_rank"]]
#                       .sort_values("overall_pick").to_string(index=False))

#                 if merged.empty or merged["overall_pick"].isna().all():
#                     write_placeholder("Merged dataset empty (check team abbreviations & weekly team column).")
#                 else:
#                     # Plot (will still render even if some team_rec_yards are NaN)
#                     fig_r = px.scatter(
#                         merged.sort_values("overall_pick"),
#                         x="overall_pick",
#                         y="team_rec_yards",
#                         text="player_name",
#                         color="draft_round",
#                         hover_data=["player_name","team_abbr","draft_round","overall_pick","team_rec_rank"],
#                         title="Rookie WR Usage Potential — Top 25 by Draft Order vs Team 2024 Receiving Yards",
#                         labels={
#                             "overall_pick": "Overall Pick (lower = earlier)",
#                             "team_rec_yards": "Team Receiving Yards (2024)",
#                             "draft_round": "Draft Round"
#                         },
#                     )
#                     fig_r.update_traces(textposition="top center")
#                     fig_r.write_html(OUT_HTML, auto_open=True)
#                     print(f"Wrote {OUT_HTML}")
# =============================================================================================






# ======== Top 25 RBs by 2024 Rushing Yards vs 2024 OL Rank — guaranteed chart ========
import os, re, requests
import pandas as pd
import numpy as np
import plotly.express as px

OUT_HTML = "rb_top25_vs_oline.html"
def OUT_PATH():
    return os.path.join(os.getcwd(), OUT_HTML)

print("\n[RB/OL] CWD:", os.getcwd())
print("[RB/OL] weekly shape:", getattr(weekly, "shape", None))
print("[RB/OL] weekly columns:", list(getattr(weekly, "columns", []))[:50])

try:
    print("[RB/OL] weekly head (5):")
    print(weekly.head(5).to_string())
except Exception:
    pass

# --- Detect columns present in your weekly data ---
wk_team_col = next((c for c in ["team","recent_team","posteam","team_abbr","club","club_code","Tm"]
                    if c in weekly.columns), None)
rush_y_col  = "rushing_yards" if "rushing_yards" in weekly.columns else None
rush_a_col  = "rushing_attempts" if "rushing_attempts" in weekly.columns else None
pos_col     = "position" if "position" in weekly.columns else None

if wk_team_col is None or rush_y_col is None:
    with open(OUT_PATH(), "w") as f:
        f.write("<h3>Missing required weekly columns.</h3>"
                "<p>Need a team col (one of team/recent_team/posteam/...) and 'rushing_yards'.</p>")
    print("[RB/OL] Wrote placeholder (missing columns):", OUT_PATH())
else:
    # --- Per-player 2024 totals (correct named-aggregation syntax) ---
    grp = weekly[weekly["season"] == 2024].groupby("player_name", as_index=False)
    named_aggs = {
        "rush_yds": (rush_y_col, "sum"),
        "team":     (wk_team_col, lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
    }
    if rush_a_col:
        named_aggs["rush_att"] = (rush_a_col, "sum")
    if pos_col:
        named_aggs["position"] = (pos_col, lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan)

    tmp = grp.agg(**named_aggs)

    # RB detection
    if "position" in tmp.columns:
        is_rb = tmp["position"].astype(str).str.upper().eq("RB")
    else:
        is_rb = tmp.get("rush_att", pd.Series(index=tmp.index, dtype=float)).fillna(0) >= 50

    rb = tmp[is_rb].copy()
    print("[RB/OL] candidate RB rows:", rb.shape[0])

    # Top 25 by rushing yards
    rb_top25 = rb.sort_values("rush_yds", ascending=False).head(25).copy()
    print("[RB/OL] Top-25 RB rows:", rb_top25.shape[0])
    if "rush_att" not in rb_top25.columns:
        rb_top25["rush_att"] = np.nan

    # --- Team normalization (full names, legacy tags, 3-letter codes) ---
    TEAM_TO_ABBR = {
        # Pass-through 3-letter + common legacy
        "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
        "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","GNB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","JAC":"JAX",
        "KC":"KC","KCC":"KC","LV":"LV","LVR":"LV","LAC":"LAC","SD":"LAC","SDG":"LAC","LAR":"LAR","LA":"LAR","STL":"LAR",
        "MIA":"MIA","MIN":"MIN","NE":"NE","NWE":"NE","NO":"NO","NOR":"NO","NYG":"NYG","NYJ":"NYJ","PHI":"PHI",
        "PIT":"PIT","SF":"SF","SFO":"SF","SEA":"SEA","TB":"TB","TAM":"TB","TEN":"TEN","WAS":"WAS","WSH":"WAS",
        # Full names
        "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
        "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
        "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
        "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
        "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","LA Chargers":"LAC","San Diego Chargers":"LAC",
        "Los Angeles Rams":"LAR","LA Rams":"LAR","St. Louis Rams":"LAR","Miami Dolphins":"MIA","Minnesota Vikings":"MIN",
        "New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG","New York Jets":"NYJ",
        "Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF","SF 49ers":"SF",
        "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
    }
    def to_abbr(x: str) -> str:
        s = str(x).strip()
        return TEAM_TO_ABBR.get(s, TEAM_TO_ABBR.get(s.upper(), s))

    rb_top25["team_norm"] = rb_top25["team"].astype(str).map(to_abbr)

    # --- 2024 OL rankings (editable block) ---
    # NOTE: Fill/adjust as you like; any team not listed will default to rank 33 (plotted at far right).
    # This list is intentionally exhaustive to avoid 33s. If your source differs, tweak numbers here.
    OLINE_2024_LIST = [
        # Example ordering — update to your preferred 2024 list/source
        ("PHI", 1), ("BAL", 2), ("DET", 3), ("DAL", 4), ("TB", 5),
        ("DEN", 6), ("KC", 7), ("GB", 8), ("CLE", 9), ("SF", 10),
        ("HOU", 11), ("LAR", 12), ("IND", 13), ("MIA", 14), ("MIN", 15),
        ("ATL", 16), ("JAX", 17), ("BUF", 18), ("PIT", 19), ("SEA", 20),
        ("CIN", 21), ("NO", 22), ("NYJ", 23), ("CHI", 24), ("LV", 25),
        ("WAS", 26), ("LA C".replace(" ",""), 27), # in case your data has LAC separate from normalization
        ("LAC", 27), ("NYG", 28), ("TEN", 29), ("CAR", 30), ("NE", 31), ("ARI", 32),
        # If you disagree with any slot, just edit the number.
    ]
    oline_df = pd.DataFrame(OLINE_2024_LIST, columns=["team_abbr","oline_rank_2024"]).drop_duplicates("team_abbr")

    # --- Merge using normalized team codes ---
    rb_top25 = rb_top25.merge(
        oline_df[["team_abbr","oline_rank_2024"]],
        left_on="team_norm", right_on="team_abbr", how="left"
    )

    # Missing -> 33 so it still plots
    rb_top25["oline_rank_2024"] = pd.to_numeric(
        rb_top25.get("oline_rank_2024", np.nan), errors="coerce"
    ).fillna(33)

    print("[RB/OL] sample team merge check (2024 ranks):")
    try:
        print(rb_top25[["player_name","team","team_norm","oline_rank_2024"]].head(12).to_string(index=False))
    except Exception:
        pass

    # Bail gracefully if empty (shouldn't happen)
    if rb_top25.empty:
        with open(OUT_PATH(), "w") as f:
            f.write("<h3>No RB rows found for 2024.</h3>"
                    "<p>Check your 'weekly' DataFrame contents and filters.</p>")
        print("[RB/OL] Wrote placeholder (no RB rows):", OUT_PATH())
    else:
        # --- Plot ---
        fig = px.scatter(
            rb_top25,
            x="oline_rank_2024",
            y="rush_yds",
            text="player_name",
            color="team",
            hover_data=["player_name","team","rush_att","oline_rank_2024"],
            title="Top 25 RBs by 2024 Rushing Yards vs 2024 Offensive Line Rank"
        )
        fig.update_traces(textposition="top center")
        out = OUT_PATH()
        fig.write_html(out, auto_open=True)
        print("[RB/OL] Chart saved to:", out)

