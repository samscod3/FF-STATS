# FF-STATS — NFL Stats Demo

Python scripts that generate interactive Plotly charts from `nfl_data_py` and public ADP feeds.  
**Nothing opens automatically** — charts are written as HTML files in the repo folder.

## Setup

```bash
# (recommended) create a virtual env
python3 -m venv .venv
source .venv/bin/activate

# install packages
pip install -r requirements.txt  # or: pip install nfl_data_py pandas numpy plotly requests rapidfuzz

#RUN THE CODE
python app.py

#MAC OS
open charts/adp_vs_points.html         # Top 25 WRs — ADP vs 2024 Fantasy Points
open charts/top_wr_2024.html           # Top 10 WRs by Receiving Yards (2024)
open charts/leaders_2024.html          # Top 15 Overall PPR (2024)
open charts/wr_yards_vs_targets.html   # Top 25 WRs — Receiving Yards vs Targets (2024)
open charts/rookie_wr_usage.html       # Rookie WR Usage — Draft Order vs Team 2024 Receiving Yards
open charts/rb_top25_vs_oline.html     # Top 25 RBs — 2024 Rush Yds vs 2024 OL Rank
open charts/qb_adp_vs_points.html      # Top 15 QBs — 2024 Fantasy Points vs 2025 ADP

#WINDOWS POWERSHELL