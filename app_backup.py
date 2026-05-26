"""
╔══════════════════════════════════════════════════════════════╗
║        FOOTBALL INTELLIGENCE SYSTEM — app.py                ║
║  All modules in one file:                                    ║
║   1. Data Pipeline      (StatsBomb ingestion)                ║
║   2. Feature Engineering                                     ║
║   3. ML Models          (Action, xG, Clustering, Scouting)   ║
║   4. FastAPI Backend                                         ║
║   5. React Frontend      (served via FastAPI)                ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
# STANDARD IMPORTS
# ─────────────────────────────────────────────
import os, json, warnings, math, logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("football_intel")

# ─────────────────────────────────────────────
# DATA / ML IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ─────────────────────────────────────────────
# FASTAPI IMPORTS
# ─────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# STATSBOMB IMPORT (optional, graceful fallback)
# ─────────────────────────────────────────────
try:
    from statsbombpy import sb
    STATSBOMB_AVAILABLE = True
    logger.info("StatsBomb library loaded.")
except ImportError:
    STATSBOMB_AVAILABLE = False
    logger.warning("statsbombpy not installed. Using synthetic data.")


# ══════════════════════════════════════════════════════════════
# MODULE 1 — DATA PIPELINE
# ══════════════════════════════════════════════════════════════

class DataPipeline:
    """Handles StatsBomb data ingestion and preprocessing."""

    def __init__(self):
        self.events_df: Optional[pd.DataFrame] = None
        self.matches_df: Optional[pd.DataFrame] = None
        self.competition_id = 43   # FIFA World Cup (free StatsBomb data)
        self.season_id = 3         # 2018

    def load_data(self) -> pd.DataFrame:
        if STATSBOMB_AVAILABLE:
            return self._load_statsbomb()
        return self._generate_synthetic_data()

    def _load_statsbomb(self) -> pd.DataFrame:
        logger.info("Loading StatsBomb open data...")
        try:
            matches = sb.matches(competition_id=self.competition_id, season_id=self.season_id)
            match_ids = matches["match_id"].tolist()[:10]
            all_events = []
            for mid in match_ids:
                events = sb.events(match_id=mid)
                all_events.append(events)
            raw = pd.concat(all_events, ignore_index=True)

            # ── Normalise to internal schema ──
            df = pd.DataFrame()
            df["player"]     = raw["player"].fillna("Unknown")
            df["team"]       = raw["team"].fillna("Unknown")
            df["position"]   = raw["position"].fillna("Midfielder")
            df["action_type"] = raw["type"].str.lower().str.replace(" ", "_").str.replace("*","",regex=False)
            # Simplify action types to match synthetic schema
            action_map = {
                "pass":"pass", "carry":"carry", "dribble":"dribble", "shot":"shot",
                "tackle":"tackle", "interception":"interception", "clearance":"clearance",
                "ball_receipt":"ball_receipt", "pressure":"pressure",
                "ball_recovery":"ball_receipt", "duel":"tackle", "block":"clearance",
                "miscontrol":"carry", "50/50":"pressure",
            }
            df["action_type"] = df["action_type"].map(
                lambda x: next((v for k,v in action_map.items() if k in str(x)), "pass")
            )
            # Location
            locs = raw["location"].apply(lambda x: x if isinstance(x, list) else [60,40])
            df["location_x"] = locs.apply(lambda x: float(x[0]) if isinstance(x,list) and len(x)>0 else 60.0)
            df["location_y"] = locs.apply(lambda x: float(x[1]) if isinstance(x,list) and len(x)>1 else 40.0)
            df["distance_to_goal"] = np.sqrt((120-df["location_x"])**2 + (40-df["location_y"])**2).round(2)

            # Outcome — use shot_outcome where available, else pass_outcome, else successful
            def get_outcome(row):
                if pd.notna(raw.loc[row.name, "shot_outcome"] if "shot_outcome" in raw.columns else np.nan):
                    o = str(raw.loc[row.name, "shot_outcome"]).lower()
                    return "successful" if o == "goal" else "unsuccessful"
                if "pass_outcome" in raw.columns and pd.notna(raw.loc[row.name, "pass_outcome"]):
                    return "unsuccessful"  # NaN pass_outcome = complete
                return "successful"

            df["outcome"] = "successful"
            if "pass_outcome" in raw.columns:
                df.loc[raw["pass_outcome"].notna(), "outcome"] = "unsuccessful"
            if "shot_outcome" in raw.columns:
                df.loc[raw["shot_outcome"].str.lower().isin(["goal"]), "outcome"] = "successful"

            # Goal flag
            df["goal"] = 0
            if "shot_outcome" in raw.columns:
                df.loc[raw["shot_outcome"].str.lower() == "goal", "goal"] = 1

            df["under_pressure"] = raw["under_pressure"].fillna(False).astype(int)
            df["minute"]         = raw["minute"].fillna(45).astype(int)
            df["match_id"]       = raw.get("match_id", pd.Series(1, index=raw.index)).fillna(1).astype(int)

            # Map position names to Forward/Midfielder/Defender
            def map_pos(p):
                p = str(p).lower()
                if any(x in p for x in ["forward","wing","attack","striker","center forward"]): return "Forward"
                if any(x in p for x in ["midfield","center mid","cam","cdm"]): return "Midfielder"
                return "Defender"
            df["position"] = df["position"].apply(map_pos)

            # Drop unknown players
            df = df[df["player"] != "Unknown"].reset_index(drop=True)
            self.events_df = df
            logger.info(f"Loaded & normalised {len(self.events_df)} events from StatsBomb.")
            return self.events_df
        except Exception as e:
            logger.warning(f"StatsBomb load failed: {e}. Falling back to synthetic.")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generates realistic synthetic football event data."""
        logger.info("Generating synthetic football data...")
        np.random.seed(42)
        n = 5000

        players = [f"Player_{i}" for i in range(1, 31)]
        teams   = ["Team_A", "Team_B"]
        positions = {p: np.random.choice(["Forward", "Midfielder", "Defender"]) for p in players}

        action_types = ["pass", "carry", "dribble", "shot", "tackle", "interception",
                        "clearance", "ball_receipt", "pressure", "cross"]
        outcomes     = ["successful", "unsuccessful"]

        data = []
        for i in range(n):
            player = np.random.choice(players)
            action = np.random.choice(action_types, p=[0.35,0.20,0.08,0.07,0.07,0.05,0.05,0.07,0.04,0.02])
            x = np.random.uniform(0, 120)
            y = np.random.uniform(0, 80)
            # shot more likely near goal
            if action == "shot":
                x = np.random.uniform(90, 120)
                y = np.random.uniform(20, 60)
            distance_to_goal = math.sqrt((120 - x)**2 + (40 - y)**2)
            goal = 1 if (action == "shot" and distance_to_goal < 20 and np.random.random() < 0.25) else 0
            outcome = np.random.choice(outcomes, p=[0.72, 0.28])

            data.append({
                "player":            player,
                "team":              np.random.choice(teams),
                "position":          positions[player],
                "action_type":       action,
                "location_x":        round(x, 2),
                "location_y":        round(y, 2),
                "distance_to_goal":  round(distance_to_goal, 2),
                "outcome":           outcome,
                "goal":              goal,
                "under_pressure":    np.random.choice([0, 1], p=[0.65, 0.35]),
                "minute":            np.random.randint(1, 91),
                "match_id":          np.random.randint(1, 11),
            })

        self.events_df = pd.DataFrame(data)
        logger.info(f"Generated {len(self.events_df)} synthetic events.")
        return self.events_df


# ══════════════════════════════════════════════════════════════
# MODULE 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Transforms raw events into model-ready features and player stats."""

    def __init__(self, events_df: pd.DataFrame):
        self.df = events_df.copy()
        self.player_stats: Optional[pd.DataFrame] = None

    def build_player_stats(self) -> pd.DataFrame:
        """Aggregate per-player statistics for scouting."""
        df = self.df

        def safe_count(group, action, outcome=None):
            filtered = group[group["action_type"] == action]
            if outcome:
                filtered = filtered[filtered["outcome"] == outcome]
            return len(filtered)

        stats = []
        for player, grp in df.groupby("player"):
            pos = grp["position"].iloc[0]
            total_actions = len(grp)
            shots        = safe_count(grp, "shot")
            shots_on_tgt = safe_count(grp, "shot", "successful")
            goals        = int(grp["goal"].sum())
            passes       = safe_count(grp, "pass")
            passes_succ  = safe_count(grp, "pass", "successful")
            pass_pct     = round(passes_succ / max(passes, 1) * 100, 1)
            key_passes   = max(0, shots_on_tgt - 1)  # proxy
            dribbles     = safe_count(grp, "dribble")
            drib_comp    = safe_count(grp, "dribble", "successful")
            tackles      = safe_count(grp, "tackle")
            tackles_won  = safe_count(grp, "tackle", "successful")
            intercepts   = safe_count(grp, "interception")
            clearances   = safe_count(grp, "clearance")
            recoveries   = intercepts + clearances  # proxy
            carries      = safe_count(grp, "carry")
            avg_x        = round(grp["location_x"].mean(), 1)

            stats.append({
                "player": player, "position": pos, "total_actions": total_actions,
                "shots": shots, "shots_on_target": shots_on_tgt, "goals": goals,
                "passes": passes, "pass_completion_pct": pass_pct, "key_passes": key_passes,
                "dribbles": dribbles, "dribbles_completed": drib_comp,
                "tackles": tackles, "tackles_won": tackles_won,
                "interceptions": intercepts, "clearances": clearances,
                "ball_recoveries": recoveries, "carries": carries, "avg_x": avg_x,
            })

        self.player_stats = pd.DataFrame(stats)
        return self.player_stats

    def build_action_features(self) -> tuple:
        """Build features for action-prediction classification."""
        df = self.df.copy()
        le_action = LabelEncoder()
        le_outcome = LabelEncoder()

        df["action_encoded"]  = le_action.fit_transform(df["action_type"])
        df["outcome_encoded"] = le_outcome.fit_transform(df["outcome"])

        features = ["location_x","location_y","distance_to_goal","under_pressure",
                    "minute","outcome_encoded"]
        X = df[features].fillna(0).values
        y = df["action_encoded"].values
        return X, y, le_action.classes_.tolist()

    def build_xg_features(self) -> tuple:
        """Build features for expected-goals model (shots only)."""
        shots = self.df[self.df["action_type"] == "shot"].copy()
        if len(shots) < 10:
            return None, None
        angle = np.arctan2(shots["location_y"] - 40, 120 - shots["location_x"])
        shots["shot_angle"] = np.abs(angle.values)
        features = ["location_x","location_y","distance_to_goal","under_pressure","shot_angle"]
        X = shots[features].fillna(0).values
        y = shots["goal"].values
        return X, y


# ══════════════════════════════════════════════════════════════
# MODULE 3 — ML MODELS
# ══════════════════════════════════════════════════════════════

class MLModels:
    """Trains and stores all ML models."""

    MODEL_DIR = Path("models")

    def __init__(self):
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.action_model    = None
        self.xg_model        = None
        self.cluster_model   = None
        self.scaler          = StandardScaler()
        self.action_labels   = []
        self.cluster_labels  = {}

    # ── Action Prediction ──
    def train_action_model(self, X, y, labels):
        logger.info("Training action prediction model...")
        self.action_labels = labels
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        logger.info(f"Action model accuracy: {acc:.2%}")
        self.action_model = model
        joblib.dump(model, self.MODEL_DIR / "action_model.pkl")
        return acc

    def predict_action(self, location_x, location_y, distance_to_goal,
                       under_pressure=0, minute=45, outcome_encoded=1) -> Dict:
        if self.action_model is None:
            return {}
        X = np.array([[location_x, location_y, distance_to_goal,
                       under_pressure, minute, outcome_encoded]])
        probs = self.action_model.predict_proba(X)[0]
        return {label: round(float(p), 4) for label, p in zip(self.action_labels, probs)}

    # ── Expected Goals (xG) ──
    def train_xg_model(self, X, y):
        logger.info("Training xG (goal probability) model...")
        if X is None or len(X) < 10:
            logger.warning("Insufficient shot data for xG model.")
            return 0.0
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        logger.info(f"xG model accuracy: {acc:.2%}")
        self.xg_model = model
        joblib.dump(model, self.MODEL_DIR / "xg_model.pkl")
        return acc

    def predict_xg(self, location_x, location_y, distance_to_goal,
                   under_pressure=0, shot_angle=0.5) -> float:
        if self.xg_model is None:
            return 0.0
        X = np.array([[location_x, location_y, distance_to_goal, under_pressure, shot_angle]])
        return round(float(self.xg_model.predict_proba(X)[0][1]), 4)

    # ── Player Clustering ──
    def train_clustering(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        logger.info("Training player clustering model...")
        features = ["shots_on_target","goals","pass_completion_pct","key_passes",
                    "dribbles_completed","tackles_won","interceptions","ball_recoveries"]
        X = player_stats[features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        player_stats = player_stats.copy()
        player_stats["cluster"] = kmeans.labels_
        self.cluster_model = kmeans
        # Name clusters
        cluster_names = {
            0: "Defensive Anchor", 1: "Creative Playmaker",
            2: "Goal Threat", 3: "Pressing Machine", 4: "Box-to-Box"
        }
        player_stats["player_profile"] = player_stats["cluster"].map(cluster_names)
        self.cluster_labels = cluster_names
        joblib.dump(kmeans, self.MODEL_DIR / "cluster_model.pkl")
        logger.info("Clustering complete.")
        return player_stats


# ══════════════════════════════════════════════════════════════
# MODULE 4 — SCOUTING ENGINE
# ══════════════════════════════════════════════════════════════

class ScoutingEngine:
    """Scores and ranks players by position-specific scouting metrics."""

    WEIGHTS = {
        "Forward":    {"shots_on_target": 0.30, "goals": 0.40, "dribbles_completed": 0.30},
        "Midfielder": {"pass_completion_pct": 0.35, "key_passes": 0.40, "ball_recoveries": 0.25},
        "Defender":   {"tackles_won": 0.35, "interceptions": 0.30, "ball_recoveries": 0.20, "clearances": 0.15},
    }

    def score_players(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        df = player_stats.copy()
        df["scouting_score"] = 0.0

        for position, weights in self.WEIGHTS.items():
            mask = df["position"] == position
            subset = df[mask].copy()
            if subset.empty:
                continue
            score = pd.Series(0.0, index=subset.index)
            for metric, weight in weights.items():
                if metric in subset.columns:
                    col = subset[metric].fillna(0)
                    max_val = col.max()
                    normalized = col / max_val if max_val > 0 else col
                    score += normalized * weight
            df.loc[mask, "scouting_score"] = (score * 100).round(1)

        return df.sort_values("scouting_score", ascending=False)

    def recommend_players(self, df: pd.DataFrame, position: str, top_n: int = 5) -> List[Dict]:
        filtered = df[df["position"] == position].nlargest(top_n, "scouting_score")
        return filtered[["player","position","scouting_score","player_profile",
                          "goals","shots_on_target","pass_completion_pct",
                          "tackles_won","interceptions","dribbles_completed"]].to_dict("records")

    def get_strengths(self, row: pd.Series) -> List[str]:
        strengths = []
        if row.get("goals", 0) >= 3:        strengths.append("Clinical Finisher")
        if row.get("pass_completion_pct",0) >= 80: strengths.append("Precise Passer")
        if row.get("tackles_won", 0) >= 5:  strengths.append("Strong Tackler")
        if row.get("dribbles_completed",0) >= 4: strengths.append("Skillful Dribbler")
        if row.get("interceptions", 0) >= 5: strengths.append("Intelligent Interceptor")
        if row.get("key_passes", 0) >= 3:   strengths.append("Creative Playmaker")
        if not strengths:                   strengths.append("Versatile Contributor")
        return strengths


# ══════════════════════════════════════════════════════════════
# MODULE 5 — SYSTEM ORCHESTRATOR (boots everything)
# ══════════════════════════════════════════════════════════════

class FootballIntelSystem:
    def __init__(self):
        self.pipeline   = DataPipeline()
        self.fe         = None
        self.ml         = MLModels()
        self.scouting   = ScoutingEngine()
        self.player_stats_scored: Optional[pd.DataFrame] = None

    def boot(self):
        logger.info("═══ Booting Football Intelligence System ═══")
        events = self.pipeline.load_data()
        self.fe = FeatureEngineer(events)

        # Feature engineering
        player_stats = self.fe.build_player_stats()
        X_action, y_action, labels = self.fe.build_action_features()
        X_xg, y_xg = self.fe.build_xg_features()

        # Train models
        self.ml.train_action_model(X_action, y_action, labels)
        self.ml.train_xg_model(X_xg, y_xg)
        player_stats = self.ml.train_clustering(player_stats)

        # Scouting scores
        self.player_stats_scored = self.scouting.score_players(player_stats)
        logger.info("═══ System Ready ═══")

    def get_player_df(self) -> pd.DataFrame:
        return self.player_stats_scored if self.player_stats_scored is not None else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# MODULE 6 — FASTAPI BACKEND + REACT FRONTEND
# ══════════════════════════════════════════════════════════════

system = FootballIntelSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    system.boot()
    yield

app = FastAPI(title="Football Intelligence System", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic Models ──
class ActionRequest(BaseModel):
    location_x: float = 80.0
    location_y: float = 40.0
    distance_to_goal: float = 25.0
    under_pressure: int = 0
    minute: int = 45

class XGRequest(BaseModel):
    location_x: float = 105.0
    location_y: float = 34.0
    distance_to_goal: float = 15.0
    under_pressure: int = 0
    shot_angle: float = 0.5


# ── API Endpoints ──
@app.get("/api/players")
def get_players(position: Optional[str] = None, top: int = 20):
    df = system.get_player_df()
    if df.empty:
        raise HTTPException(500, "System not ready")
    if position:
        df = df[df["position"] == position]
    cols = ["player","position","scouting_score","player_profile","goals",
            "shots_on_target","pass_completion_pct","tackles_won",
            "interceptions","dribbles_completed","key_passes","ball_recoveries"]
    return df[cols].head(top).fillna(0).to_dict("records")

@app.get("/api/scout/{position}")
def scout_position(position: str, top_n: int = 5):
    if position not in ["Forward","Midfielder","Defender"]:
        raise HTTPException(400, "Position must be Forward, Midfielder, or Defender")
    df = system.get_player_df()
    results = system.scouting.recommend_players(df, position, top_n)
    return {"position": position, "recommendations": results}

@app.post("/api/predict/action")
def predict_action(req: ActionRequest):
    probs = system.ml.predict_action(
        req.location_x, req.location_y, req.distance_to_goal,
        req.under_pressure, req.minute
    )
    if not probs:
        raise HTTPException(500, "Action model not ready")
    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
    return {"action_probabilities": sorted_probs, "top_action": list(sorted_probs.keys())[0]}

@app.post("/api/predict/xg")
def predict_xg(req: XGRequest):
    xg = system.ml.predict_xg(
        req.location_x, req.location_y, req.distance_to_goal,
        req.under_pressure, req.shot_angle
    )
    return {"xG": xg, "goal_probability_pct": round(xg * 100, 1)}

@app.get("/api/stats/overview")
def stats_overview():
    df = system.get_player_df()
    if df.empty:
        return {}
    return {
        "total_players": len(df),
        "positions": df["position"].value_counts().to_dict(),
        "profiles": df["player_profile"].value_counts().to_dict(),
        "top_scorer": df.nlargest(1,"goals")[["player","goals"]].to_dict("records")[0],
        "top_defender": df[df["position"]=="Defender"].nlargest(1,"tackles_won")[["player","tackles_won"]].to_dict("records")[0] if not df[df["position"]=="Defender"].empty else {},
        "top_midfielder": df[df["position"]=="Midfielder"].nlargest(1,"pass_completion_pct")[["player","pass_completion_pct"]].to_dict("records")[0] if not df[df["position"]=="Midfielder"].empty else {},
    }

@app.get("/api/player/{player_name}")
def get_player(player_name: str):
    df = system.get_player_df()
    row = df[df["player"] == player_name]
    if row.empty:
        raise HTTPException(404, f"Player '{player_name}' not found")
    data = row.iloc[0].to_dict()
    data["strengths"] = system.scouting.get_strengths(row.iloc[0])
    return data


# ══════════════════════════════════════════════════════════════
# MODULE 7 — REACT FRONTEND (served via FastAPI)
# ══════════════════════════════════════════════════════════════

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Football Intelligence System</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.2/babel.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --pitch: #0a1628;
    --panel: #0f1e35;
    --card:  #152542;
    --line:  #1e3358;
    --accent:#00e5a0;
    --gold:  #ffd166;
    --red:   #ef4565;
    --blue:  #4cc9f0;
    --text:  #e8f4f8;
    --muted: #6b8cad;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--pitch); color:var(--text); font-family:'Inter',sans-serif; min-height:100vh; }

  /* ── HEADER ── */
  header {
    background: linear-gradient(135deg, #0a1628 0%, #0d2040 50%, #0a1628 100%);
    border-bottom: 1px solid var(--line);
    padding: 0 2rem;
    display: flex; align-items:center; justify-content:space-between;
    height: 64px;
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(10px);
  }
  .logo { font-family:'Barlow Condensed',sans-serif; font-size:1.5rem; font-weight:800; letter-spacing:2px; }
  .logo span { color:var(--accent); }
  .status-dot { width:8px; height:8px; border-radius:50%; background:var(--accent); animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

  /* ── NAV ── */
  nav { display:flex; gap:0.25rem; padding:1rem 2rem 0; border-bottom:1px solid var(--line); overflow-x:auto; }
  .nav-btn {
    font-family:'Barlow Condensed',sans-serif; font-size:0.85rem; font-weight:600; letter-spacing:1.5px;
    text-transform:uppercase; padding:0.6rem 1.2rem; border:none; border-radius:6px 6px 0 0;
    cursor:pointer; transition:all 0.2s; background:transparent; color:var(--muted);
    border-bottom:2px solid transparent; white-space:nowrap;
  }
  .nav-btn.active { color:var(--accent); border-bottom-color:var(--accent); background:rgba(0,229,160,0.06); }
  .nav-btn:hover:not(.active) { color:var(--text); background:rgba(255,255,255,0.04); }

  /* ── LAYOUT ── */
  main { padding:1.5rem 2rem; max-width:1400px; margin:0 auto; }
  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; }
  .grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:1.25rem; }
  .grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; }
  @media(max-width:900px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr;}}

  /* ── CARDS ── */
  .card {
    background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:1.25rem; position:relative; overflow:hidden;
    transition: transform 0.2s, border-color 0.2s;
  }
  .card:hover { transform:translateY(-2px); border-color:rgba(0,229,160,0.3); }
  .card-accent { border-left:3px solid var(--accent); }
  .card-gold   { border-left:3px solid var(--gold); }
  .card-blue   { border-left:3px solid var(--blue); }
  .card-red    { border-left:3px solid var(--red); }

  .card h3 { font-family:'Barlow Condensed',sans-serif; font-size:0.75rem; letter-spacing:2px;
             text-transform:uppercase; color:var(--muted); margin-bottom:0.5rem; }
  .card .big-num { font-family:'Barlow Condensed',sans-serif; font-size:2.8rem; font-weight:700;
                   line-height:1; color:var(--text); }
  .card .big-num span { color:var(--accent); }
  .card .sub { font-size:0.78rem; color:var(--muted); margin-top:0.3rem; }

  /* ── SECTION TITLE ── */
  .section-title {
    font-family:'Barlow Condensed',sans-serif; font-size:1.3rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase; color:var(--text);
    margin:1.5rem 0 1rem; display:flex; align-items:center; gap:0.75rem;
  }
  .section-title::after { content:''; flex:1; height:1px; background:var(--line); }

  /* ── TABLE ── */
  .table-wrap { overflow-x:auto; border-radius:10px; border:1px solid var(--line); }
  table { width:100%; border-collapse:collapse; font-size:0.83rem; }
  th { background:#0d1e36; color:var(--muted); font-family:'Barlow Condensed',sans-serif;
       font-size:0.72rem; letter-spacing:1.5px; text-transform:uppercase;
       padding:0.75rem 1rem; text-align:left; white-space:nowrap; }
  td { padding:0.7rem 1rem; border-top:1px solid var(--line); white-space:nowrap; }
  tr:hover td { background:rgba(0,229,160,0.04); }
  .score-bar { display:flex; align-items:center; gap:0.5rem; }
  .bar { height:6px; border-radius:3px; background:linear-gradient(90deg,var(--accent),var(--blue)); }
  .rank { font-family:'Barlow Condensed',sans-serif; font-weight:700; color:var(--muted); }
  .rank-1 { color:var(--gold); }
  .rank-2 { color:#c0c0c0; }
  .rank-3 { color:#cd7f32; }

  /* ── PITCH VISUALIZER ── */
  .pitch-container { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:1.25rem; }
  .pitch { width:100%; aspect-ratio:120/80; background:#1a3a1a; border-radius:8px;
           position:relative; border:2px solid #2d5a2d; overflow:hidden; }
  .pitch-lines { position:absolute; inset:0; }
  .shot-dot { position:absolute; width:12px; height:12px; border-radius:50%;
              transform:translate(-50%,-50%); cursor:pointer; transition:transform 0.2s;
              border:2px solid rgba(255,255,255,0.3); }
  .shot-dot:hover { transform:translate(-50%,-50%) scale(1.5); z-index:10; }

  /* ── FORM CONTROLS ── */
  .form-group { margin-bottom:1rem; }
  label { display:block; font-size:0.75rem; color:var(--muted); margin-bottom:0.4rem;
          font-family:'Barlow Condensed',sans-serif; letter-spacing:1px; text-transform:uppercase; }
  input[type=range] { width:100%; accent-color:var(--accent); cursor:pointer; }
  .range-val { font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; font-weight:600;
               color:var(--accent); float:right; }
  select, input[type=text] {
    width:100%; background:var(--panel); border:1px solid var(--line); color:var(--text);
    padding:0.6rem 0.8rem; border-radius:8px; font-size:0.85rem; outline:none;
  }
  select:focus, input[type=text]:focus { border-color:var(--accent); }

  .btn {
    font-family:'Barlow Condensed',sans-serif; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; padding:0.65rem 1.5rem; border:none; border-radius:8px;
    cursor:pointer; font-size:0.9rem; transition:all 0.2s;
  }
  .btn-primary { background:var(--accent); color:#0a1628; }
  .btn-primary:hover { background:#00c988; transform:translateY(-1px); }
  .btn-secondary { background:var(--line); color:var(--text); }
  .btn-secondary:hover { background:var(--card); }

  /* ── PROB BARS ── */
  .prob-list { display:flex; flex-direction:column; gap:0.6rem; }
  .prob-row { display:flex; align-items:center; gap:0.75rem; }
  .prob-label { font-size:0.78rem; width:120px; color:var(--text); text-transform:capitalize; flex-shrink:0; }
  .prob-track { flex:1; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
  .prob-fill { height:100%; border-radius:4px; transition:width 0.6s ease; }
  .prob-pct { font-family:'Barlow Condensed',sans-serif; font-size:0.85rem; font-weight:600;
              width:42px; text-align:right; color:var(--accent); flex-shrink:0; }

  /* ── TAGS ── */
  .tag { display:inline-block; padding:0.2rem 0.6rem; border-radius:20px; font-size:0.72rem;
         font-family:'Barlow Condensed',sans-serif; letter-spacing:0.5px; margin:0.15rem; }
  .tag-green  { background:rgba(0,229,160,0.15); color:var(--accent); border:1px solid rgba(0,229,160,0.3); }
  .tag-blue   { background:rgba(76,201,240,0.15); color:var(--blue); border:1px solid rgba(76,201,240,0.3); }
  .tag-gold   { background:rgba(255,209,102,0.15); color:var(--gold); border:1px solid rgba(255,209,102,0.3); }
  .tag-red    { background:rgba(239,69,101,0.15); color:var(--red); border:1px solid rgba(239,69,101,0.3); }

  /* ── PLAYER CARD ── */
  .player-card {
    background:var(--card); border:1px solid var(--line); border-radius:12px; padding:1.25rem;
    transition:all 0.2s;
  }
  .player-card:hover { border-color:var(--accent); transform:translateY(-2px); }
  .player-card .name { font-family:'Barlow Condensed',sans-serif; font-size:1.15rem; font-weight:700; }
  .player-card .pos  { font-size:0.72rem; color:var(--muted); }
  .stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:0.75rem; }
  .stat-item { background:var(--panel); border-radius:6px; padding:0.4rem 0.6rem; }
  .stat-item .s-val { font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; font-weight:600; color:var(--accent); }
  .stat-item .s-lbl { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }

  /* ── LOADING ── */
  .loading { display:flex; justify-content:center; align-items:center; height:200px; gap:0.5rem; }
  .dot { width:10px; height:10px; border-radius:50%; background:var(--accent);
         animation: bounce 1.2s infinite; }
  .dot:nth-child(2){animation-delay:0.2s;} .dot:nth-child(3){animation-delay:0.4s;}
  @keyframes bounce{0%,80%,100%{transform:scale(0);}40%{transform:scale(1);}}

  .empty-state { text-align:center; padding:3rem; color:var(--muted); font-size:0.9rem; }

  /* ── XG METER ── */
  .xg-meter { position:relative; }
  .xg-arc { width:180px; height:90px; margin:0 auto; }
  .xg-value { font-family:'Barlow Condensed',sans-serif; font-size:3.5rem; font-weight:800;
              text-align:center; line-height:1; }
  .xg-label { text-align:center; font-size:0.75rem; color:var(--muted); letter-spacing:2px;
              text-transform:uppercase; margin-top:0.25rem; }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const { useState, useEffect, useCallback } = React;

const API = "";  // same origin

// ── Fetch helpers ──
const fetchJSON = (url, opts) => fetch(url, opts).then(r => r.json());

// ── Top-level App ──
function App() {
  const [tab, setTab] = useState("dashboard");
  const [overview, setOverview] = useState(null);

  useEffect(() => {
    fetchJSON(`${API}/api/stats/overview`).then(setOverview).catch(console.error);
  }, []);

  const tabs = [
    {id:"dashboard",  label:"⚡ Dashboard"},
    {id:"scouting",   label:"🔍 Scouting"},
    {id:"action",     label:"🎯 Action Predictor"},
    {id:"xg",         label:"⚽ xG Model"},
    {id:"players",    label:"👤 All Players"},
  ];

  return (
    <div>
      <header>
        <div className="logo">FOOTBALL <span>INTEL</span> SYSTEM</div>
        <div style={{display:"flex",alignItems:"center",gap:"0.5rem"}}>
          <div className="status-dot"/>
          <span style={{fontSize:"0.75rem",color:"var(--muted)"}}>LIVE</span>
        </div>
      </header>

      <nav>
        {tabs.map(t => (
          <button key={t.id} className={`nav-btn${tab===t.id?" active":""}`}
                  onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>

      <main>
        {tab==="dashboard" && <Dashboard overview={overview}/>}
        {tab==="scouting"  && <Scouting/>}
        {tab==="action"    && <ActionPredictor/>}
        {tab==="xg"        && <XGModel/>}
        {tab==="players"   && <AllPlayers/>}
      </main>
    </div>
  );
}

// ── Dashboard ──
function Dashboard({ overview }) {
  if (!overview) return <div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div>;

  const kpis = [
    {label:"Total Players", value: overview.total_players, color:"accent"},
    {label:"Top Scorer",    value: overview.top_scorer?.player?.replace("Player_","#"),
     sub: `${overview.top_scorer?.goals} goals`, color:"gold"},
    {label:"Best Defender", value: overview.top_defender?.player?.replace("Player_","#"),
     sub: `${overview.top_defender?.tackles_won} tackles`, color:"blue"},
    {label:"Best Midfielder",value: overview.top_midfielder?.player?.replace("Player_","#"),
     sub: `${overview.top_midfielder?.pass_completion_pct}% pass acc`, color:"red"},
  ];

  const posColors = {Forward:"var(--red)",Midfielder:"var(--blue)",Defender:"var(--accent)"};

  return (
    <div>
      <div className="grid-4" style={{marginTop:"0.5rem"}}>
        {kpis.map((k,i) => (
          <div key={i} className={`card card-${k.color}`}>
            <h3>{k.label}</h3>
            <div className="big-num">{k.value}</div>
            {k.sub && <div className="sub">{k.sub}</div>}
          </div>
        ))}
      </div>

      <div className="section-title">Squad Composition</div>
      <div className="grid-2">
        <div className="card">
          <h3>Position Breakdown</h3>
          <div style={{marginTop:"1rem",display:"flex",flexDirection:"column",gap:"0.75rem"}}>
            {Object.entries(overview.positions||{}).map(([pos,cnt]) => (
              <div key={pos}>
                <div style={{display:"flex",justifyContent:"space-between",marginBottom:"0.3rem"}}>
                  <span style={{fontSize:"0.85rem"}}>{pos}</span>
                  <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:posColors[pos]}}>{cnt}</span>
                </div>
                <div style={{height:"6px",background:"var(--line)",borderRadius:"3px",overflow:"hidden"}}>
                  <div style={{height:"100%",width:`${cnt/overview.total_players*100}%`,
                               background:posColors[pos],borderRadius:"3px",transition:"width 1s"}}/>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3>Player Profiles (Clusters)</h3>
          <div style={{marginTop:"1rem",display:"flex",flexDirection:"column",gap:"0.5rem"}}>
            {Object.entries(overview.profiles||{}).map(([profile,cnt],i) => {
              const colors=["var(--accent)","var(--gold)","var(--blue)","var(--red)","#b5179e"];
              return (
                <div key={profile} style={{display:"flex",alignItems:"center",gap:"0.75rem"}}>
                  <div style={{width:"10px",height:"10px",borderRadius:"50%",background:colors[i],flexShrink:0}}/>
                  <span style={{fontSize:"0.82rem",flex:1}}>{profile}</span>
                  <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:colors[i]}}>{cnt}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="section-title">Quick Links</div>
      <div className="grid-3">
        {[{title:"Scout Forwards",desc:"Find goal threats by shots, goals & dribbles",tab:"scouting",color:"red"},
          {title:"Predict Actions",desc:"Model next player action in a sequence",tab:"action",color:"blue"},
          {title:"xG Calculator",desc:"Estimate goal probability from shot position",tab:"xg",color:"accent"},
        ].map(c => (
          <div key={c.tab} className={`card card-${c.color}`} style={{cursor:"pointer"}}
               onClick={()=>document.querySelector(`[data-tab="${c.tab}"]`)?.click()}>
            <h3 style={{color:`var(--${c.color})`}}>{c.title}</h3>
            <p style={{fontSize:"0.83rem",color:"var(--muted)",marginTop:"0.5rem"}}>{c.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Scouting Board ──
function Scouting() {
  const [position, setPosition] = useState("Forward");
  const [results, setResults]   = useState([]);
  const [loading, setLoading]   = useState(false);

  const scout = useCallback(() => {
    setLoading(true);
    fetchJSON(`${API}/api/scout/${position}?top_n=8`)
      .then(d => { setResults(d.recommendations||[]); setLoading(false); })
      .catch(() => setLoading(false));
  }, [position]);

  useEffect(() => { scout(); }, [scout]);

  const metricsByPos = {
    Forward:    [{k:"goals",l:"Goals"},{k:"shots_on_target",l:"Shots on Target"},{k:"dribbles_completed",l:"Dribbles"}],
    Midfielder: [{k:"pass_completion_pct",l:"Pass %"},{k:"key_passes",l:"Key Passes"},{k:"ball_recoveries",l:"Recoveries"}],
    Defender:   [{k:"tackles_won",l:"Tackles Won"},{k:"interceptions",l:"Interceptions"},{k:"ball_recoveries",l:"Recoveries"}],
  };
  const metrics = metricsByPos[position] || [];

  return (
    <div>
      <div style={{display:"flex",gap:"1rem",alignItems:"center",marginBottom:"1.25rem",flexWrap:"wrap"}}>
        <div className="section-title" style={{margin:0}}>Scouting Board</div>
        {["Forward","Midfielder","Defender"].map(p => (
          <button key={p} className={`btn ${position===p?"btn-primary":"btn-secondary"}`}
                  onClick={()=>setPosition(p)}>{p}</button>
        ))}
      </div>

      {loading ? <div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div> :
      results.length === 0 ? <div className="empty-state">No data available</div> :
      <div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Player</th>
                <th>Profile</th>
                <th>Score</th>
                {metrics.map(m => <th key={m.k}>{m.l}</th>)}
              </tr>
            </thead>
            <tbody>
              {results.map((p,i) => (
                <tr key={p.player}>
                  <td><span className={`rank rank-${i+1}`}>{i+1}</span></td>
                  <td><strong>{p.player}</strong></td>
                  <td><span className="tag tag-blue">{p.player_profile||"—"}</span></td>
                  <td>
                    <div className="score-bar">
                      <div className="bar" style={{width:`${p.scouting_score}%`,maxWidth:"80px"}}/>
                      <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:"var(--accent)"}}>
                        {p.scouting_score}
                      </span>
                    </div>
                  </td>
                  {metrics.map(m => (
                    <td key={m.k} style={{fontFamily:"Barlow Condensed",fontWeight:600}}>
                      {typeof p[m.k]==="number" ? (Number.isInteger(p[m.k]) ? p[m.k] : p[m.k].toFixed(1)) : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="section-title" style={{marginTop:"1.5rem"}}>Player Cards</div>
        <div className="grid-3">
          {results.slice(0,6).map(p => (
            <div key={p.player} className="player-card">
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
                <div>
                  <div className="name">{p.player}</div>
                  <div className="pos">{p.position}</div>
                </div>
                <div style={{fontFamily:"Barlow Condensed",fontSize:"1.5rem",fontWeight:800,color:"var(--accent)"}}>
                  {p.scouting_score}
                </div>
              </div>
              <div className="stat-grid">
                {metrics.map(m => (
                  <div key={m.k} className="stat-item">
                    <div className="s-val">{typeof p[m.k]==="number"?(Number.isInteger(p[m.k])?p[m.k]:p[m.k].toFixed(1)):"—"}</div>
                    <div className="s-lbl">{m.l}</div>
                  </div>
                ))}
              </div>
              <div style={{marginTop:"0.75rem"}}>
                <span className="tag tag-green">{p.player_profile||"—"}</span>
              </div>
            </div>
          ))}
        </div>
      </div>}
    </div>
  );
}

// ── Action Predictor ──
function ActionPredictor() {
  const [params, setParams] = useState({location_x:80,location_y:40,distance_to_goal:25,under_pressure:0,minute:45});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const predict = () => {
    setLoading(true);
    fetchJSON(`${API}/api/predict/action`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(params)
    }).then(d => { setResult(d); setLoading(false); }).catch(() => setLoading(false));
  };

  useEffect(() => { predict(); }, []);

  const set = (k,v) => setParams(p => ({...p,[k]:v}));
  const actionColors = {pass:"#4cc9f0",carry:"#7209b7",dribble:"#ffd166",shot:"#ef4565",
                        tackle:"#00e5a0",interception:"#06d6a0",clearance:"#118ab2",
                        ball_receipt:"#8338ec",pressure:"#fb5607",cross:"#ffbe0b"};

  return (
    <div className="grid-2">
      <div className="card">
        <div className="section-title" style={{margin:"0 0 1rem"}}>Pitch Position</div>

        {/* Mini pitch */}
        <div style={{background:"#1a3a1a",borderRadius:"8px",border:"2px solid #2d5a2d",
                     aspectRatio:"120/80",position:"relative",marginBottom:"1.25rem",cursor:"crosshair"}}
             onClick={e => {
               const r = e.currentTarget.getBoundingClientRect();
               const x = ((e.clientX-r.left)/r.width)*120;
               const y = ((e.clientY-r.top)/r.height)*80;
               const dist = Math.sqrt(Math.pow(120-x,2)+Math.pow(40-y,2));
               set("location_x",Math.round(x*10)/10);
               set("location_y",Math.round(y*10)/10);
               set("distance_to_goal",Math.round(dist*10)/10);
             }}>
          {/* Pitch markings */}
          <svg style={{position:"absolute",inset:0,width:"100%",height:"100%"}} viewBox="0 0 120 80">
            <rect x="0" y="0" width="120" height="80" fill="none" stroke="#2d5a2d" strokeWidth="0.5"/>
            <line x1="60" y1="0" x2="60" y2="80" stroke="#2d5a2d" strokeWidth="0.5"/>
            <circle cx="60" cy="40" r="9.15" fill="none" stroke="#2d5a2d" strokeWidth="0.5"/>
            <rect x="102" y="18" width="18" height="44" fill="none" stroke="#2d5a2d" strokeWidth="0.5"/>
            <rect x="114" y="30" width="6" height="20" fill="none" stroke="#4a8a4a" strokeWidth="0.5"/>
            <rect x="0" y="18" width="18" height="44" fill="none" stroke="#2d5a2d" strokeWidth="0.5"/>
            <circle cx={params.location_x} cy={params.location_y} r="2.5" fill="var(--accent)" opacity="0.9">
              <animate attributeName="r" values="2.5;3.5;2.5" dur="1.5s" repeatCount="indefinite"/>
            </circle>
          </svg>
          <div style={{position:"absolute",bottom:"4px",right:"6px",fontSize:"0.6rem",color:"#2d5a2d"}}>
            Click to set position
          </div>
        </div>

        <div className="form-group">
          <label>Minute <span className="range-val">{params.minute}'</span></label>
          <input type="range" min="1" max="90" value={params.minute}
                 onChange={e=>set("minute",+e.target.value)}/>
        </div>
        <div className="form-group">
          <label>Under Pressure</label>
          <select value={params.under_pressure} onChange={e=>set("under_pressure",+e.target.value)}>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0.5rem",marginBottom:"0.75rem",
                     fontSize:"0.78rem",color:"var(--muted)"}}>
          <div>X: <b style={{color:"var(--text)"}}>{params.location_x}</b></div>
          <div>Y: <b style={{color:"var(--text)"}}>{params.location_y}</b></div>
          <div>Dist: <b style={{color:"var(--text)"}}>{params.distance_to_goal}m</b></div>
        </div>
        <button className="btn btn-primary" onClick={predict} style={{width:"100%"}}>
          {loading ? "Predicting…" : "Predict Action"}
        </button>
      </div>

      <div className="card">
        <div className="section-title" style={{margin:"0 0 1rem"}}>Action Probabilities</div>
        {loading && <div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div>}
        {result && !loading && (
          <div>
            <div style={{background:"var(--panel)",borderRadius:"8px",padding:"0.75rem",marginBottom:"1rem",
                         display:"flex",alignItems:"center",gap:"0.75rem"}}>
              <div style={{fontSize:"0.75rem",color:"var(--muted)"}}>Top Predicted Action</div>
              <div style={{fontFamily:"Barlow Condensed",fontSize:"1.4rem",fontWeight:700,
                           color:actionColors[result.top_action]||"var(--accent)",textTransform:"capitalize"}}>
                {result.top_action}
              </div>
            </div>
            <div className="prob-list">
              {Object.entries(result.action_probabilities||{})
                .sort((a,b)=>b[1]-a[1])
                .slice(0,8)
                .map(([action,prob]) => (
                <div key={action} className="prob-row">
                  <div className="prob-label">{action}</div>
                  <div className="prob-track">
                    <div className="prob-fill"
                         style={{width:`${prob*100}%`,background:actionColors[action]||"var(--accent)"}}/>
                  </div>
                  <div className="prob-pct">{(prob*100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── xG Model ──
function XGModel() {
  const [params, setParams] = useState({location_x:105,location_y:34,distance_to_goal:15,under_pressure:0,shot_angle:0.5});
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const predict = () => {
    setLoading(true);
    fetchJSON(`${API}/api/predict/xg`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(params)
    }).then(d => {
      setResult(d);
      setHistory(h => [{...params,xG:d.xG,pct:d.goal_probability_pct},...h].slice(0,5));
      setLoading(false);
    }).catch(()=>setLoading(false));
  };

  useEffect(() => { predict(); }, []);

  const set = (k,v) => setParams(p => ({...p,[k]:v}));
  const xg = result?.xG || 0;
  const color = xg < 0.1 ? "var(--blue)" : xg < 0.3 ? "var(--gold)" : "var(--red)";

  return (
    <div className="grid-2">
      <div className="card">
        <div className="section-title" style={{margin:"0 0 1rem"}}>Shot Parameters</div>

        <div className="form-group">
          <label>Shot Distance (m) <span className="range-val">{params.distance_to_goal}m</span></label>
          <input type="range" min="1" max="40" value={params.distance_to_goal}
                 onChange={e=>{
                   const d=+e.target.value;
                   const x=120-d*0.8;
                   set("distance_to_goal",d); set("location_x",Math.round(x));
                 }}/>
        </div>
        <div className="form-group">
          <label>Shot Angle (rad) <span className="range-val">{params.shot_angle.toFixed(2)}</span></label>
          <input type="range" min="0" max="1.5" step="0.05" value={params.shot_angle}
                 onChange={e=>set("shot_angle",+e.target.value)}/>
        </div>
        <div className="form-group">
          <label>Under Pressure</label>
          <select value={params.under_pressure} onChange={e=>set("under_pressure",+e.target.value)}>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>
        <button className="btn btn-primary" onClick={predict} style={{width:"100%",marginTop:"0.5rem"}}>
          {loading ? "Calculating…" : "Calculate xG"}
        </button>

        {history.length > 0 && (
          <div style={{marginTop:"1.25rem"}}>
            <div style={{fontSize:"0.72rem",color:"var(--muted)",marginBottom:"0.5rem",
                         fontFamily:"Barlow Condensed",letterSpacing:"1px",textTransform:"uppercase"}}>
              Shot History
            </div>
            {history.map((h,i) => (
              <div key={i} style={{display:"flex",justifyContent:"space-between",alignItems:"center",
                                   padding:"0.4rem 0",borderBottom:"1px solid var(--line)",fontSize:"0.8rem"}}>
                <span style={{color:"var(--muted)"}}>Dist: {h.distance_to_goal}m</span>
                <span style={{fontFamily:"Barlow Condensed",fontWeight:700,
                              color:h.xG<0.1?"var(--blue)":h.xG<0.3?"var(--gold)":"var(--red)"}}>
                  {h.pct}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card" style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
        <div style={{fontSize:"0.75rem",color:"var(--muted)",letterSpacing:"2px",textTransform:"uppercase",marginBottom:"1rem"}}>
          Expected Goal Value
        </div>

        {/* Arc meter */}
        <svg width="200" height="110" viewBox="0 0 200 110">
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="var(--line)" strokeWidth="12" strokeLinecap="round"/>
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke={color} strokeWidth="12"
                strokeLinecap="round" strokeDasharray="251"
                strokeDashoffset={251*(1-Math.min(xg*2,1))}
                style={{transition:"stroke-dashoffset 0.8s ease, stroke 0.4s"}}/>
          <text x="100" y="95" textAnchor="middle" fontFamily="Barlow Condensed" fontWeight="800"
                fontSize="28" fill={color}>{(xg*100).toFixed(1)}%</text>
          <text x="100" y="110" textAnchor="middle" fontFamily="Barlow Condensed"
                fontSize="11" fill="#6b8cad" letterSpacing="1">GOAL PROBABILITY</text>
        </svg>

        <div style={{marginTop:"1.5rem",textAlign:"center"}}>
          <div style={{fontFamily:"Barlow Condensed",fontSize:"4rem",fontWeight:"800",color,lineHeight:1}}>
            {xg.toFixed(3)}
          </div>
          <div style={{fontSize:"0.75rem",color:"var(--muted)",letterSpacing:"2px",textTransform:"uppercase",marginTop:"0.25rem"}}>
            xG Score
          </div>
        </div>

        <div style={{marginTop:"1.5rem",display:"flex",gap:"1rem"}}>
          {[["Low","< 0.10","var(--blue)"],["Medium","0.10–0.30","var(--gold)"],["High","≥ 0.30","var(--red)"]].map(([l,r,c])=>(
            <div key={l} style={{textAlign:"center"}}>
              <div style={{width:"10px",height:"10px",borderRadius:"50%",background:c,margin:"0 auto 4px"}}/>
              <div style={{fontSize:"0.65rem",color:"var(--muted)"}}>{l}</div>
              <div style={{fontSize:"0.62rem",color:"var(--muted)"}}>{r}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── All Players ──
function AllPlayers() {
  const [players, setPlayers]   = useState([]);
  const [position, setPosition] = useState("all");
  const [search, setSearch]     = useState("");
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    setLoading(true);
    const pos = position!=="all" ? `?position=${position}&top=50` : "?top=50";
    fetchJSON(`${API}/api/players${pos}`)
      .then(d => { setPlayers(d); setLoading(false); })
      .catch(()=>setLoading(false));
  }, [position]);

  const filtered = players.filter(p =>
    p.player.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <div style={{display:"flex",gap:"0.75rem",marginBottom:"1.25rem",flexWrap:"wrap",alignItems:"center"}}>
        <input type="text" placeholder="Search player…" value={search}
               onChange={e=>setSearch(e.target.value)} style={{maxWidth:"220px"}}/>
        {["all","Forward","Midfielder","Defender"].map(p=>(
          <button key={p} className={`btn ${position===p?"btn-primary":"btn-secondary"}`}
                  onClick={()=>setPosition(p)} style={{textTransform:"capitalize"}}>
            {p==="all"?"All":p}
          </button>
        ))}
      </div>

      {loading ? <div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div> :
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Player</th><th>Pos</th><th>Profile</th>
              <th>Score</th><th>Goals</th><th>SOT</th><th>Pass%</th>
              <th>Key Pass</th><th>Tackles W</th><th>Intercept</th><th>Dribbles</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((p,i)=>(
              <tr key={p.player}>
                <td className="rank">{i+1}</td>
                <td><strong>{p.player}</strong></td>
                <td>
                  <span className={`tag ${p.position==="Forward"?"tag-red":p.position==="Midfielder"?"tag-blue":"tag-green"}`}>
                    {p.position?.slice(0,3)}
                  </span>
                </td>
                <td><span className="tag tag-gold" style={{fontSize:"0.65rem"}}>{p.player_profile||"—"}</span></td>
                <td>
                  <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:"var(--accent)"}}>
                    {p.scouting_score}
                  </span>
                </td>
                <td>{p.goals}</td>
                <td>{p.shots_on_target}</td>
                <td>{p.pass_completion_pct?.toFixed(1)}%</td>
                <td>{p.key_passes}</td>
                <td>{p.tackles_won}</td>
                <td>{p.interceptions}</td>
                <td>{p.dribbles_completed}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>}
    </div>
  );
}

ReactDOM.render(<App/>, document.getElementById("root"));
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("\n" + "═"*60)
    print("  ⚽  FOOTBALL INTELLIGENCE SYSTEM")
    print("  Open your browser → http://localhost:8000")
    print("  API Docs          → http://localhost:8000/docs")
    print("═"*60 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
