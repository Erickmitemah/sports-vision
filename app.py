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
# MODULE 6b — ADMIN CRUD ENDPOINTS
# ══════════════════════════════════════════════════════════════

class PlayerCreate(BaseModel):
    player: str
    position: str
    team: str = "Unknown"
    nationality: str = ""
    age: int = 0
    goals: float = 0
    shots_on_target: float = 0
    pass_completion_pct: float = 0.0
    key_passes: float = 0
    dribbles_completed: float = 0
    tackles_won: float = 0
    interceptions: float = 0
    clearances: float = 0
    ball_recoveries: float = 0
    total_actions: int = 0
    passes: float = 0

class PlayerUpdate(BaseModel):
    position: Optional[str] = None
    team: Optional[str] = None
    nationality: Optional[str] = None
    age: Optional[int] = None
    goals: Optional[float] = None
    shots_on_target: Optional[float] = None
    pass_completion_pct: Optional[float] = None
    key_passes: Optional[float] = None
    dribbles_completed: Optional[float] = None
    tackles_won: Optional[float] = None
    interceptions: Optional[float] = None
    clearances: Optional[float] = None
    ball_recoveries: Optional[float] = None
    total_actions: Optional[int] = None
    passes: Optional[float] = None

def _rescore(df: pd.DataFrame) -> pd.DataFrame:
    try:
        features = ["shots_on_target","goals","pass_completion_pct","key_passes",
                    "dribbles_completed","tackles_won","interceptions","ball_recoveries"]
        X = df[features].fillna(0).values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        km = KMeans(n_clusters=min(5, len(df)), random_state=42, n_init=10)
        km.fit(X_sc)
        df = df.copy()
        df["cluster"] = km.labels_
        cnames = {0:"Defensive Anchor",1:"Creative Playmaker",2:"Goal Threat",
                  3:"Pressing Machine",4:"Box-to-Box"}
        df["player_profile"] = df["cluster"].map(cnames)
    except Exception:
        pass
    return system.scouting.score_players(df)

@app.post("/api/admin/players", status_code=201)
def admin_add_player(p: PlayerCreate):
    df = system.get_player_df()
    if p.player in df["player"].values:
        raise HTTPException(400, f"Player already exists.")
    new_row = {
        "player": p.player, "position": p.position, "team": p.team,
        "nationality": p.nationality, "age": p.age,
        "goals": p.goals, "shots_on_target": p.shots_on_target,
        "pass_completion_pct": p.pass_completion_pct, "key_passes": p.key_passes,
        "dribbles_completed": p.dribbles_completed, "tackles_won": p.tackles_won,
        "interceptions": p.interceptions, "clearances": p.clearances,
        "ball_recoveries": p.ball_recoveries, "total_actions": p.total_actions,
        "passes": p.passes, "avg_x": 60.0, "carries": 0,
        "cluster": 0, "player_profile": "Versatile Contributor", "scouting_score": 0.0,
    }
    updated = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    system.player_stats_scored = _rescore(updated)
    return {"message": f"Player added.", "player": p.player}

@app.put("/api/admin/players/{player_name}")
def admin_update_player(player_name: str, p: PlayerUpdate):
    df = system.get_player_df().copy()
    idx = df.index[df["player"] == player_name].tolist()
    if not idx:
        raise HTTPException(404, f"Player not found.")
    i = idx[0]
    for field, val in p.model_dump(exclude_none=True).items():
        if field in df.columns:
            df.at[i, field] = val
    system.player_stats_scored = _rescore(df)
    return {"message": f"Player updated.", "player": player_name}

@app.delete("/api/admin/players/{player_name}")
def admin_delete_player(player_name: str):
    df = system.get_player_df()
    if player_name not in df["player"].values:
        raise HTTPException(404, f"Player not found.")
    updated = df[df["player"] != player_name].reset_index(drop=True)
    system.player_stats_scored = _rescore(updated)
    return {"message": f"Player removed."}

@app.get("/api/admin/players")
def admin_list_players(search: str = "", position: str = ""):
    df = system.get_player_df()
    if search:
        df = df[df["player"].str.lower().str.contains(search.lower(), na=False)]
    if position:
        df = df[df["position"] == position]
    cols = ["player","position","team","nationality","age","scouting_score","player_profile",
            "goals","shots_on_target","pass_completion_pct","key_passes",
            "dribbles_completed","tackles_won","interceptions","clearances","ball_recoveries"]
    available = [c for c in cols if c in df.columns]
    return df[available].fillna(0).to_dict("records")



# ══════════════════════════════════════════════════════════════
# MODULE 6c — TRAINING STUDIO ENDPOINTS
# ══════════════════════════════════════════════════════════════

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# In-memory training dataset store
training_records: List[Dict] = []

class TrainingRecord(BaseModel):
    player: str
    position: str
    age: int = 25
    goals: float = 0
    shots_on_target: float = 0
    pass_completion_pct: float = 75.0
    key_passes: float = 0
    dribbles_completed: float = 0
    tackles_won: float = 0
    interceptions: float = 0
    clearances: float = 0
    ball_recoveries: float = 0
    matches_played: int = 10
    # Manual quality label (0–100), used as training target
    quality_label: float = 50.0

class TrainRequest(BaseModel):
    model_type: str = "random_forest"   # random_forest | gradient_boost | ridge

# Quality model (retrained on demand)
quality_model_store: Dict = {"model": None, "accuracy": None, "features": [], "trained_on": 0}

QUALITY_FEATURES = [
    "goals","shots_on_target","pass_completion_pct","key_passes",
    "dribbles_completed","tackles_won","interceptions","clearances",
    "ball_recoveries","matches_played","age"
]

def _build_quality_model(records: List[Dict], model_type: str):
    df = pd.DataFrame(records)
    X = df[QUALITY_FEATURES].fillna(0).values
    y = df["quality_label"].values
    if len(df) < 4:
        raise ValueError("Need at least 4 training records to train.")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    if model_type == "gradient_boost":
        from sklearn.ensemble import GradientBoostingRegressor
        mdl = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_type == "ridge":
        from sklearn.linear_model import Ridge
        mdl = Ridge(alpha=1.0)
    else:
        from sklearn.ensemble import RandomForestRegressor
        mdl = RandomForestRegressor(n_estimators=100, random_state=42)
    mdl.fit(X_tr_s, y_tr)
    preds = mdl.predict(X_te_s)
    rmse  = float(np.sqrt(np.mean((preds - y_te)**2)))
    r2    = float(1 - np.sum((preds-y_te)**2)/np.sum((y_te-np.mean(y_te))**2)) if len(y_te)>1 else 0.0
    quality_model_store["model"]      = mdl
    quality_model_store["scaler"]     = scaler
    quality_model_store["accuracy"]   = {"rmse": round(rmse,2), "r2": round(r2,3)}
    quality_model_store["features"]   = QUALITY_FEATURES
    quality_model_store["trained_on"] = len(records)
    quality_model_store["model_type"] = model_type
    return quality_model_store["accuracy"]

def _predict_quality(row: Dict) -> Dict:
    mdl    = quality_model_store.get("model")
    scaler = quality_model_store.get("scaler")
    if mdl is None:
        return {"quality_score": None, "grade": "N/A", "error": "Model not trained yet"}
    X = np.array([[row.get(f,0) for f in QUALITY_FEATURES]])
    X_s = scaler.transform(X)
    raw = float(mdl.predict(X_s)[0])
    score = min(max(round(raw,1), 0), 100)
    grade = "World Class" if score>=85 else "Elite" if score>=75 else "Quality" if score>=65 else "Decent" if score>=50 else "Developing"
    return {"quality_score": score, "grade": grade}

@app.post("/api/train/record")
def add_training_record(r: TrainingRecord):
    rec = r.model_dump()
    # Remove any existing record for same player
    global training_records
    training_records = [x for x in training_records if x["player"] != r.player]
    training_records.append(rec)
    return {"message": f"Record added for {r.player}", "total_records": len(training_records)}

@app.delete("/api/train/record/{player_name}")
def delete_training_record(player_name: str):
    global training_records
    before = len(training_records)
    training_records = [x for x in training_records if x["player"] != player_name]
    if len(training_records) == before:
        raise HTTPException(404, "Record not found")
    return {"message": f"Removed {player_name}", "total_records": len(training_records)}

@app.get("/api/train/records")
def get_training_records():
    return {"records": training_records, "total": len(training_records)}

@app.post("/api/train/run")
def run_training(req: TrainRequest):
    if len(training_records) < 4:
        raise HTTPException(400, f"Need at least 4 records. You have {len(training_records)}.")
    try:
        acc = _build_quality_model(training_records, req.model_type)
        return {
            "success": True,
            "model_type": req.model_type,
            "trained_on": len(training_records),
            "metrics": acc,
            "message": f"Model trained on {len(training_records)} players. RMSE={acc['rmse']}, R²={acc['r2']}"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/train/predict")
def predict_quality(data: dict):
    result = _predict_quality(data)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result

@app.get("/api/train/model-status")
def model_status():
    return {
        "trained": quality_model_store["model"] is not None,
        "model_type": quality_model_store.get("model_type","—"),
        "trained_on": quality_model_store.get("trained_on",0),
        "accuracy": quality_model_store.get("accuracy"),
        "features": QUALITY_FEATURES,
    }

@app.post("/api/train/predict-all")
def predict_all_players():
    """Rate every player in the system using the quality model."""
    mdl = quality_model_store.get("model")
    if mdl is None:
        raise HTTPException(400, "Train the model first.")
    df = system.get_player_df()
    results = []
    for _, row in df.iterrows():
        rec = {f: float(row.get(f,0)) for f in QUALITY_FEATURES}
        rec["matches_played"] = int(row.get("total_actions",50)//5)
        rec["age"] = int(row.get("age",25))
        pred = _predict_quality(rec)
        results.append({
            "player": row["player"],
            "position": row["position"],
            "quality_score": pred["quality_score"],
            "grade": pred["grade"],
            "scouting_score": row.get("scouting_score",0),
        })
    results.sort(key=lambda x: x["quality_score"] or 0, reverse=True)
    return {"ratings": results}


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
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--pitch:#0a1628;--panel:#0f1e35;--card:#152542;--line:#1e3358;--accent:#00e5a0;--gold:#ffd166;--red:#ef4565;--blue:#4cc9f0;--text:#e8f4f8;--muted:#6b8cad;--admin:#b48eff;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--pitch);color:var(--text);font-family:"Inter",sans-serif;min-height:100vh;}
header{background:linear-gradient(135deg,#0a1628,#0d2040,#0a1628);border-bottom:1px solid var(--line);padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:64px;position:sticky;top:0;z-index:100;}
.logo{font-family:"Barlow Condensed",sans-serif;font-size:1.5rem;font-weight:800;letter-spacing:2px;}
.logo span{color:var(--accent);}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.4;}}
nav{display:flex;gap:.25rem;padding:1rem 2rem 0;border-bottom:1px solid var(--line);overflow-x:auto;}
.nav-btn{font-family:"Barlow Condensed",sans-serif;font-size:.85rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;padding:.6rem 1.2rem;border:none;border-radius:6px 6px 0 0;cursor:pointer;transition:all .2s;background:transparent;color:var(--muted);border-bottom:2px solid transparent;white-space:nowrap;}
.nav-btn.active{color:var(--accent);border-bottom-color:var(--accent);background:rgba(0,229,160,.06);}
.nav-btn.admin-btn.active{color:var(--admin);border-bottom-color:var(--admin);background:rgba(180,142,255,.06);}
.nav-btn:hover:not(.active){color:var(--text);background:rgba(255,255,255,.04);}
main{padding:1.5rem 2rem;max-width:1400px;margin:0 auto;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;}
@media(max-width:900px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr;}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1.25rem;position:relative;overflow:hidden;transition:transform .2s,border-color .2s;}
.card:hover{transform:translateY(-2px);border-color:rgba(0,229,160,.3);}
.card-accent{border-left:3px solid var(--accent);}
.card-gold{border-left:3px solid var(--gold);}
.card-blue{border-left:3px solid var(--blue);}
.card-red{border-left:3px solid var(--red);}
.card-admin{border-left:3px solid var(--admin);}
.card h3{font-family:"Barlow Condensed",sans-serif;font-size:.75rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;}
.card .big-num{font-family:"Barlow Condensed",sans-serif;font-size:2.8rem;font-weight:700;line-height:1;color:var(--text);}
.card .sub{font-size:.78rem;color:var(--muted);margin-top:.3rem;}
.section-title{font-family:"Barlow Condensed",sans-serif;font-size:1.3rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--text);margin:1.5rem 0 1rem;display:flex;align-items:center;gap:.75rem;}
.section-title::after{content:"";flex:1;height:1px;background:var(--line);}
.table-wrap{overflow-x:auto;border-radius:10px;border:1px solid var(--line);}
table{width:100%;border-collapse:collapse;font-size:.83rem;}
th{background:#0d1e36;color:var(--muted);font-family:"Barlow Condensed",sans-serif;font-size:.72rem;letter-spacing:1.5px;text-transform:uppercase;padding:.75rem 1rem;text-align:left;white-space:nowrap;}
td{padding:.7rem 1rem;border-top:1px solid var(--line);white-space:nowrap;}
tr:hover td{background:rgba(0,229,160,.04);}
.score-bar{display:flex;align-items:center;gap:.5rem;}
.bar{height:6px;border-radius:3px;background:linear-gradient(90deg,var(--accent),var(--blue));}
.rank{font-family:"Barlow Condensed",sans-serif;font-weight:700;color:var(--muted);}
.rank-1{color:var(--gold);}.rank-2{color:#c0c0c0;}.rank-3{color:#cd7f32;}
.prob-list{display:flex;flex-direction:column;gap:.6rem;}
.prob-row{display:flex;align-items:center;gap:.75rem;}
.prob-label{font-size:.78rem;width:120px;color:var(--text);text-transform:capitalize;flex-shrink:0;}
.prob-track{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden;}
.prob-fill{height:100%;border-radius:4px;transition:width .6s ease;}
.prob-pct{font-family:"Barlow Condensed",sans-serif;font-size:.85rem;font-weight:600;width:42px;text-align:right;color:var(--accent);flex-shrink:0;}
.tag{display:inline-block;padding:.2rem .6rem;border-radius:20px;font-size:.72rem;font-family:"Barlow Condensed",sans-serif;letter-spacing:.5px;margin:.15rem;}
.tag-green{background:rgba(0,229,160,.15);color:var(--accent);border:1px solid rgba(0,229,160,.3);}
.tag-blue{background:rgba(76,201,240,.15);color:var(--blue);border:1px solid rgba(76,201,240,.3);}
.tag-gold{background:rgba(255,209,102,.15);color:var(--gold);border:1px solid rgba(255,209,102,.3);}
.tag-red{background:rgba(239,69,101,.15);color:var(--red);border:1px solid rgba(239,69,101,.3);}
.tag-admin{background:rgba(180,142,255,.15);color:var(--admin);border:1px solid rgba(180,142,255,.3);}
.player-card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1.25rem;transition:all .2s;}
.player-card:hover{border-color:var(--accent);transform:translateY(-2px);}
.player-card .name{font-family:"Barlow Condensed",sans-serif;font-size:1.15rem;font-weight:700;}
.player-card .pos{font-size:.72rem;color:var(--muted);}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-top:.75rem;}
.stat-item{background:var(--panel);border-radius:6px;padding:.4rem .6rem;}
.stat-item .s-val{font-family:"Barlow Condensed",sans-serif;font-size:1.1rem;font-weight:600;color:var(--accent);}
.stat-item .s-lbl{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;}
.loading{display:flex;justify-content:center;align-items:center;height:200px;gap:.5rem;}
.dot{width:10px;height:10px;border-radius:50%;background:var(--accent);animation:bounce 1.2s infinite;}
.dot:nth-child(2){animation-delay:.2s;}.dot:nth-child(3){animation-delay:.4s;}
@keyframes bounce{0%,80%,100%{transform:scale(0);}40%{transform:scale(1);}}
.empty-state{text-align:center;padding:3rem;color:var(--muted);font-size:.9rem;}
.btn{font-family:"Barlow Condensed",sans-serif;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;padding:.65rem 1.5rem;border:none;border-radius:8px;cursor:pointer;font-size:.9rem;transition:all .2s;}
.btn-primary{background:var(--accent);color:#0a1628;}.btn-primary:hover{background:#00c988;transform:translateY(-1px);}
.btn-secondary{background:var(--line);color:var(--text);}.btn-secondary:hover{background:#253d5e;}
.btn-danger{background:rgba(239,69,101,.15);color:var(--red);border:1px solid rgba(239,69,101,.3);}.btn-danger:hover{background:var(--red);color:#fff;}
.btn-edit{background:rgba(180,142,255,.15);color:var(--admin);border:1px solid rgba(180,142,255,.3);}.btn-edit:hover{background:var(--admin);color:#0a1628;}
.btn-sm{padding:.35rem .9rem;font-size:.78rem;}
/* form */
.form-group{margin-bottom:1rem;}
label{display:block;font-size:.75rem;color:var(--muted);margin-bottom:.4rem;font-family:"Barlow Condensed",sans-serif;letter-spacing:1px;text-transform:uppercase;}
input[type=text],input[type=number],input[type=range],select,textarea{width:100%;background:var(--panel);border:1px solid var(--line);color:var(--text);padding:.6rem .8rem;border-radius:8px;font-size:.85rem;outline:none;font-family:"Inter",sans-serif;}
input:focus,select:focus,textarea:focus{border-color:var(--accent);}
input[type=range]{padding:0;accent-color:var(--accent);cursor:pointer;}
.range-val{font-family:"Barlow Condensed",sans-serif;font-size:1.1rem;font-weight:600;color:var(--accent);float:right;}
/* modal */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:200;display:flex;align-items:center;justify-content:center;padding:1rem;}
.modal{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:1.75rem;width:100%;max-width:620px;max-height:90vh;overflow-y:auto;}
.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.25rem;}
.modal-title{font-family:"Barlow Condensed",sans-serif;font-size:1.4rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;}
.close-btn{background:none;border:none;color:var(--muted);font-size:1.4rem;cursor:pointer;line-height:1;}.close-btn:hover{color:var(--text);}
/* toast */
.toast{position:fixed;bottom:2rem;right:2rem;padding:.85rem 1.25rem;border-radius:10px;font-size:.85rem;font-weight:500;z-index:300;animation:fadeIn .3s ease;border:1px solid;}
.toast-success{background:rgba(0,229,160,.15);color:var(--accent);border-color:rgba(0,229,160,.4);}
.toast-error{background:rgba(239,69,101,.15);color:var(--red);border-color:rgba(239,69,101,.4);}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
/* admin table actions */
.action-btns{display:flex;gap:.4rem;}
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const {useState,useEffect,useCallback,useRef} = React;
const API = "";

const fetchJSON = (url,opts) => fetch(url,opts).then(r=>{
  if(!r.ok) return r.json().then(e=>{throw new Error(e.detail||"Error")});
  return r.json();
});

// ── Toast ──
function Toast({msg,type,onDone}){
  useEffect(()=>{const t=setTimeout(onDone,3000);return()=>clearTimeout(t);},[]);
  return <div className={`toast toast-${type}`}>{msg}</div>;
}

// ── App ──
function App(){
  const [tab,setTab]=useState("dashboard");
  const [overview,setOverview]=useState(null);
  const [toast,setToast]=useState(null);

  const showToast=(msg,type="success")=>{setToast({msg,type,id:Date.now()});};

  useEffect(()=>{
    fetchJSON(`${API}/api/stats/overview`).then(setOverview).catch(console.error);
  },[]);

  const tabs=[
    {id:"dashboard",label:"⚡ Dashboard"},
    {id:"scouting",label:"🔍 Scouting"},
    {id:"action",label:"🎯 Action Predictor"},
    {id:"xg",label:"⚽ xG Model"},
    {id:"players",label:"👤 All Players"},
    {id:"admin",label:"⚙ Admin",admin:true},
    {id:"training",label:"🧠 Training Studio",admin:true},
  ];

  return(
    <div>
      <header>
        <div className="logo">FOOTBALL <span>INTEL</span> SYSTEM</div>
        <div style={{display:"flex",alignItems:"center",gap:".5rem"}}>
          <div className="status-dot"/>
          <span style={{fontSize:".75rem",color:"var(--muted)"}}>LIVE</span>
        </div>
      </header>
      <nav>
        {tabs.map(t=>(
          <button key={t.id} className={`nav-btn${t.admin?" admin-btn":""}${tab===t.id?" active":""}`}
                  onClick={()=>setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      <main>
        {tab==="dashboard" && <Dashboard overview={overview}/>}
        {tab==="scouting"  && <Scouting/>}
        {tab==="action"    && <ActionPredictor/>}
        {tab==="xg"        && <XGModel/>}
        {tab==="players"   && <AllPlayers/>}
        {tab==="admin"     && <AdminDashboard showToast={showToast}/>}
        {tab==="training"  && <TrainingStudio showToast={showToast}/>}
      </main>
      {toast && <Toast key={toast.id} msg={toast.msg} type={toast.type} onDone={()=>setToast(null)}/>}
    </div>
  );
}

// ── Dashboard ──
function Dashboard({overview}){
  if(!overview) return <div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div>;
  const kpis=[
    {label:"Total Players",value:overview.total_players,color:"accent"},
    {label:"Top Scorer",value:overview.top_scorer?.player?.replace("Player_","#"),sub:`${overview.top_scorer?.goals} goals`,color:"gold"},
    {label:"Best Defender",value:overview.top_defender?.player?.replace("Player_","#"),sub:`${overview.top_defender?.tackles_won} tackles`,color:"blue"},
    {label:"Best Midfielder",value:overview.top_midfielder?.player?.replace("Player_","#"),sub:`${overview.top_midfielder?.pass_completion_pct?.toFixed(1)}% pass acc`,color:"red"},
  ];
  const posColors={Forward:"var(--red)",Midfielder:"var(--blue)",Defender:"var(--accent)"};
  return(
    <div>
      <div className="grid-4" style={{marginTop:".5rem"}}>
        {kpis.map((k,i)=>(
          <div key={i} className={`card card-${k.color}`}>
            <h3>{k.label}</h3>
            <div className="big-num">{k.value}</div>
            {k.sub&&<div className="sub">{k.sub}</div>}
          </div>
        ))}
      </div>
      <div className="section-title">Squad Composition</div>
      <div className="grid-2">
        <div className="card">
          <h3>Position Breakdown</h3>
          <div style={{marginTop:"1rem",display:"flex",flexDirection:"column",gap:".75rem"}}>
            {Object.entries(overview.positions||{}).map(([pos,cnt])=>(
              <div key={pos}>
                <div style={{display:"flex",justifyContent:"space-between",marginBottom:".3rem"}}>
                  <span style={{fontSize:".85rem"}}>{pos}</span>
                  <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:posColors[pos]}}>{cnt}</span>
                </div>
                <div style={{height:"6px",background:"var(--line)",borderRadius:"3px",overflow:"hidden"}}>
                  <div style={{height:"100%",width:`${cnt/overview.total_players*100}%`,background:posColors[pos],borderRadius:"3px",transition:"width 1s"}}/>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="card">
          <h3>Player Profiles (Clusters)</h3>
          <div style={{marginTop:"1rem",display:"flex",flexDirection:"column",gap:".5rem"}}>
            {Object.entries(overview.profiles||{}).map(([profile,cnt],i)=>{
              const colors=["var(--accent)","var(--gold)","var(--blue)","var(--red)","#b5179e"];
              return(
                <div key={profile} style={{display:"flex",alignItems:"center",gap:".75rem"}}>
                  <div style={{width:"10px",height:"10px",borderRadius:"50%",background:colors[i],flexShrink:0}}/>
                  <span style={{fontSize:".82rem",flex:1}}>{profile}</span>
                  <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:colors[i]}}>{cnt}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Scouting ──
function Scouting(){
  const [position,setPosition]=useState("Forward");
  const [results,setResults]=useState([]);
  const [loading,setLoading]=useState(false);
  const scout=useCallback(()=>{
    setLoading(true);
    fetchJSON(`${API}/api/scout/${position}?top_n=8`)
      .then(d=>{setResults(d.recommendations||[]);setLoading(false);})
      .catch(()=>setLoading(false));
  },[position]);
  useEffect(()=>{scout();},[scout]);
  const metricsByPos={
    Forward:[{k:"goals",l:"Goals"},{k:"shots_on_target",l:"SOT"},{k:"dribbles_completed",l:"Dribbles"}],
    Midfielder:[{k:"pass_completion_pct",l:"Pass %"},{k:"key_passes",l:"Key Passes"},{k:"ball_recoveries",l:"Recoveries"}],
    Defender:[{k:"tackles_won",l:"Tackles Won"},{k:"interceptions",l:"Interceptions"},{k:"ball_recoveries",l:"Recoveries"}],
  };
  const metrics=metricsByPos[position]||[];
  return(
    <div>
      <div style={{display:"flex",gap:"1rem",alignItems:"center",marginBottom:"1.25rem",flexWrap:"wrap"}}>
        <div className="section-title" style={{margin:0}}>Scouting Board</div>
        {["Forward","Midfielder","Defender"].map(p=>(
          <button key={p} className={`btn ${position===p?"btn-primary":"btn-secondary"}`} onClick={()=>setPosition(p)}>{p}</button>
        ))}
      </div>
      {loading?<div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div>:
      results.length===0?<div className="empty-state">No data</div>:
      <div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>#</th><th>Player</th><th>Profile</th><th>Score</th>{metrics.map(m=><th key={m.k}>{m.l}</th>)}</tr></thead>
            <tbody>
              {results.map((p,i)=>(
                <tr key={p.player}>
                  <td><span className={`rank rank-${i+1}`}>{i+1}</span></td>
                  <td><strong>{p.player}</strong></td>
                  <td><span className="tag tag-blue">{p.player_profile||"—"}</span></td>
                  <td>
                    <div className="score-bar">
                      <div className="bar" style={{width:`${p.scouting_score}%`,maxWidth:"80px"}}/>
                      <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:"var(--accent)"}}>{p.scouting_score}</span>
                    </div>
                  </td>
                  {metrics.map(m=><td key={m.k} style={{fontFamily:"Barlow Condensed",fontWeight:600}}>{typeof p[m.k]==="number"?(Number.isInteger(p[m.k])?p[m.k]:p[m.k].toFixed(1)):"—"}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="section-title" style={{marginTop:"1.5rem"}}>Player Cards</div>
        <div className="grid-3">
          {results.slice(0,6).map(p=>(
            <div key={p.player} className="player-card">
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
                <div><div className="name">{p.player}</div><div className="pos">{p.position}</div></div>
                <div style={{fontFamily:"Barlow Condensed",fontSize:"1.5rem",fontWeight:800,color:"var(--accent)"}}>{p.scouting_score}</div>
              </div>
              <div className="stat-grid">
                {metrics.map(m=>(
                  <div key={m.k} className="stat-item">
                    <div className="s-val">{typeof p[m.k]==="number"?(Number.isInteger(p[m.k])?p[m.k]:p[m.k].toFixed(1)):"—"}</div>
                    <div className="s-lbl">{m.l}</div>
                  </div>
                ))}
              </div>
              <div style={{marginTop:".75rem"}}><span className="tag tag-green">{p.player_profile||"—"}</span></div>
            </div>
          ))}
        </div>
      </div>}
    </div>
  );
}

// ── Action Predictor ──
function ActionPredictor(){
  const [params,setParams]=useState({location_x:80,location_y:40,distance_to_goal:25,under_pressure:0,minute:45});
  const [result,setResult]=useState(null);
  const [loading,setLoading]=useState(false);
  const predict=()=>{
    setLoading(true);
    fetchJSON(`${API}/api/predict/action`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(params)})
      .then(d=>{setResult(d);setLoading(false);}).catch(()=>setLoading(false));
  };
  useEffect(()=>{predict();},[]);
  const set=(k,v)=>setParams(p=>({...p,[k]:v}));
  const actionColors={pass:"#4cc9f0",carry:"#7209b7",dribble:"#ffd166",shot:"#ef4565",tackle:"#00e5a0",interception:"#06d6a0",clearance:"#118ab2",ball_receipt:"#8338ec",pressure:"#fb5607",cross:"#ffbe0b"};
  return(
    <div className="grid-2">
      <div className="card">
        <div className="section-title" style={{margin:"0 0 1rem"}}>Pitch Position</div>
        <div style={{background:"#1a3a1a",borderRadius:"8px",border:"2px solid #2d5a2d",aspectRatio:"120/80",position:"relative",marginBottom:"1.25rem",cursor:"crosshair"}}
             onClick={e=>{
               const r=e.currentTarget.getBoundingClientRect();
               const x=((e.clientX-r.left)/r.width)*120;
               const y=((e.clientY-r.top)/r.height)*80;
               const dist=Math.sqrt(Math.pow(120-x,2)+Math.pow(40-y,2));
               set("location_x",Math.round(x*10)/10);set("location_y",Math.round(y*10)/10);set("distance_to_goal",Math.round(dist*10)/10);
             }}>
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
        </div>
        <div className="form-group">
          <label>Minute <span className="range-val">{params.minute}'</span></label>
          <input type="range" min="1" max="90" value={params.minute} onChange={e=>set("minute",+e.target.value)}/>
        </div>
        <div className="form-group">
          <label>Under Pressure</label>
          <select value={params.under_pressure} onChange={e=>set("under_pressure",+e.target.value)}>
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:".5rem",marginBottom:".75rem",fontSize:".78rem",color:"var(--muted)"}}>
          <div>X: <b style={{color:"var(--text)"}}>{params.location_x}</b></div>
          <div>Y: <b style={{color:"var(--text)"}}>{params.location_y}</b></div>
          <div>Dist: <b style={{color:"var(--text)"}}>{params.distance_to_goal}m</b></div>
        </div>
        <button className="btn btn-primary" onClick={predict} style={{width:"100%"}}>{loading?"Predicting…":"Predict Action"}</button>
      </div>
      <div className="card">
        <div className="section-title" style={{margin:"0 0 1rem"}}>Action Probabilities</div>
        {loading&&<div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div>}
        {result&&!loading&&(
          <div>
            <div style={{background:"var(--panel)",borderRadius:"8px",padding:".75rem",marginBottom:"1rem",display:"flex",alignItems:"center",gap:".75rem"}}>
              <div style={{fontSize:".75rem",color:"var(--muted)"}}>Top Predicted Action</div>
              <div style={{fontFamily:"Barlow Condensed",fontSize:"1.4rem",fontWeight:700,color:actionColors[result.top_action]||"var(--accent)",textTransform:"capitalize"}}>{result.top_action}</div>
            </div>
            <div className="prob-list">
              {Object.entries(result.action_probabilities||{}).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([action,prob])=>(
                <div key={action} className="prob-row">
                  <div className="prob-label">{action}</div>
                  <div className="prob-track"><div className="prob-fill" style={{width:`${prob*100}%`,background:actionColors[action]||"var(--accent)"}}/></div>
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
function XGModel(){
  const [params,setParams]=useState({location_x:105,location_y:34,distance_to_goal:15,under_pressure:0,shot_angle:0.5});
  const [result,setResult]=useState(null);
  const [history,setHistory]=useState([]);
  const [loading,setLoading]=useState(false);
  const predict=()=>{
    setLoading(true);
    fetchJSON(`${API}/api/predict/xg`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(params)})
      .then(d=>{setResult(d);setHistory(h=>[{...params,xG:d.xG,pct:d.goal_probability_pct},...h].slice(0,5));setLoading(false);})
      .catch(()=>setLoading(false));
  };
  useEffect(()=>{predict();},[]);
  const set=(k,v)=>setParams(p=>({...p,[k]:v}));
  const xg=result?.xG||0;
  const color=xg<0.1?"var(--blue)":xg<0.3?"var(--gold)":"var(--red)";
  return(
    <div className="grid-2">
      <div className="card">
        <div className="section-title" style={{margin:"0 0 1rem"}}>Shot Parameters</div>
        <div className="form-group">
          <label>Shot Distance (m) <span className="range-val">{params.distance_to_goal}m</span></label>
          <input type="range" min="1" max="40" value={params.distance_to_goal} onChange={e=>{const d=+e.target.value;set("distance_to_goal",d);set("location_x",120-d*0.8);}}/>
        </div>
        <div className="form-group">
          <label>Shot Angle (rad) <span className="range-val">{params.shot_angle.toFixed(2)}</span></label>
          <input type="range" min="0" max="1.5" step="0.05" value={params.shot_angle} onChange={e=>set("shot_angle",+e.target.value)}/>
        </div>
        <div className="form-group">
          <label>Under Pressure</label>
          <select value={params.under_pressure} onChange={e=>set("under_pressure",+e.target.value)}>
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <button className="btn btn-primary" onClick={predict} style={{width:"100%",marginTop:".5rem"}}>{loading?"Calculating…":"Calculate xG"}</button>
        {history.length>0&&(
          <div style={{marginTop:"1.25rem"}}>
            <div style={{fontSize:".72rem",color:"var(--muted)",marginBottom:".5rem",fontFamily:"Barlow Condensed",letterSpacing:"1px",textTransform:"uppercase"}}>Shot History</div>
            {history.map((h,i)=>(
              <div key={i} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:".4rem 0",borderBottom:"1px solid var(--line)",fontSize:".8rem"}}>
                <span style={{color:"var(--muted)"}}>Dist: {h.distance_to_goal}m</span>
                <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:h.xG<0.1?"var(--blue)":h.xG<0.3?"var(--gold)":"var(--red)"}}>{h.pct}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="card" style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
        <div style={{fontSize:".75rem",color:"var(--muted)",letterSpacing:"2px",textTransform:"uppercase",marginBottom:"1rem"}}>Expected Goal Value</div>
        <svg width="200" height="110" viewBox="0 0 200 110">
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="var(--line)" strokeWidth="12" strokeLinecap="round"/>
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke={color} strokeWidth="12" strokeLinecap="round" strokeDasharray="251" strokeDashoffset={251*(1-Math.min(xg*2,1))} style={{transition:"stroke-dashoffset .8s ease,stroke .4s"}}/>
          <text x="100" y="95" textAnchor="middle" fontFamily="Barlow Condensed" fontWeight="800" fontSize="28" fill={color}>{(xg*100).toFixed(1)}%</text>
          <text x="100" y="110" textAnchor="middle" fontFamily="Barlow Condensed" fontSize="11" fill="#6b8cad" letterSpacing="1">GOAL PROBABILITY</text>
        </svg>
        <div style={{marginTop:"1.5rem",textAlign:"center"}}>
          <div style={{fontFamily:"Barlow Condensed",fontSize:"4rem",fontWeight:"800",color,lineHeight:1}}>{xg.toFixed(3)}</div>
          <div style={{fontSize:".75rem",color:"var(--muted)",letterSpacing:"2px",textTransform:"uppercase",marginTop:".25rem"}}>xG Score</div>
        </div>
      </div>
    </div>
  );
}

// ── All Players ──
function AllPlayers(){
  const [players,setPlayers]=useState([]);
  const [position,setPosition]=useState("all");
  const [search,setSearch]=useState("");
  const [loading,setLoading]=useState(true);
  useEffect(()=>{
    setLoading(true);
    const pos=position!=="all"?`?position=${position}&top=100`:"?top=100";
    fetchJSON(`${API}/api/players${pos}`).then(d=>{setPlayers(d);setLoading(false);}).catch(()=>setLoading(false));
  },[position]);
  const filtered=players.filter(p=>p.player.toLowerCase().includes(search.toLowerCase()));
  return(
    <div>
      <div style={{display:"flex",gap:".75rem",marginBottom:"1.25rem",flexWrap:"wrap",alignItems:"center"}}>
        <input type="text" placeholder="Search player…" value={search} onChange={e=>setSearch(e.target.value)} style={{maxWidth:"220px"}}/>
        {["all","Forward","Midfielder","Defender"].map(p=>(
          <button key={p} className={`btn ${position===p?"btn-primary":"btn-secondary"}`} onClick={()=>setPosition(p)}>{p==="all"?"All":p}</button>
        ))}
      </div>
      {loading?<div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div>:
      <div className="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Player</th><th>Pos</th><th>Profile</th><th>Score</th><th>Goals</th><th>SOT</th><th>Pass%</th><th>Key Pass</th><th>Tackles W</th><th>Intercept</th><th>Dribbles</th></tr></thead>
          <tbody>
            {filtered.map((p,i)=>(
              <tr key={p.player}>
                <td className="rank">{i+1}</td>
                <td><strong>{p.player}</strong></td>
                <td><span className={`tag ${p.position==="Forward"?"tag-red":p.position==="Midfielder"?"tag-blue":"tag-green"}`}>{p.position?.slice(0,3)}</span></td>
                <td><span className="tag tag-gold" style={{fontSize:".65rem"}}>{p.player_profile||"—"}</span></td>
                <td><span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:"var(--accent)"}}>{p.scouting_score}</span></td>
                <td>{p.goals}</td><td>{p.shots_on_target}</td>
                <td>{p.pass_completion_pct?.toFixed(1)}%</td>
                <td>{p.key_passes}</td><td>{p.tackles_won}</td><td>{p.interceptions}</td><td>{p.dribbles_completed}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// ADMIN DASHBOARD
// ══════════════════════════════════════════════════════════════

const EMPTY_FORM={player:"",position:"Forward",team:"",nationality:"",age:"",goals:"",shots_on_target:"",pass_completion_pct:"",key_passes:"",dribbles_completed:"",tackles_won:"",interceptions:"",clearances:"",ball_recoveries:"",total_actions:"",passes:""};

function AdminDashboard({showToast}){
  const [players,setPlayers]=useState([]);
  const [loading,setLoading]=useState(true);
  const [search,setSearch]=useState("");
  const [filterPos,setFilterPos]=useState("");
  const [showModal,setShowModal]=useState(false);
  const [editPlayer,setEditPlayer]=useState(null);  // null = add mode
  const [form,setForm]=useState(EMPTY_FORM);
  const [delConfirm,setDelConfirm]=useState(null);
  const [saving,setSaving]=useState(false);

  const load=useCallback(()=>{
    setLoading(true);
    const q=new URLSearchParams();
    if(search) q.set("search",search);
    if(filterPos) q.set("position",filterPos);
    fetchJSON(`${API}/api/admin/players?${q}`).then(d=>{setPlayers(d);setLoading(false);}).catch(()=>setLoading(false));
  },[search,filterPos]);

  useEffect(()=>{load();},[load]);

  const openAdd=()=>{setForm(EMPTY_FORM);setEditPlayer(null);setShowModal(true);};
  const openEdit=(p)=>{
    setForm({
      player:p.player,position:p.position,team:p.team||"",nationality:p.nationality||"",
      age:p.age||"",goals:p.goals||"",shots_on_target:p.shots_on_target||"",
      pass_completion_pct:p.pass_completion_pct||"",key_passes:p.key_passes||"",
      dribbles_completed:p.dribbles_completed||"",tackles_won:p.tackles_won||"",
      interceptions:p.interceptions||"",clearances:p.clearances||"",
      ball_recoveries:p.ball_recoveries||"",total_actions:p.total_actions||"",passes:p.passes||""
    });
    setEditPlayer(p.player);
    setShowModal(true);
  };

  const savePlayer=async()=>{
    if(!form.player.trim()){showToast("Player name is required","error");return;}
    if(!["Forward","Midfielder","Defender"].includes(form.position)){showToast("Invalid position","error");return;}
    setSaving(true);
    try{
      const body={...form};
      ["age","goals","shots_on_target","pass_completion_pct","key_passes","dribbles_completed",
       "tackles_won","interceptions","clearances","ball_recoveries","total_actions","passes"]
        .forEach(k=>{body[k]=body[k]===""?0:Number(body[k]);});
      if(editPlayer){
        const {player,...upd}=body;
        await fetchJSON(`${API}/api/admin/players/${encodeURIComponent(editPlayer)}`,
          {method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(upd)});
        showToast(`${editPlayer} updated successfully`);
      }else{
        await fetchJSON(`${API}/api/admin/players`,
          {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
        showToast(`${form.player} added to system`);
      }
      setShowModal(false);load();
    }catch(e){showToast(e.message||"Failed to save","error");}
    setSaving(false);
  };

  const deletePlayer=async(name)=>{
    try{
      await fetchJSON(`${API}/api/admin/players/${encodeURIComponent(name)}`,{method:"DELETE"});
      showToast(`${name} removed from system`);
      setDelConfirm(null);load();
    }catch(e){showToast(e.message||"Failed to delete","error");}
  };

  const setF=(k,v)=>setForm(f=>({...f,[k]:v}));

  const posColors={Forward:"tag-red",Midfielder:"tag-blue",Defender:"tag-green"};

  return(
    <div>
      {/* Header bar */}
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1.25rem",flexWrap:"wrap",gap:"1rem"}}>
        <div className="section-title" style={{margin:0}}>
          <span style={{color:"var(--admin)"}}>⚙</span> Admin — Player Management
        </div>
        <button className="btn btn-primary" onClick={openAdd}>+ Add Player</button>
      </div>

      {/* Stats strip */}
      <div className="grid-4" style={{marginBottom:"1.25rem"}}>
        {[
          {label:"Total Players",val:players.length,color:"admin"},
          {label:"Forwards",val:players.filter(p=>p.position==="Forward").length,color:"red"},
          {label:"Midfielders",val:players.filter(p=>p.position==="Midfielder").length,color:"blue"},
          {label:"Defenders",val:players.filter(p=>p.position==="Defender").length,color:"accent"},
        ].map((s,i)=>(
          <div key={i} className={`card card-${s.color}`}>
            <h3>{s.label}</h3>
            <div className="big-num" style={{fontSize:"2.2rem"}}>{s.val}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div style={{display:"flex",gap:".75rem",marginBottom:"1rem",flexWrap:"wrap",alignItems:"center"}}>
        <input type="text" placeholder="Search by name…" value={search} onChange={e=>setSearch(e.target.value)} style={{maxWidth:"220px"}}/>
        <select value={filterPos} onChange={e=>setFilterPos(e.target.value)} style={{maxWidth:"180px"}}>
          <option value="">All Positions</option>
          <option value="Forward">Forward</option>
          <option value="Midfielder">Midfielder</option>
          <option value="Defender">Defender</option>
        </select>
        <button className="btn btn-secondary" onClick={load}>Refresh</button>
      </div>

      {/* Table */}
      {loading?<div className="loading"><div className="dot"/><div className="dot"/><div className="dot"/></div>:
      players.length===0?<div className="empty-state">No players found</div>:
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Player</th><th>Pos</th><th>Team</th><th>Nationality</th><th>Age</th>
              <th>Score</th><th>Profile</th><th>Goals</th><th>Pass%</th><th>Tackles</th>
              <th style={{textAlign:"center"}}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p,i)=>(
              <tr key={p.player}>
                <td style={{color:"var(--muted)",fontFamily:"Barlow Condensed"}}>{i+1}</td>
                <td><strong>{p.player}</strong></td>
                <td><span className={`tag ${posColors[p.position]||"tag-green"}`}>{p.position?.slice(0,3)}</span></td>
                <td style={{color:"var(--muted)",fontSize:".8rem"}}>{p.team||"—"}</td>
                <td style={{color:"var(--muted)",fontSize:".8rem"}}>{p.nationality||"—"}</td>
                <td style={{color:"var(--muted)"}}>{p.age||"—"}</td>
                <td><span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:"var(--accent)"}}>{p.scouting_score}</span></td>
                <td><span className="tag tag-admin" style={{fontSize:".65rem"}}>{p.player_profile||"—"}</span></td>
                <td>{p.goals}</td>
                <td>{typeof p.pass_completion_pct==="number"?p.pass_completion_pct.toFixed(1)+"%":"—"}</td>
                <td>{p.tackles_won}</td>
                <td>
                  <div className="action-btns">
                    <button className="btn btn-edit btn-sm" onClick={()=>openEdit(p)}>Edit</button>
                    <button className="btn btn-danger btn-sm" onClick={()=>setDelConfirm(p.player)}>Remove</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>}

      {/* ── Add / Edit Modal ── */}
      {showModal&&(
        <div className="modal-bg" onClick={e=>{if(e.target===e.currentTarget)setShowModal(false);}}>
          <div className="modal">
            <div className="modal-header">
              <div className="modal-title" style={{color:editPlayer?"var(--admin)":"var(--accent)"}}>{editPlayer?"Edit Player":"Add New Player"}</div>
              <button className="close-btn" onClick={()=>setShowModal(false)}>×</button>
            </div>

            {/* Basic Info */}
            <div style={{fontSize:".7rem",color:"var(--muted)",letterSpacing:"1.5px",textTransform:"uppercase",marginBottom:".75rem",paddingBottom:".4rem",borderBottom:"1px solid var(--line)"}}>Basic Information</div>
            <div className="grid-2">
              <div className="form-group">
                <label>Full Name *</label>
                <input type="text" value={form.player} onChange={e=>setF("player",e.target.value)} placeholder="e.g. Lionel Messi" disabled={!!editPlayer}/>
              </div>
              <div className="form-group">
                <label>Position *</label>
                <select value={form.position} onChange={e=>setF("position",e.target.value)}>
                  <option>Forward</option><option>Midfielder</option><option>Defender</option>
                </select>
              </div>
              <div className="form-group">
                <label>Team / Club</label>
                <input type="text" value={form.team} onChange={e=>setF("team",e.target.value)} placeholder="e.g. FC Barcelona"/>
              </div>
              <div className="form-group">
                <label>Nationality</label>
                <input type="text" value={form.nationality} onChange={e=>setF("nationality",e.target.value)} placeholder="e.g. Argentina"/>
              </div>
              <div className="form-group">
                <label>Age</label>
                <input type="number" value={form.age} onChange={e=>setF("age",e.target.value)} placeholder="e.g. 27" min="15" max="50"/>
              </div>
            </div>

            {/* Attacking stats */}
            <div style={{fontSize:".7rem",color:"var(--red)",letterSpacing:"1.5px",textTransform:"uppercase",margin:"1rem 0 .75rem",paddingBottom:".4rem",borderBottom:"1px solid var(--line)"}}>⚽ Attacking Stats</div>
            <div className="grid-2">
              {[{k:"goals",l:"Goals",ph:"0"},{k:"shots_on_target",l:"Shots on Target",ph:"0"},{k:"dribbles_completed",l:"Dribbles Completed",ph:"0"}].map(f=>(
                <div key={f.k} className="form-group">
                  <label>{f.l}</label>
                  <input type="number" value={form[f.k]} onChange={e=>setF(f.k,e.target.value)} placeholder={f.ph} min="0"/>
                </div>
              ))}
            </div>

            {/* Midfield stats */}
            <div style={{fontSize:".7rem",color:"var(--blue)",letterSpacing:"1.5px",textTransform:"uppercase",margin:"1rem 0 .75rem",paddingBottom:".4rem",borderBottom:"1px solid var(--line)"}}>🔵 Midfield Stats</div>
            <div className="grid-2">
              {[{k:"passes",l:"Total Passes",ph:"0"},{k:"pass_completion_pct",l:"Pass Completion %",ph:"75.0"},{k:"key_passes",l:"Key Passes",ph:"0"}].map(f=>(
                <div key={f.k} className="form-group">
                  <label>{f.l}</label>
                  <input type="number" value={form[f.k]} onChange={e=>setF(f.k,e.target.value)} placeholder={f.ph} min="0" step={f.k==="pass_completion_pct"?"0.1":"1"}/>
                </div>
              ))}
            </div>

            {/* Defensive stats */}
            <div style={{fontSize:".7rem",color:"var(--accent)",letterSpacing:"1.5px",textTransform:"uppercase",margin:"1rem 0 .75rem",paddingBottom:".4rem",borderBottom:"1px solid var(--line)"}}>🛡 Defensive Stats</div>
            <div className="grid-2">
              {[{k:"tackles_won",l:"Tackles Won",ph:"0"},{k:"interceptions",l:"Interceptions",ph:"0"},{k:"clearances",l:"Clearances",ph:"0"},{k:"ball_recoveries",l:"Ball Recoveries",ph:"0"}].map(f=>(
                <div key={f.k} className="form-group">
                  <label>{f.l}</label>
                  <input type="number" value={form[f.k]} onChange={e=>setF(f.k,e.target.value)} placeholder={f.ph} min="0"/>
                </div>
              ))}
            </div>

            <div style={{display:"flex",gap:".75rem",justifyContent:"flex-end",marginTop:"1.25rem"}}>
              <button className="btn btn-secondary" onClick={()=>setShowModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={savePlayer} disabled={saving}>
                {saving?"Saving…":editPlayer?"Update Player":"Add Player"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Delete Confirm ── */}
      {delConfirm&&(
        <div className="modal-bg" onClick={e=>{if(e.target===e.currentTarget)setDelConfirm(null);}}>
          <div className="modal" style={{maxWidth:"420px",textAlign:"center"}}>
            <div style={{fontSize:"2.5rem",marginBottom:".75rem"}}>⚠️</div>
            <div className="modal-title" style={{color:"var(--red)",marginBottom:".75rem"}}>Remove Player?</div>
            <p style={{color:"var(--muted)",fontSize:".9rem",marginBottom:"1.5rem"}}>
              Are you sure you want to remove <strong style={{color:"var(--text)"}}>{delConfirm}</strong> from the system? This cannot be undone.
            </p>
            <div style={{display:"flex",gap:".75rem",justifyContent:"center"}}>
              <button className="btn btn-secondary" onClick={()=>setDelConfirm(null)}>Cancel</button>
              <button className="btn btn-danger" onClick={()=>deletePlayer(delConfirm)}>Yes, Remove</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════
// TRAINING STUDIO COMPONENT
// ══════════════════════════════════════════════════════════════

const TRAIN_EMPTY={player:"",position:"Forward",age:25,goals:0,shots_on_target:0,pass_completion_pct:75,key_passes:0,dribbles_completed:0,tackles_won:0,interceptions:0,clearances:0,ball_recoveries:0,matches_played:10,quality_label:50};

function TrainingStudio({showToast}){
  const [records,setRecords]=useState([]);
  const [form,setForm]=useState({...TRAIN_EMPTY});
  const [modelStatus,setModelStatus]=useState(null);
  const [modelType,setModelType]=useState("random_forest");
  const [training,setTraining]=useState(false);
  const [ratings,setRatings]=useState([]);
  const [ratingAll,setRatingAll]=useState(false);
  const [predictForm,setPredictForm]=useState({...TRAIN_EMPTY});
  const [liveRating,setLiveRating]=useState(null);
  const [activeSection,setActiveSection]=useState("data");  // data | train | rate | all

  const loadRecords=()=>{
    fetchJSON(`${API}/api/train/records`).then(d=>setRecords(d.records||[])).catch(()=>{});
  };
  const loadStatus=()=>{
    fetchJSON(`${API}/api/train/model-status`).then(setModelStatus).catch(()=>{});
  };
  useEffect(()=>{loadRecords();loadStatus();},[]);

  const setF=(k,v)=>setForm(f=>({...f,[k]:v}));
  const setPF=(k,v)=>setPredictForm(f=>({...f,[k]:v}));

  const addRecord=async()=>{
    if(!form.player.trim()){showToast("Player name required","error");return;}
    try{
      await fetchJSON(`${API}/api/train/record`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(form)});
      showToast(`${form.player} added to training set`);
      setForm({...TRAIN_EMPTY});
      loadRecords();
    }catch(e){showToast(e.message,"error");}
  };

  const removeRecord=async(name)=>{
    try{
      await fetchJSON(`${API}/api/train/record/${encodeURIComponent(name)}`,{method:"DELETE"});
      showToast(`${name} removed`);
      loadRecords();
    }catch(e){showToast(e.message,"error");}
  };

  const runTraining=async()=>{
    if(records.length<4){showToast("Need at least 4 training records","error");return;}
    setTraining(true);
    try{
      const res=await fetchJSON(`${API}/api/train/run`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_type:modelType})});
      showToast(`✓ Model trained! R²=${res.metrics.r2}`);
      loadStatus();
    }catch(e){showToast(e.message,"error");}
    setTraining(false);
  };

  const predictLive=async()=>{
    if(!modelStatus?.trained){showToast("Train model first","error");return;}
    try{
      const res=await fetchJSON(`${API}/api/train/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(predictForm)});
      setLiveRating(res);
    }catch(e){showToast(e.message,"error");}
  };

  const rateAllPlayers=async()=>{
    if(!modelStatus?.trained){showToast("Train model first","error");return;}
    setRatingAll(true);
    try{
      const res=await fetchJSON(`${API}/api/train/predict-all`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})});
      setRatings(res.ratings||[]);
      setActiveSection("all");
    }catch(e){showToast(e.message,"error");}
    setRatingAll(false);
  };

  const gradeColor={
    "World Class":"var(--gold)","Elite":"var(--accent)","Quality":"var(--blue)",
    "Decent":"var(--muted)","Developing":"var(--red)"
  };

  const modelLabels={"random_forest":"Random Forest","gradient_boost":"Gradient Boosting","ridge":"Ridge Regression"};

  const sections=[
    {id:"data",label:"1 · Training Data"},
    {id:"train",label:"2 · Train Model"},
    {id:"rate",label:"3 · Rate a Player"},
    {id:"all",label:"4 · Rate All Players"},
  ];

  return(
    <div>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1.25rem",flexWrap:"wrap",gap:"1rem"}}>
        <div className="section-title" style={{margin:0}}><span style={{color:"var(--admin)"}}>🧠</span> Training Studio</div>
        {modelStatus?.trained&&(
          <div style={{display:"flex",alignItems:"center",gap:".6rem",background:"rgba(0,229,160,.08)",border:"1px solid rgba(0,229,160,.25)",borderRadius:"8px",padding:".45rem .9rem"}}>
            <div style={{width:"8px",height:"8px",borderRadius:"50%",background:"var(--accent)"}}/>
            <span style={{fontSize:".78rem",color:"var(--accent)",fontFamily:"Barlow Condensed",letterSpacing:"1px"}}>
              MODEL ACTIVE · {modelLabels[modelStatus.model_type]} · R²={modelStatus.accuracy?.r2}
            </span>
          </div>
        )}
      </div>

      {/* Section tabs */}
      <div style={{display:"flex",gap:".5rem",marginBottom:"1.5rem",flexWrap:"wrap"}}>
        {sections.map(s=>(
          <button key={s.id} className={`btn ${activeSection===s.id?"btn-primary":"btn-secondary"}`}
                  onClick={()=>setActiveSection(s.id)} style={{fontSize:".8rem",padding:".5rem 1rem"}}>
            {s.label}
          </button>
        ))}
      </div>

      {/* ── SECTION 1: Training Data ── */}
      {activeSection==="data"&&(
        <div>
          <div className="grid-2" style={{gap:"1.5rem"}}>
            {/* Input form */}
            <div className="card card-admin">
              <div style={{fontFamily:"Barlow Condensed",fontSize:"1rem",fontWeight:700,letterSpacing:"2px",textTransform:"uppercase",color:"var(--admin)",marginBottom:"1rem"}}>
                Add Training Record
              </div>
              <div className="grid-2">
                <div className="form-group">
                  <label>Player Name *</label>
                  <input type="text" value={form.player} onChange={e=>setF("player",e.target.value)} placeholder="e.g. Lionel Messi"/>
                </div>
                <div className="form-group">
                  <label>Position</label>
                  <select value={form.position} onChange={e=>setF("position",e.target.value)}>
                    <option>Forward</option><option>Midfielder</option><option>Defender</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Age</label>
                  <input type="number" value={form.age} onChange={e=>setF("age",+e.target.value)} min="15" max="50"/>
                </div>
                <div className="form-group">
                  <label>Matches Played</label>
                  <input type="number" value={form.matches_played} onChange={e=>setF("matches_played",+e.target.value)} min="1"/>
                </div>
              </div>

              <div style={{fontSize:".7rem",color:"var(--red)",letterSpacing:"1.5px",textTransform:"uppercase",margin:".75rem 0 .5rem",paddingBottom:".3rem",borderBottom:"1px solid var(--line)"}}>⚽ Attacking</div>
              <div className="grid-2">
                {[{k:"goals",l:"Goals"},{k:"shots_on_target",l:"Shots on Target"},{k:"dribbles_completed",l:"Dribbles"}].map(f=>(
                  <div key={f.k} className="form-group">
                    <label>{f.l}</label>
                    <input type="number" value={form[f.k]} onChange={e=>setF(f.k,+e.target.value)} min="0" step="0.1"/>
                  </div>
                ))}
              </div>

              <div style={{fontSize:".7rem",color:"var(--blue)",letterSpacing:"1.5px",textTransform:"uppercase",margin:".75rem 0 .5rem",paddingBottom:".3rem",borderBottom:"1px solid var(--line)"}}>🔵 Midfield</div>
              <div className="grid-2">
                {[{k:"pass_completion_pct",l:"Pass Completion %",step:"0.1"},{k:"key_passes",l:"Key Passes",step:"1"}].map(f=>(
                  <div key={f.k} className="form-group">
                    <label>{f.l}</label>
                    <input type="number" value={form[f.k]} onChange={e=>setF(f.k,+e.target.value)} min="0" step={f.step||"1"}/>
                  </div>
                ))}
              </div>

              <div style={{fontSize:".7rem",color:"var(--accent)",letterSpacing:"1.5px",textTransform:"uppercase",margin:".75rem 0 .5rem",paddingBottom:".3rem",borderBottom:"1px solid var(--line)"}}>🛡 Defensive</div>
              <div className="grid-2">
                {[{k:"tackles_won",l:"Tackles Won"},{k:"interceptions",l:"Interceptions"},{k:"clearances",l:"Clearances"},{k:"ball_recoveries",l:"Recoveries"}].map(f=>(
                  <div key={f.k} className="form-group">
                    <label>{f.l}</label>
                    <input type="number" value={form[f.k]} onChange={e=>setF(f.k,+e.target.value)} min="0"/>
                  </div>
                ))}
              </div>

              {/* Quality label slider */}
              <div style={{background:"rgba(180,142,255,.08)",border:"1px solid rgba(180,142,255,.25)",borderRadius:"8px",padding:"1rem",marginTop:".75rem"}}>
                <label style={{color:"var(--admin)"}}>
                  ⭐ Quality Label (Training Target)
                  <span style={{float:"right",fontFamily:"Barlow Condensed",fontSize:"1.3rem",fontWeight:700,color:"var(--admin)"}}>{form.quality_label}/100</span>
                </label>
                <input type="range" min="0" max="100" value={form.quality_label} onChange={e=>setF("quality_label",+e.target.value)} style={{marginTop:".4rem",accentColor:"var(--admin)"}}/>
                <div style={{display:"flex",justifyContent:"space-between",fontSize:".65rem",color:"var(--muted)",marginTop:".3rem"}}>
                  <span>0 — Poor</span><span>50 — Decent</span><span>100 — World Class</span>
                </div>
              </div>

              <button className="btn btn-primary" onClick={addRecord} style={{width:"100%",marginTop:"1rem",background:"var(--admin)",color:"#0a1628"}}>
                + Add to Training Set
              </button>
            </div>

            {/* Records table */}
            <div>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:".75rem"}}>
                <div style={{fontFamily:"Barlow Condensed",fontSize:"1rem",fontWeight:700,letterSpacing:"2px",textTransform:"uppercase",color:"var(--text)"}}>
                  Training Records
                  <span style={{marginLeft:".6rem",background:"rgba(180,142,255,.2)",color:"var(--admin)",borderRadius:"12px",padding:".15rem .6rem",fontSize:".75rem"}}>{records.length}</span>
                </div>
              </div>
              {records.length===0?(
                <div style={{background:"var(--card)",border:"1px solid var(--line)",borderRadius:"12px",padding:"3rem",textAlign:"center",color:"var(--muted)"}}>
                  <div style={{fontSize:"2rem",marginBottom:".5rem"}}>📋</div>
                  Add at least 4 players to train the model
                </div>
              ):(
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Player</th><th>Pos</th><th>Age</th><th>Goals</th><th>Pass%</th><th>Tackles</th><th>⭐ Label</th><th></th></tr></thead>
                    <tbody>
                      {records.map((r,i)=>(
                        <tr key={i}>
                          <td><strong>{r.player}</strong></td>
                          <td><span className={`tag ${r.position==="Forward"?"tag-red":r.position==="Midfielder"?"tag-blue":"tag-green"}`}>{r.position?.slice(0,3)}</span></td>
                          <td>{r.age}</td>
                          <td>{r.goals}</td>
                          <td>{r.pass_completion_pct}%</td>
                          <td>{r.tackles_won}</td>
                          <td>
                            <span style={{fontFamily:"Barlow Condensed",fontWeight:700,
                              color:r.quality_label>=75?"var(--gold)":r.quality_label>=50?"var(--accent)":"var(--muted)"}}>
                              {r.quality_label}
                            </span>
                          </td>
                          <td><button className="btn btn-danger btn-sm" onClick={()=>removeRecord(r.player)}>✕</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {records.length>0&&records.length<4&&(
                <div style={{marginTop:".75rem",padding:".75rem",background:"rgba(255,209,102,.08)",border:"1px solid rgba(255,209,102,.25)",borderRadius:"8px",fontSize:".8rem",color:"var(--gold)"}}>
                  ⚠ Add {4-records.length} more record{4-records.length!==1?"s":""} to enable training
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── SECTION 2: Train Model ── */}
      {activeSection==="train"&&(
        <div className="grid-2" style={{gap:"1.5rem"}}>
          <div className="card card-admin">
            <div style={{fontFamily:"Barlow Condensed",fontSize:"1rem",fontWeight:700,letterSpacing:"2px",textTransform:"uppercase",color:"var(--admin)",marginBottom:"1rem"}}>
              Configure & Train
            </div>

            <div className="form-group">
              <label>Algorithm</label>
              <select value={modelType} onChange={e=>setModelType(e.target.value)}>
                <option value="random_forest">🌲 Random Forest</option>
                <option value="gradient_boost">🚀 Gradient Boosting</option>
                <option value="ridge">📐 Ridge Regression</option>
              </select>
            </div>

            <div style={{background:"var(--panel)",borderRadius:"8px",padding:"1rem",marginBottom:"1rem",fontSize:".82rem"}}>
              {modelType==="random_forest"&&<div><div style={{color:"var(--accent)",fontWeight:600,marginBottom:".4rem"}}>Random Forest</div><div style={{color:"var(--muted)"}}>Ensemble of 100 decision trees. Best for general use — handles non-linear relationships and is robust to outliers.</div></div>}
              {modelType==="gradient_boost"&&<div><div style={{color:"var(--gold)",fontWeight:600,marginBottom:".4rem"}}>Gradient Boosting</div><div style={{color:"var(--muted)"}}>Sequentially builds trees to correct previous errors. Highest accuracy on structured data but slower to train.</div></div>}
              {modelType==="ridge"&&<div><div style={{color:"var(--blue)",fontWeight:600,marginBottom:".4rem"}}>Ridge Regression</div><div style={{color:"var(--muted)"}}>Regularised linear model. Fastest to train and easiest to interpret — good when data is limited.</div></div>}
            </div>

            <div style={{background:"var(--panel)",borderRadius:"8px",padding:"1rem",marginBottom:"1rem"}}>
              <div style={{fontSize:".72rem",color:"var(--muted)",textTransform:"uppercase",letterSpacing:"1px",marginBottom:".6rem"}}>Training Features</div>
              <div style={{display:"flex",flexWrap:"wrap",gap:".3rem"}}>
                {["goals","shots_on_target","pass_completion_pct","key_passes","dribbles_completed","tackles_won","interceptions","clearances","ball_recoveries","matches_played","age"].map(f=>(
                  <span key={f} className="tag tag-admin" style={{fontSize:".65rem"}}>{f}</span>
                ))}
              </div>
            </div>

            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:".75rem",background:"var(--panel)",borderRadius:"8px",marginBottom:"1rem",fontSize:".82rem"}}>
              <span style={{color:"var(--muted)"}}>Training records</span>
              <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:records.length>=4?"var(--accent)":"var(--red)"}}>{records.length} / 4 min</span>
            </div>

            <button className="btn btn-primary" onClick={runTraining} disabled={training||records.length<4}
                    style={{width:"100%",background:"var(--admin)",color:"#0a1628",opacity:records.length<4?.5:1}}>
              {training?"⏳ Training…":"🚀 Train Quality Model"}
            </button>
          </div>

          {/* Model status card */}
          <div className="card" style={{borderLeft:"3px solid var(--admin)"}}>
            <div style={{fontFamily:"Barlow Condensed",fontSize:"1rem",fontWeight:700,letterSpacing:"2px",textTransform:"uppercase",color:"var(--admin)",marginBottom:"1rem"}}>
              Model Status
            </div>
            {!modelStatus?.trained?(
              <div style={{textAlign:"center",padding:"2rem",color:"var(--muted)"}}>
                <div style={{fontSize:"2.5rem",marginBottom:".75rem"}}>🤖</div>
                <div>No model trained yet.</div>
                <div style={{fontSize:".78rem",marginTop:".3rem"}}>Add data and click Train.</div>
              </div>
            ):(
              <div>
                <div style={{display:"flex",alignItems:"center",gap:".75rem",marginBottom:"1.25rem",padding:"1rem",background:"rgba(0,229,160,.06)",borderRadius:"8px",border:"1px solid rgba(0,229,160,.2)"}}>
                  <div style={{fontSize:"2rem"}}>✅</div>
                  <div>
                    <div style={{fontFamily:"Barlow Condensed",fontSize:"1rem",fontWeight:700,color:"var(--accent)"}}>Model Trained</div>
                    <div style={{fontSize:".78rem",color:"var(--muted)"}}>{modelLabels[modelStatus.model_type]} · {modelStatus.trained_on} records</div>
                  </div>
                </div>
                {[
                  {label:"Algorithm",val:modelLabels[modelStatus.model_type],color:"var(--admin)"},
                  {label:"Trained On",val:`${modelStatus.trained_on} players`,color:"var(--text)"},
                  {label:"R² Score",val:modelStatus.accuracy?.r2,color:modelStatus.accuracy?.r2>0.7?"var(--accent)":"var(--gold)",hint:"(1.0 = perfect fit)"},
                  {label:"RMSE",val:modelStatus.accuracy?.rmse,color:"var(--text)",hint:"(lower is better)"},
                ].map(m=>(
                  <div key={m.label} style={{display:"flex",justifyContent:"space-between",padding:".6rem 0",borderBottom:"1px solid var(--line)",alignItems:"center"}}>
                    <span style={{fontSize:".82rem",color:"var(--muted)"}}>{m.label} <span style={{fontSize:".7rem"}}>{m.hint||""}</span></span>
                    <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:m.color}}>{m.val}</span>
                  </div>
                ))}
                <button className="btn btn-secondary" onClick={rateAllPlayers} disabled={ratingAll}
                        style={{width:"100%",marginTop:"1.25rem"}}>
                  {ratingAll?"Rating…":"Rate All System Players →"}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── SECTION 3: Rate a Player ── */}
      {activeSection==="rate"&&(
        <div className="grid-2" style={{gap:"1.5rem"}}>
          <div className="card card-admin">
            <div style={{fontFamily:"Barlow Condensed",fontSize:"1rem",fontWeight:700,letterSpacing:"2px",textTransform:"uppercase",color:"var(--admin)",marginBottom:"1rem"}}>
              Player to Rate
            </div>
            <div className="grid-2">
              <div className="form-group">
                <label>Player Name</label>
                <input type="text" value={predictForm.player} onChange={e=>setPF("player",e.target.value)} placeholder="e.g. Harry Kane"/>
              </div>
              <div className="form-group">
                <label>Position</label>
                <select value={predictForm.position} onChange={e=>setPF("position",e.target.value)}>
                  <option>Forward</option><option>Midfielder</option><option>Defender</option>
                </select>
              </div>
              <div className="form-group">
                <label>Age</label>
                <input type="number" value={predictForm.age} onChange={e=>setPF("age",+e.target.value)} min="15" max="50"/>
              </div>
              <div className="form-group">
                <label>Matches Played</label>
                <input type="number" value={predictForm.matches_played} onChange={e=>setPF("matches_played",+e.target.value)} min="1"/>
              </div>
            </div>
            <div className="grid-2">
              {[{k:"goals",l:"Goals"},{k:"shots_on_target",l:"Shots on Target"},{k:"pass_completion_pct",l:"Pass%",step:"0.1"},{k:"key_passes",l:"Key Passes"},{k:"dribbles_completed",l:"Dribbles"},{k:"tackles_won",l:"Tackles Won"},{k:"interceptions",l:"Interceptions"},{k:"clearances",l:"Clearances"},{k:"ball_recoveries",l:"Recoveries"}].map(f=>(
                <div key={f.k} className="form-group">
                  <label>{f.l}</label>
                  <input type="number" value={predictForm[f.k]} onChange={e=>setPF(f.k,+e.target.value)} min="0" step={f.step||"1"}/>
                </div>
              ))}
            </div>
            <button className="btn btn-primary" onClick={predictLive}
                    style={{width:"100%",background:"var(--admin)",color:"#0a1628",opacity:modelStatus?.trained?1:.5}}
                    disabled={!modelStatus?.trained}>
              {modelStatus?.trained?"🎯 Predict Quality Rating":"Train model first"}
            </button>
          </div>

          {/* Result */}
          <div className="card" style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",borderLeft:"3px solid var(--admin)"}}>
            {!liveRating?(
              <div style={{textAlign:"center",color:"var(--muted)"}}>
                <div style={{fontSize:"3rem",marginBottom:".75rem"}}>🎯</div>
                <div>Fill in the stats and click Predict to get a quality rating</div>
                {!modelStatus?.trained&&<div style={{marginTop:".75rem",fontSize:".78rem",color:"var(--red)"}}>⚠ Train the model first in section 2</div>}
              </div>
            ):(
              <div style={{textAlign:"center",width:"100%"}}>
                <div style={{fontSize:".75rem",color:"var(--muted)",letterSpacing:"2px",textTransform:"uppercase",marginBottom:".75rem"}}>Quality Rating</div>
                <div style={{fontFamily:"Barlow Condensed",fontSize:"5rem",fontWeight:800,color:gradeColor[liveRating.grade]||"var(--text)",lineHeight:1}}>
                  {liveRating.quality_score}
                </div>
                <div style={{fontFamily:"Barlow Condensed",fontSize:"1.6rem",fontWeight:700,color:gradeColor[liveRating.grade],marginTop:".25rem",marginBottom:"1.5rem"}}>
                  {liveRating.grade}
                </div>
                {/* Grade legend */}
                <div style={{display:"flex",gap:".4rem",justifyContent:"center",flexWrap:"wrap"}}>
                  {[["World Class","var(--gold)","85+"],["Elite","var(--accent)","75+"],["Quality","var(--blue)","65+"],["Decent","var(--muted)","50+"],["Developing","var(--red)","<50"]].map(([g,c,r])=>(
                    <div key={g} style={{textAlign:"center",padding:".3rem .6rem",borderRadius:"6px",
                                        background:liveRating.grade===g?"rgba(255,255,255,.1)":"transparent",
                                        border:`1px solid ${liveRating.grade===g?c:"transparent"}`}}>
                      <div style={{fontSize:".65rem",color:c,fontWeight:600}}>{g}</div>
                      <div style={{fontSize:".6rem",color:"var(--muted)"}}>{r}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── SECTION 4: Rate All Players ── */}
      {activeSection==="all"&&(
        <div>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1rem"}}>
            <div style={{fontFamily:"Barlow Condensed",fontSize:"1rem",fontWeight:700,letterSpacing:"2px",textTransform:"uppercase"}}>
              All Player Quality Ratings
            </div>
            <button className="btn btn-secondary" onClick={rateAllPlayers} disabled={ratingAll||!modelStatus?.trained}>
              {ratingAll?"Rating…":"Re-Rate All"}
            </button>
          </div>
          {ratings.length===0?(
            <div style={{textAlign:"center",padding:"3rem",color:"var(--muted)"}}>
              <div style={{fontSize:"2.5rem",marginBottom:".75rem"}}>📊</div>
              {modelStatus?.trained?"Click Re-Rate All to generate ratings":"Train model first in Section 2"}
            </div>
          ):(
            <div className="table-wrap">
              <table>
                <thead><tr><th>#</th><th>Player</th><th>Pos</th><th>Quality Score</th><th>Grade</th><th>Scouting Score</th></tr></thead>
                <tbody>
                  {ratings.map((r,i)=>(
                    <tr key={r.player}>
                      <td><span className={`rank rank-${i+1}`}>{i+1}</span></td>
                      <td><strong>{r.player}</strong></td>
                      <td><span className={`tag ${r.position==="Forward"?"tag-red":r.position==="Midfielder"?"tag-blue":"tag-green"}`}>{r.position?.slice(0,3)}</span></td>
                      <td>
                        <div style={{display:"flex",alignItems:"center",gap:".6rem"}}>
                          <div style={{flex:1,height:"8px",background:"var(--line)",borderRadius:"4px",overflow:"hidden",maxWidth:"120px"}}>
                            <div style={{height:"100%",width:`${r.quality_score}%`,background:gradeColor[r.grade]||"var(--accent)",borderRadius:"4px",transition:"width .6s"}}/>
                          </div>
                          <span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:gradeColor[r.grade]||"var(--text)",minWidth:"35px"}}>{r.quality_score}</span>
                        </div>
                      </td>
                      <td><span className="tag" style={{background:`${gradeColor[r.grade]}22`,color:gradeColor[r.grade],border:`1px solid ${gradeColor[r.grade]}55`}}>{r.grade}</span></td>
                      <td><span style={{fontFamily:"Barlow Condensed",fontWeight:700,color:"var(--accent)"}}>{r.scouting_score}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

ReactDOM.render(<App/>,document.getElementById("root"));
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
