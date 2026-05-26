# ⚽ Football Intelligence System

A full-stack ML application for football analytics — predicting player actions, estimating goal probability (xG), clustering player profiles and powering a scouting engine.

---

## 🗂 What's Inside (`app.py`)

| Module | Description |
|--------|-------------|
| **DataPipeline** | Loads StatsBomb open data or generates synthetic match events |
| **FeatureEngineer** | Builds player stats and ML-ready feature matrices |
| **MLModels** | Action classifier (Random Forest), xG model (Gradient Boosting), Player clustering (K-Means) |
| **ScoutingEngine** | Position-specific weighted scoring + player recommendations |
| **FastAPI Backend** | REST API with `/api/players`, `/api/scout`, `/api/predict/action`, `/api/predict/xg` |
| **React Frontend** | Full dashboard served at `http://localhost:8000` |

---

## 🚀 Quick Start

### Windows
```bash
# 1. Setup (run once)
setup.bat

# 2. Start the app
run.bat
```

### Mac / Linux
```bash
# 1. Setup (run once)
chmod +x setup.sh run.sh
./setup.sh

# 2. Start the app
./run.sh
```

### Manual
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open → **http://localhost:8000**

---

## 🖥 Dashboard Tabs

| Tab | What it does |
|-----|-------------|
| **Dashboard** | Overview KPIs, squad composition, player profile clusters |
| **Scouting** | Top players ranked by position-specific metrics with player cards |
| **Action Predictor** | Click pitch to set position → model predicts most likely next action |
| **xG Model** | Adjust shot parameters → get expected goal probability with arc meter |
| **All Players** | Full filterable/searchable table of all players and their stats |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/players` | All players (filter by `?position=Forward`) |
| GET | `/api/scout/{position}` | Top scouted players by position |
| POST | `/api/predict/action` | Predict player action from pitch coords |
| POST | `/api/predict/xg` | Get expected goal value for a shot |
| GET | `/api/stats/overview` | Dashboard overview stats |
| GET | `/api/player/{name}` | Single player detail + strengths |
| GET | `/docs` | Interactive Swagger API docs |

---

## 🤖 ML Models

### Action Prediction (Random Forest)
- **Input:** location_x, location_y, distance_to_goal, under_pressure, minute
- **Output:** Probability distribution over 10 action types (pass, shot, dribble, etc.)

### Expected Goals / xG (Gradient Boosting)
- **Input:** Shot location, angle, distance, pressure
- **Output:** Goal probability 0–1

### Player Clustering (K-Means, k=5)
- **Profiles:** Defensive Anchor, Creative Playmaker, Goal Threat, Pressing Machine, Box-to-Box

### Scouting Scorer (Weighted Algorithm)
- **Forwards:** Shots on target (30%), Goals (40%), Dribbles completed (30%)
- **Midfielders:** Pass completion (35%), Key passes (40%), Ball recoveries (25%)
- **Defenders:** Tackles won (35%), Interceptions (30%), Recoveries (20%), Clearances (15%)

---

## 📦 Dependencies
- `fastapi` + `uvicorn` — backend server
- `scikit-learn` — ML models
- `pandas` + `numpy` — data processing
- `statsbombpy` — free football event data
- React (CDN) — frontend UI

---

## 📝 Notes
- If StatsBomb data fails to load, the system automatically falls back to realistic synthetic data — the app will still fully function.
- Trained model files are saved to `./models/` after first boot.
- The app boots and trains all models on startup (~5–15 seconds).
