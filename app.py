import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import gzip


with gzip.open("random_forest_model_compressed.joblib.gz", "rb") as f:
    loaded_rf_model = joblib.load(f)

# Load data and models
df = pd.read_csv('cleaned_featured_matches_without_draws.csv')
df['date'] = pd.to_datetime(df['date'])

loaded_lr_model = joblib.load('logistic_regression_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

cutoff_date = pd.to_datetime('2025-05-12')
manual_cols = ['ga', 'gf', 'xg', 'xga', 'poss', 'sh', 'sot', 'shot_dist', 'points']

# Set page config
st.set_page_config(page_title="Football Predictor", layout="centered")

# Initialize session state
if "selected_comp" not in st.session_state:
    st.session_state.selected_comp = None

# CSS Styling
st.markdown("""
    <style>
        .main, .stApp {
            background-color: black;
            color: white;
            font-family: 'Aptos', sans-serif;
        }
        .stButton>button {
            background-color: black;
            color: white;
            margin-top: 5px;
            border-radius: 8px;
            width: 100%;
            height: 80px;
            white-space: normal;
            word-wrap: break-word;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-weight: bold;
            transition-duration: 0.4s;
            border-color: auto;
        }
        .stButton>button:hover {
            color: #92D050 !important;
            border-color: #92D050 !important;
        }
        .stButton>button:active {
            color: black !important;
            background-color: #92D050 !important;
            border-color: #92D050 !important;
        }
        .competition-logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 70px;
            height: 70px;
        }
        .competition-container {
            text-align: center;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.image("logo.png", width=1920)
st.markdown('<h5 style="text-align: center;">Predicts Win/Loss using recent performance metrics.</h5>', unsafe_allow_html=True)

# --- COMPETITION SELECTION ---
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

if st.session_state.selected_comp is None:
    st.markdown('<h3 style="text-align: center;">CHOOSE A COMPETITION</h3>', unsafe_allow_html=True)

    competitions_dict = {
        "English Premier League": "epl-logo.png",
        "French Ligue 1": "french-ligue-1-logo.png",
        "German League": "german-league-logo.png",
        "Italian Serie A": "italian-serie-a-logo.png",
        "Spanish La Liga": "spanish-laliga-logo.png"
    }

    cols = st.columns(len(competitions_dict))
    for i, (comp_name, logo_file) in enumerate(competitions_dict.items()):
        with cols[i]:
            logo_base64 = image_to_base64(logo_file)
            st.markdown(f'''
                <div class="competition-container">
                    <img src="data:image/png;base64,{logo_base64}" class="competition-logo"/>
                </div>
            ''', unsafe_allow_html=True)
            if st.button(comp_name, key=comp_name):
                st.session_state.selected_comp = comp_name

# --- TEAM & OPPONENT SELECTION ---
if st.session_state.selected_comp:
    selected_comp = st.session_state.selected_comp
    st.markdown(f"### SELECTED COMPETITION: **{selected_comp}**")

    if st.button("🔄 Reselect Competition"):
        st.session_state.selected_comp = None

    venue_map = {"Home": "home", "Away": "away", "Neutral": "neutral"}
    selected_venue = st.selectbox("SELECT VENUE", list(venue_map.keys()))
    venue_code = venue_map[selected_venue]

    season_teams = sorted(df[(df['comp'] == selected_comp) & (df['date'].dt.year == 2025)]['team'].unique())
    selected_team = st.selectbox("SELECT TEAM", season_teams)
    opponent_teams = [team for team in season_teams if team != selected_team]
    selected_opp = st.selectbox("SELECT OPPONENT", opponent_teams)

    if st.button("PREDICT MATCH OUTCOME"):
        team_matches = df[(df['team'] == selected_team) & (df['comp'] == selected_comp) & (df['date'] <= cutoff_date)]
        opp_matches = df[(df['team'] == selected_opp) & (df['comp'] == selected_comp) & (df['date'] <= cutoff_date)]

        last10_team = team_matches.sort_values('date').tail(10)
        last10_opp = opp_matches.sort_values('date').tail(10)

        if len(last10_team) < 1 or len(last10_opp) < 1:
            st.warning("⚠️ Not enough match data available to compute rolling averages for both teams.")
        else:
            team_avg = last10_team[manual_cols].mean()
            team_avg.index = ['team_rolling_' + col for col in team_avg.index]

            opp_avg = last10_opp[manual_cols].mean()
            opp_avg.index = ['opp_rolling_' + col for col in opp_avg.index]

            input_features = pd.concat([team_avg, opp_avg], axis=0).to_frame().T
            input_scaled = loaded_scaler.transform(input_features)

            rf_pred = loaded_rf_model.predict(input_scaled)[0]
            rf_proba = loaded_rf_model.predict_proba(input_scaled)[0]

            lr_pred = loaded_lr_model.predict(input_scaled)[0]
            lr_proba = loaded_lr_model.predict_proba(input_scaled)[0]

            st.subheader(f"🧠 Prediction Results: {selected_team} vs {selected_opp} ({venue_code})")
            result_df = pd.DataFrame({
                "Model": ["Random Forest", "Logistic Regression"],
                "Prediction": ["WIN" if rf_pred == 1 else "LOSS", "WIN" if lr_pred == 1 else "LOSS"],
                "Win Probability": [rf_proba[1], lr_proba[1]],
                "Loss Probability": [rf_proba[0], lr_proba[0]]
            })

            st.table(result_df.style.format({"Win Probability": "{:.2%}", "Loss Probability": "{:.2%}"}))
