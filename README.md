# Football Win-Loss Predictor

A Streamlit web app that predicts football match outcomes (Win/Loss) using two machine learning models—Random Forest and Logistic Regression—trained on rolling averages of recent match statistics.

---

## 📊 Project Overview

1. **Data Preparation & Feature Engineering**  
   - Loaded historical match data (`cleaned_featured_matches_without_draws.csv`) that was web scrapped through FBref.  
   - Converted dates to `datetime` and sorted chronologically.  
   - Computed rolling averages of key match stats (last 10 games):  
     - Goals For/Against (`gf`, `ga`)  
     - Expected Goals (`xg`, `xga`)  
     - Possession (`poss`)  
     - Shots (`sh`, `sot`)  
     - Average Shot Distance (`shot_dist`)  
     - Points earned (`points`)

2. **Model Training & Evaluation**  
   - **Random Forest** (2,000 trees)  
   - **Logistic Regression** (max_iter=10 000)  
   - Train/test split by date: training up to Jan 31 2025, testing Feb 1–May 10 2025  
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
   - Visualized Confusion Matrices & ROC Curves for both models

3. **Model Serialization**  
   - Saved models and `StandardScaler` with `joblib`.  
   - Compressed large Random Forest model using `gzip + joblib` to reduce disk footprint.  
   - Verified compressed model yields identical predictions.

4. **Streamlit Web App**  
   - Interactive UI with black theme & custom logo.  
   - **Competition Selection** via league logos + named buttons (EPL, Ligue 1, Bundesliga, Serie A, La Liga).  
   - **Venue**, **Team**, **Opponent** selectors.  
   - On “Predict Match Outcome” click:  
     1. Extract last 10 matches per team before cutoff (May 12 2025).  
     2. Compute rolling averages, scale features, feed models.  
     3. Display side-by-side predictions & probabilities for both models.  
   - Button hover/text/border styling and active-click color behavior.

5. **Deployment Setup**  
   - **GitHub** repo with `app.py`, model files, data, logos, `requirements.txt`.  
   - **Streamlit Cloud** – automatic deployment from `main` branch.  
   - Instructions to compress heavy models, remove `gzip` from requirements, and push updates.
