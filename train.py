# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

df = pd.read_csv("data/cleaned_wwe_matches.csv")

# === EDIT: choose features / target from your CSV ===
# Example placeholders (replace with actual column names)
FEATURES = ["wrestler_1_win_rate", "wrestler_2_win_rate", "storyline_rivalry", "recent_form"]
TARGET = "winner_label"  # e.g., 0 or 1 (wrestler1 wins)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(clf, "wwe_model.pkl")
print("Model saved to wwe_model.pkl")
