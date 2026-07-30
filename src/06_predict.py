"""
06_predict.py
Inférence sur de nouveaux clients pour la prédiction du churn.
- Charge le meilleur modèle MLP sauvegardé
- Charge le Random Forest entraîné
- Prédit la probabilité de churn pour chaque client
- Classifie par niveau de risque : Faible / Modéré / Élevé / Critique
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

os.makedirs("output/predictions", exist_ok=True)

# ── Features ──────────────────────────────────────────────────
FEATURE_COLS = [
    "Account Length", "VMail Message",
    "Day Mins",   "Day Calls",   "Day Charge",
    "Eve Mins",   "Eve Calls",   "Eve Charge",
    "Night Mins", "Night Calls", "Night Charge",
    "Intl Mins",  "Intl Calls",  "Intl Charge",
    "CustServ Calls",
    "Total Mins", "Total Calls", "Total Charge",
    "Intl Charge Ratio", "Avg Call Duration",
    "CustServ Rate", "Night Day Ratio", "Charge Per Min",
]
N_FEATURES = len(FEATURE_COLS)

# ── Niveau de risque ──────────────────────────────────────────
def get_risk_level(prob):
    if prob < 0.25:
        return "🟢 Faible"
    elif prob < 0.45:
        return "🟡 Modéré"
    elif prob < 0.65:
        return "🟠 Élevé"
    else:
        return "🔴 Critique"

def get_action(prob):
    if prob < 0.25:
        return "Aucune action requise"
    elif prob < 0.45:
        return "Surveillance — suivi mensuel"
    elif prob < 0.65:
        return "Offre de rétention ciblée"
    else:
        return "Contact prioritaire — offre personnalisée urgente"

# ── Architecture MLP ──────────────────────────────────────────
class ChurnMLP(nn.Module):
    def __init__(self, n_features):
        super(ChurnMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.network(x)

# ── Feature engineering (même logique que 03_feature_engineering.py) ──
def add_features(df):
    df = df.copy()
    df["Total Mins"]        = df["Day Mins"] + df["Eve Mins"] + df["Night Mins"] + df["Intl Mins"]
    df["Total Calls"]       = df["Day Calls"] + df["Eve Calls"] + df["Night Calls"] + df["Intl Calls"]
    df["Total Charge"]      = df["Day Charge"] + df["Eve Charge"] + df["Night Charge"] + df["Intl Charge"]
    df["Intl Charge Ratio"] = df["Intl Charge"] / df["Total Charge"].replace(0, np.nan)
    df["Avg Call Duration"] = df["Total Mins"] / df["Total Calls"].replace(0, np.nan)
    df["CustServ Rate"]     = df["CustServ Calls"] / df["Account Length"].replace(0, np.nan)
    df["Night Day Ratio"]   = df["Night Mins"] / df["Day Mins"].replace(0, np.nan)
    df["Charge Per Min"]    = df["Total Charge"] / df["Total Mins"].replace(0, np.nan)
    return df.fillna(0)

# ── Chargement du scaler (fitté sur le train) ─────────────────
print("=" * 55)
print("Chargement des données de référence (train)...")
train_pd = pd.read_parquet("output/parquet/train_fe")
BASE_COLS = [
    "Account Length", "VMail Message",
    "Day Mins",   "Day Calls",   "Day Charge",
    "Eve Mins",   "Eve Calls",   "Eve Charge",
    "Night Mins", "Night Calls", "Night Charge",
    "Intl Mins",  "Intl Calls",  "Intl Charge",
    "CustServ Calls",
]
scaler = StandardScaler()
scaler.fit(train_pd[BASE_COLS])
print(f"Scaler chargé depuis {len(train_pd)} exemples d'entraînement ✓")

# ── Chargement du modèle MLP ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = ChurnMLP(N_FEATURES).to(device)
mlp_model.load_state_dict(torch.load("output/models/best_churn_mlp.pt", map_location=device))
mlp_model.eval()
print(f"Modèle MLP chargé ✓ (device: {device})")

# ── Entraînement rapide du Random Forest sur le train ─────────
print("Entraînement Random Forest...")
X_train = train_pd[FEATURE_COLS].values
y_train = train_pd["Churn"].values
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Random Forest prêt ✓")

# ── Nouveaux clients à prédire ────────────────────────────────
# Charger de vrais clients du jeu de test
test_pd = pd.read_parquet("output/parquet/test_fe")

# Sélectionner des profils contrastés : 3 vrais churns + 2 non-churns
churns     = test_pd[test_pd["Churn"] == 1].head(3)
non_churns = test_pd[test_pd["Churn"] == 0].head(2)
nouveaux_clients_fe = pd.concat([churns, non_churns]).reset_index(drop=True)

# Etiquettes réelles pour vérification
labels_reels = nouveaux_clients_fe["Churn"].tolist()
clients_fe   = nouveaux_clients_fe[FEATURE_COLS].copy()

# PAS besoin de rescaler — les données sont déjà normalisées
# Remplacer la ligne scaler.transform par :
# clients_fe[BASE_COLS] = scaler.transform(clients_fe[BASE_COLS])  ← SUPPRIMER cette ligne
# Chargement de vrais clients du jeu de test (déjà preprocessés et normalisés)
test_pd = pd.read_parquet("output/parquet/test_fe")

# 3 vrais churns + 2 non-churns
churns     = test_pd[test_pd["Churn"] == 1].head(3)
non_churns = test_pd[test_pd["Churn"] == 0].head(2)
clients_fe = pd.concat([churns, non_churns]).reset_index(drop=True)

labels_reels = clients_fe["Churn"].tolist()

# ── Prédiction MLP ────────────────────────────────────────────
X_new = torch.tensor(clients_fe[FEATURE_COLS].values.astype(np.float32)).to(device)
with torch.no_grad():
    mlp_probs = torch.sigmoid(mlp_model(X_new)).cpu().numpy().flatten()

# ── Prédiction Random Forest ──────────────────────────────────
rf_probs = rf_model.predict_proba(clients_fe[FEATURE_COLS].values)[:, 1]

# ── Résultats ─────────────────────────────────────────────────
print("\n" + "=" * 75)
print("RÉSULTATS DE PRÉDICTION — NOUVEAUX CLIENTS")
print("=" * 75)
print(f"{'Client':<8} {'MLP P(churn)':<14} {'RF P(churn)':<13} {'Niveau risque':<18} {'Action recommandée'}")
print("-" * 75)

results = []
for i, (mlp_p, rf_p) in enumerate(zip(mlp_probs, rf_probs)):
    # Moyenne des deux modèles pour la décision finale
    ensemble_p = (mlp_p + rf_p) / 2
    risk = get_risk_level(ensemble_p)
    action = get_action(ensemble_p)
    reel = "✓ Churn" if labels_reels[i] == 1 else "✗ Non-churn"
    print(f"Client {i+1:<2} {mlp_p:<14.3f} {rf_p:<13.3f} {risk:<18} {reel:<14} {action}")
    results.append({
        "client_id":    i + 1,
        "mlp_prob":     round(float(mlp_p), 4),
        "rf_prob":      round(float(rf_p), 4),
        "ensemble_prob": round(float(ensemble_p), 4),
        "risk_level":   risk,
        "action":       action,
        "churn_predit": int(ensemble_p >= 0.40),
    })

# ── Sauvegarde ────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("output/predictions/predictions_nouveaux_clients.csv", index=False)

print("\n" + "=" * 75)
print(f"Clients à risque élevé/critique : {sum(r['churn_predit'] for r in results)}/{len(results)}")
print("Résultats sauvegardés : output/predictions/predictions_nouveaux_clients.csv")
print("=" * 75)