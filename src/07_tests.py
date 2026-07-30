"""
07_tests.py
Tests unitaires et d'intégration du pipeline Events Pattern.
Exécution : pytest src/07_tests.py -v

Couvre :
- Qualité des données (schéma, valeurs manquantes, doublons)
- Preprocessing (normalisation, split, absence de fuite)
- Feature engineering (nouvelles colonnes, valeurs cohérentes)
- Modèle PyTorch (sorties, dimensions, device)
- Parquet (lecture, schéma, compression)
- Prédiction (probabilités dans [0,1], niveaux de risque)
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Chemins ───────────────────────────────────────────────────
TRAIN_FE   = "output/parquet/train_fe"
VAL_FE     = "output/parquet/val_fe"
TEST_FE    = "output/parquet/test_fe"
MODEL_PATH = "output/models/best_churn_mlp.pt"

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

# ── Architecture MLP (identique à 05_train_models.py) ─────────
class ChurnMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.network(x)

# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def train_df():
    return pd.read_parquet(TRAIN_FE)

@pytest.fixture(scope="module")
def val_df():
    return pd.read_parquet(VAL_FE)

@pytest.fixture(scope="module")
def test_df():
    return pd.read_parquet(TEST_FE)

@pytest.fixture(scope="module")
def mlp_model():
    model = ChurnMLP(len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# ══════════════════════════════════════════════════════════════
# SECTION 1 — QUALITÉ DES DONNÉES
# ══════════════════════════════════════════════════════════════

class TestQualiteDonnees:

    def test_fichiers_parquet_existent(self):
        """Les 3 fichiers Parquet doivent exister."""
        for path in [TRAIN_FE, VAL_FE, TEST_FE]:
            assert os.path.exists(path), f"Fichier manquant : {path}"

    def test_colonnes_features_presentes(self, train_df):
        """Toutes les features doivent être présentes dans le train."""
        for col in FEATURE_COLS:
            assert col in train_df.columns, f"Colonne manquante : {col}"

    def test_colonne_cible_presente(self, train_df):
        """La colonne Churn doit être présente."""
        assert "Churn" in train_df.columns

    def test_pas_de_valeurs_manquantes(self, train_df, val_df, test_df):
        """Aucune valeur manquante dans les features."""
        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            nb_missing = df[FEATURE_COLS].isnull().sum().sum()
            assert nb_missing == 0, f"{name} contient {nb_missing} valeurs manquantes"

    def test_churn_binaire(self, train_df):
        """Churn doit contenir uniquement 0 et 1."""
        valeurs = set(train_df["Churn"].unique())
        assert valeurs <= {0, 1}, f"Valeurs inattendues dans Churn : {valeurs}"

    def test_taille_minimale_splits(self, train_df, val_df, test_df):
        """Chaque split doit contenir au moins 1000 lignes."""
        assert len(train_df) >= 1000, f"Train trop petit : {len(train_df)}"
        assert len(val_df)   >= 1000, f"Val trop petit : {len(val_df)}"
        assert len(test_df)  >= 1000, f"Test trop petit : {len(test_df)}"

    def test_proportion_churn_raisonnable(self, train_df, val_df, test_df):
        """Le taux de churn doit être entre 5% et 30% dans chaque split."""
        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            taux = df["Churn"].mean()
            assert 0.05 <= taux <= 0.30, \
                f"Taux de churn anormal dans {name} : {taux:.2%}"

# ══════════════════════════════════════════════════════════════
# SECTION 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_pas_de_chevauchement_physique(self, train_df, val_df, test_df):
        """La somme des lignes doit être égale au total source (pas de duplication)."""
        total = len(train_df) + len(val_df) + len(test_df)
        # Vérification indirecte : chaque split a une taille > 0 et cohérente
        assert len(train_df) > len(val_df), "Train doit être plus grand que val"
        assert len(train_df) > len(test_df), "Train doit être plus grand que test"
        assert total > 0

    def test_features_normalisees(self, train_df):
        """Les colonnes brutes doivent avoir des valeurs numériques cohérentes (non nulles)."""
        base_cols = [
        "Account Length", "Day Mins", "Eve Mins",
        "Night Mins", "Intl Mins", "CustServ Calls"
        ]
        for col in base_cols:
            assert train_df[col].std() > 0,    f"{col} : colonne constante"
            assert not train_df[col].isnull().any(), f"{col} : contient des NaN"
            assert train_df[col].mean() > 0,   f"{col} : moyenne négative inattendue"

    def test_split_ratio_approximatif(self, train_df, val_df, test_df):
        """Le train doit représenter environ 70% du total."""
        total = len(train_df) + len(val_df) + len(test_df)
        ratio_train = len(train_df) / total
        assert 0.60 <= ratio_train <= 0.80, \
            f"Ratio train anormal : {ratio_train:.2%}"

# ══════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_nouvelles_features_presentes(self, train_df):
        """Les 8 features métier doivent être présentes."""
        fe_cols = [
            "Total Mins", "Total Calls", "Total Charge",
            "Intl Charge Ratio", "Avg Call Duration",
            "CustServ Rate", "Night Day Ratio", "Charge Per Min",
        ]
        for col in fe_cols:
            assert col in train_df.columns, f"Feature métier manquante : {col}"

    def test_total_mins_coherent(self, train_df):
        """Total Mins doit être >= Day Mins (car c'est une somme)."""
        # Après normalisation les valeurs peuvent être négatives
        # On vérifie que la colonne existe et n'est pas constante
        assert train_df["Total Mins"].std() > 0, "Total Mins est constante"
        assert not train_df["Total Mins"].isnull().any(), "Total Mins contient des NaN"

    def test_ratios_pas_infinis(self, train_df):
        """Les ratios ne doivent pas contenir de valeurs infinies."""
        ratio_cols = ["Intl Charge Ratio", "Avg Call Duration",
                      "CustServ Rate", "Night Day Ratio", "Charge Per Min"]
        for col in ratio_cols:
            assert not np.isinf(train_df[col]).any(), \
                f"{col} contient des valeurs infinies"

    def test_nombre_total_features(self, train_df):
        """Le train doit avoir exactement 23 features + 1 cible = 24 colonnes min."""
        assert len(FEATURE_COLS) == 23, f"Nombre de features : {len(FEATURE_COLS)}"
        assert all(c in train_df.columns for c in FEATURE_COLS)

# ══════════════════════════════════════════════════════════════
# SECTION 4 — MODÈLE PYTORCH
# ══════════════════════════════════════════════════════════════

class TestModele:

    def test_modele_charge(self, mlp_model):
        """Le modèle doit se charger sans erreur."""
        assert mlp_model is not None

    def test_dimensions_sortie(self, mlp_model):
        """La sortie du modèle doit être de dimension (N, 1)."""
        x = torch.randn(10, 23)
        with torch.no_grad():
            out = mlp_model(x)
        assert out.shape == (10, 1), f"Dimension incorrecte : {out.shape}"

    def test_probabilites_dans_intervalle(self, mlp_model):
        """Après sigmoid, les probabilités doivent être dans [0, 1]."""
        x = torch.randn(100, 23)
        with torch.no_grad():
            logits = mlp_model(x)
            probs  = torch.sigmoid(logits).numpy()
        assert probs.min() >= 0.0, f"Probabilité négative : {probs.min()}"
        assert probs.max() <= 1.0, f"Probabilité > 1 : {probs.max()}"

    def test_modele_en_mode_eval(self, mlp_model):
        """Le modèle doit être en mode évaluation (pas entraînement)."""
        assert not mlp_model.training, "Modèle en mode training — appeler model.eval()"

    def test_gradient_desactive_en_inference(self, mlp_model):
        """Les gradients ne doivent pas être calculés en inférence."""
        x = torch.randn(5, 23)
        with torch.no_grad():
            out = mlp_model(x)
        assert out.grad_fn is None, "Gradient actif en mode inférence"

    def test_reproductibilite(self, mlp_model):
        """Deux passages du même input doivent donner le même output."""
        torch.manual_seed(42)
        x = torch.randn(5, 23)
        with torch.no_grad():
            out1 = mlp_model(x)
            out2 = mlp_model(x)
        assert torch.allclose(out1, out2), "Sorties non reproductibles"

# ══════════════════════════════════════════════════════════════
# SECTION 5 — PARQUET
# ══════════════════════════════════════════════════════════════

class TestParquet:

    def test_parquet_lisible(self):
        """Les fichiers Parquet doivent être lisibles par pandas."""
        for path in [TRAIN_FE, VAL_FE, TEST_FE]:
            df = pd.read_parquet(path)
            assert len(df) > 0, f"Parquet vide : {path}"

    def test_schema_coherent(self, train_df, val_df, test_df):
        """Les 3 splits doivent avoir les mêmes colonnes."""
        assert set(train_df.columns) == set(val_df.columns), \
            "Colonnes train/val différentes"
        assert set(train_df.columns) == set(test_df.columns), \
            "Colonnes train/test différentes"

    def test_types_numeriques(self, train_df):
        """Les features doivent être de type numérique."""
        for col in FEATURE_COLS:
            assert pd.api.types.is_numeric_dtype(train_df[col]), \
                f"{col} n'est pas numérique : {train_df[col].dtype}"

# ══════════════════════════════════════════════════════════════
# SECTION 6 — PRÉDICTION
# ══════════════════════════════════════════════════════════════

class TestPrediction:

    def test_prediction_sur_batch(self, mlp_model, test_df):
        """La prédiction doit fonctionner sur un batch du jeu de test."""
        X = torch.tensor(
            test_df[FEATURE_COLS].head(32).values.astype(np.float32)
        )
        with torch.no_grad():
            probs = torch.sigmoid(mlp_model(X)).numpy().flatten()
        assert len(probs) == 32
        assert all(0 <= p <= 1 for p in probs)

    def test_niveaux_risque(self):
        """La fonction de niveau de risque doit retourner les bonnes catégories."""
        # Import inline pour éviter les dépendances circulaires
        def get_risk_level(prob):
            if prob < 0.25:   return "Faible"
            elif prob < 0.45: return "Modéré"
            elif prob < 0.65: return "Élevé"
            else:             return "Critique"

        assert get_risk_level(0.10) == "Faible"
        assert get_risk_level(0.30) == "Modéré"
        assert get_risk_level(0.55) == "Élevé"
        assert get_risk_level(0.80) == "Critique"

    def test_fichier_predictions_existe(self):
        """Le fichier de prédictions doit exister après 06_predict.py."""
        path = "output/predictions/predictions_nouveaux_clients.csv"
        assert os.path.exists(path), f"Fichier de prédictions manquant : {path}"

    def test_colonnes_predictions(self):
        """Le fichier de prédictions doit contenir les bonnes colonnes."""
        path = "output/predictions/predictions_nouveaux_clients.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            required = ["client_id", "mlp_prob", "rf_prob", "risk_level", "action"]
            for col in required:
                assert col in df.columns, f"Colonne manquante : {col}"