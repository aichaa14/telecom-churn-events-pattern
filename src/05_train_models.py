"""
05_train_models.py
Entraînement du réseau de neurones MLP (Multi-Layer Perceptron)
pour la prédiction du churn télécom.

Architecture : 23 → 64 → 32 → 16 → 1
  - 23 features (15 originales + 8 features métier du feature engineering)
  - Activation ReLU + Dropout(0.3) sur les couches cachées
  - Sigmoid en sortie → P(churn) ∈ [0, 1]

Gestion CPU/GPU : détection automatique via torch.device
Loss         : BCELoss (Binary Cross Entropy)
Optimiseur   : Adam (lr=0.001)
Early stopping : patience=5 epochs
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.makedirs("output/models",  exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/metrics", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. DEVICE (CPU / GPU)
# ══════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*55}")
print(f"Device utilisé : {device}")
if device.type == "cuda":
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"{'='*55}\n")

# ══════════════════════════════════════════════════════════════
# 2. CHARGEMENT DES DONNÉES (Parquet → pandas → torch)
# ══════════════════════════════════════════════════════════════
print("[1/6] Chargement des données Parquet...")

FEATURE_COLS = [
    "Account Length", "VMail Message",
    "Day Mins",   "Day Calls",   "Day Charge",
    "Eve Mins",   "Eve Calls",   "Eve Charge",
    "Night Mins", "Night Calls", "Night Charge",
    "Intl Mins",  "Intl Calls",  "Intl Charge",
    "CustServ Calls",
    # Features métier (03_feature_engineering.py)
    "Total Mins", "Total Calls", "Total Charge",
    "Intl Charge Ratio", "Avg Call Duration",
    "CustServ Rate", "Night Day Ratio", "Charge Per Min",
]
TARGET_COL = "Churn"
N_FEATURES = len(FEATURE_COLS)

train_pd = pd.read_parquet("output/parquet/train_fe")
val_pd   = pd.read_parquet("output/parquet/val_fe")
test_pd  = pd.read_parquet("output/parquet/test_fe")

# Vérifier que toutes les colonnes sont présentes
missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in train_pd.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans le Parquet : {missing}")

print(f"  Train : {len(train_pd)} lignes | Val : {len(val_pd)} | Test : {len(test_pd)}")
print(f"  Features : {N_FEATURES}")
print(f"  Churn train : {train_pd[TARGET_COL].mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# 3. DATASET PYTORCH
# ══════════════════════════════════════════════════════════════
class ChurnDataset(Dataset):
    def __init__(self, df, feature_cols, target_col):
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(1)  # (N,) → (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ChurnDataset(train_pd, FEATURE_COLS, TARGET_COL)
val_ds   = ChurnDataset(val_pd,   FEATURE_COLS, TARGET_COL)
test_ds  = ChurnDataset(test_pd,  FEATURE_COLS, TARGET_COL)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# ══════════════════════════════════════════════════════════════
# 4. ARCHITECTURE DU MODÈLE
# ══════════════════════════════════════════════════════════════
class ChurnMLP(nn.Module):
    """
    MLP pour classification binaire churn / non-churn.
    Architecture : 23 → 64 → 32 → 16 → 1
    """
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

model = ChurnMLP(N_FEATURES).to(device)
print(f"\n[2/6] Architecture du modèle :")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Paramètres totaux : {total_params:,}")

# ══════════════════════════════════════════════════════════════
# 5. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════
# Calcul du pos_weight pour compenser le déséquilibre des classes
# pos_weight = nb_negatifs / nb_positifs → pénalise davantage les FN
n_neg = (train_pd[TARGET_COL] == 0).sum()
n_pos = (train_pd[TARGET_COL] == 1).sum()
pos_weight_val = (n_neg / n_pos) * 0.4  # réduction du biais vers le churn
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
print(f"\n  pos_weight = {pos_weight_val:.2f} (ratio négatifs/positifs)")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

EPOCHS   = 50
PATIENCE = 5  # early stopping

history = {
    "train_loss": [], "val_loss": [],
    "train_f1":   [], "val_f1":   [],
}

best_val_loss   = float("inf")
best_model_path = "output/models/best_churn_mlp.pt"
patience_counter = 0

print(f"\n[3/6] Entraînement (max {EPOCHS} epochs, early stopping patience={PATIENCE})...")
print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Train F1':>9} | {'Val F1':>7} | {'LR':>8}")
print("  " + "-" * 65)

t_train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    # ── Phase entraînement ──────────────────────────────────
    model.train()
    train_losses, train_preds, train_labels = [], [], []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        probs_batch = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs_batch >= 0.4).astype(int)  # seuil abaissé → plus sensible au churn
        train_preds.extend(preds.flatten())
        train_labels.extend(y_batch.cpu().numpy().flatten().astype(int))

    train_loss = np.mean(train_losses)
    train_f1   = f1_score(train_labels, train_preds, zero_division=0)

    # ── Phase validation ────────────────────────────────────
    model.eval()
    val_losses, val_preds, val_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            val_losses.append(loss.item())
            probs_batch = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs_batch >= 0.4).astype(int)
            val_preds.extend(preds.flatten())
            val_labels.extend(y_batch.cpu().numpy().flatten().astype(int))

    val_loss = np.mean(val_losses)
    val_f1   = f1_score(val_labels, val_preds, zero_division=0)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    print(f"  {epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | "
          f"{train_f1:>9.4f} | {val_f1:>7.4f} | {current_lr:>8.6f}")

    # ── Early stopping ──────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping déclenché à l'epoch {epoch} "
                  f"(val_loss n'a pas amélioré depuis {PATIENCE} epochs)")
            break

t_train_total = time.time() - t_train_start
print(f"\n  Temps d'entraînement total : {t_train_total:.1f}s")
print(f"  Meilleur modèle sauvegardé : {best_model_path}")

# ══════════════════════════════════════════════════════════════
# 6. ÉVALUATION FINALE SUR LE TEST SET
# ══════════════════════════════════════════════════════════════
print(f"\n[4/6] Évaluation finale sur le jeu de test...")

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_probs, all_preds, all_labels = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        probs   = torch.sigmoid(model(X_batch)).cpu().numpy()
        preds   = (probs >= 0.4).astype(int)
        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())
        all_labels.extend(y_batch.numpy().flatten().astype(int))

from sklearn.metrics import f1_score
import numpy as np

thresholds = np.arange(0.1, 0.9, 0.05)
best_thresh = 0.4
best_f1 = 0

for t in thresholds:
    preds_t = (np.array(all_probs) >= t).astype(int)
    f1_t = f1_score(all_labels, preds_t, zero_division=0)
    print(f"Seuil {t:.2f} → F1={f1_t:.4f}")
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thresh = t

print(f"\nMeilleur seuil : {best_thresh:.2f} → F1={best_f1:.4f}")

accuracy = accuracy_score(all_labels, all_preds)
f1       = f1_score(all_labels, all_preds, zero_division=0)
auc_roc  = roc_auc_score(all_labels, all_probs)
cm       = confusion_matrix(all_labels, all_preds)

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"MLP (AUC = {auc_roc:.3f})")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Aléatoire (AUC = 0.5)")
plt.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC — MLP PyTorch")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/roc_curve_mlp.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n{'='*55}")
print("RÉSULTATS SUR LE JEU DE TEST")
print(f"{'='*55}")
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  F1-score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc_roc:.4f}")
print(f"\n  Matrice de confusion :")
print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
print(f"\n{classification_report(all_labels, all_preds, target_names=['Non-churn','Churn'])}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

print("\n=== COMPARAISON : Random Forest ===")

X_train_np = train_pd[FEATURE_COLS].values
y_train_np = train_pd[TARGET_COL].values
X_test_np  = test_pd[FEATURE_COLS].values
y_test_np  = test_pd[TARGET_COL].values

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",  # gère le déséquilibre automatiquement
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_np, y_train_np)

rf_probs = rf.predict_proba(X_test_np)[:, 1]
rf_preds = (rf_probs >= 0.5).astype(int)

rf_f1  = f1_score(y_test_np, rf_preds, zero_division=0)
rf_auc = roc_auc_score(y_test_np, rf_probs)

print(f"Random Forest — F1: {rf_f1:.4f} | AUC-ROC: {rf_auc:.4f}")

print("\n=== TABLEAU COMPARATIF ===")
print(f"{'Modèle':<20} {'F1':>8} {'AUC-ROC':>10}")
print("-" * 40)
print(f"{'MLP PyTorch':<20} {f1:>8.4f} {auc_roc:>10.4f}")
print(f"{'Random Forest':<20} {rf_f1:>8.4f} {rf_auc:>10.4f}")

models = ["MLP PyTorch", "Random Forest"]
f1s  = [f1, rf_f1]
aucs = [auc_roc, rf_auc]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
bars0 = axes[0].bar(models, f1s, color=["steelblue", "tomato"], width=0.5)
axes[0].set_title("F1-score comparatif")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("F1-score")
for bar, v in zip(bars0, f1s):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

bars1 = axes[1].bar(models, aucs, color=["steelblue", "tomato"], width=0.5)
axes[1].set_title("AUC-ROC comparatif")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("AUC-ROC")
for bar, v in zip(bars1, aucs):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

plt.suptitle("MLP PyTorch vs Random Forest — Comparaison des performances", fontsize=12)
plt.tight_layout()
plt.savefig("output/figures/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# 7. VISUALISATIONS
# ══════════════════════════════════════════════════════════════
print("[5/6] Génération des visualisations...")

epochs_range = range(1, len(history["train_loss"]) + 1)

# Courbes Loss
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="steelblue")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",   color="tomato")
axes[0].set_title("Courbe de perte (Loss)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCELoss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Courbes F1
axes[1].plot(epochs_range, history["train_f1"], label="Train F1", color="steelblue")
axes[1].plot(epochs_range, history["val_f1"],   label="Val F1",   color="tomato")
axes[1].set_title("Courbe F1-score")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("F1-score")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Entraînement MLP — Prédiction Churn Télécom", fontsize=13)
plt.tight_layout()
plt.savefig("output/figures/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Matrice de confusion
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Non-churn", "Churn"])
ax.set_yticklabels(["Non-churn", "Churn"])
ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
ax.set_title(f"Matrice de confusion\nF1={f1:.3f} | AUC={auc_roc:.3f}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("  output/figures/training_curves.png")
print("  output/figures/confusion_matrix.png")

# ══════════════════════════════════════════════════════════════
# 8. SAUVEGARDE DES MÉTRIQUES
# ══════════════════════════════════════════════════════════════
print("\n[6/6] Sauvegarde des métriques...")

metrics = {
    "accuracy":  round(accuracy, 4),
    "f1_score":  round(f1, 4),
    "auc_roc":   round(auc_roc, 4),
    "train_time_s": round(t_train_total, 1),
    "device":    str(device),
    "epochs_run": len(history["train_loss"]),
    "best_val_loss": round(best_val_loss, 4),
}

pd.DataFrame([metrics]).to_csv("output/metrics/test_metrics.csv", index=False)
print("  output/metrics/test_metrics.csv")

print(f"\n{'='*55}")
print("ENTRAÎNEMENT TERMINÉ ✓")
print(f"{'='*55}")
print(f"  F1-score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc_roc:.4f}")
print(f"  Device    : {device}")
print(f"  Durée     : {t_train_total:.1f}s")