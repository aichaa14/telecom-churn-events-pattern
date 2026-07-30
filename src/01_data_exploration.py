from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time

# =====================================================
# Début du chronométrage
# =====================================================

start_time = time.time()

# =====================================================
# Création de la session Spark
# =====================================================

spark = SparkSession.builder \
    .appName("01_Data_Exploration") \
    .getOrCreate()

# =====================================================
# Chargement du dataset
# =====================================================

df = spark.read.csv(
    "data/CDR.csv",
    header=True,
    inferSchema=True
)

# =====================================================
# Dimensions
# =====================================================

rows = df.count()
cols = len(df.columns)

print("="*60)
print("Dimensions")
print("="*60)

print(f"Lignes    : {rows}")
print(f"Colonnes  : {cols}")
print(f"Partitions: {df.rdd.getNumPartitions()}")

# =====================================================
# Schéma
# =====================================================

print("\nSchéma")

df.printSchema()

# =====================================================
# Types des variables
# =====================================================

print("\nTypes des variables")

for c,d in df.dtypes:
    print(f"{c:20} -> {d}")

# =====================================================
# Statistiques descriptives
# =====================================================

print("\nStatistiques")

df.describe().show()

# =====================================================
# Valeurs manquantes
# =====================================================

print("\nValeurs manquantes")

missing = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

missing.show(truncate=False)

# =====================================================
# Doublons
# =====================================================

duplicates = rows - df.dropDuplicates().count()

print(f"\nDoublons : {duplicates}")

# =====================================================
# Distribution du churn
# =====================================================

print("\nDistribution du Churn")

churn = df.groupBy("Churn").count().orderBy("Churn")

churn.show()

pdf = churn.toPandas()

plt.figure(figsize=(6,4))

plt.bar(pdf["Churn"].astype(str), pdf["count"])

plt.title("Distribution du Churn")

plt.xlabel("Classe")

plt.ylabel("Nombre")

plt.savefig(
    "output/figures/churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Corrélations
# =====================================================

numeric_cols = [

'Account Length',

'VMail Message',

'Day Mins',

'Day Calls',

'Day Charge',

'Eve Mins',

'Eve Calls',

'Eve Charge',

'Night Mins',

'Night Calls',

'Night Charge',

'Intl Mins',

'Intl Calls',

'Intl Charge',

'CustServ Calls'

]

assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features"
)

corr_df = assembler.transform(df)

corr_matrix = Correlation.corr(
    corr_df,
    "features",
    "pearson"
).head()[0]

corr = pd.DataFrame(
    corr_matrix.toArray(),
    columns=numeric_cols,
    index=numeric_cols
)

plt.figure(figsize=(12,10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Matrice de corrélation")

plt.savefig(
    "output/figures/correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Histogrammes
# =====================================================

os.makedirs("output/figures/histograms", exist_ok=True)

pdf = df.select(numeric_cols).toPandas()

for c in numeric_cols:

    plt.figure(figsize=(5,4))

    plt.hist(
        pdf[c],
        bins=30
    )

    plt.title(c)

    plt.tight_layout()

    plt.savefig(
        f"output/figures/histograms/{c}.png",
        dpi=300
    )

    plt.close()

pdf_features = df.select(numeric_cols).toPandas()
key_features = ["Day Mins", "CustServ Calls", "Intl Mins", "VMail Message", "Account Length", "Day Charge"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flatten(), key_features):
    ax.hist(pdf_features[col].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("Valeur")
    ax.set_ylabel("Fréquence")
plt.suptitle("Distribution des features clés — CDR Telecom", fontsize=13)
plt.tight_layout()
plt.savefig("output/figures/features_histograms.png", dpi=150, bbox_inches="tight")
plt.close()

pdf_box = df.select(["CustServ Calls", "Day Mins", "Intl Mins", "Churn"]).toPandas()
pdf_box["Churn"] = pdf_box["Churn"].map({True: "Churn", False: "Non-churn"})
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, ["CustServ Calls", "Day Mins", "Intl Mins"]):
    groups = [pdf_box[pdf_box["Churn"] == g][col].dropna() for g in ["Non-churn", "Churn"]]
    ax.boxplot(groups, tick_labels=["Non-churn", "Churn"])
    ax.set_title(col)
    ax.set_ylabel("Valeur")
plt.suptitle("Distribution par classe — Churn vs Non-churn", fontsize=13)
plt.tight_layout()
plt.savefig("output/figures/boxplot_churn.png", dpi=150, bbox_inches="tight")
plt.close()
# =====================================================
# Conversion Parquet
# =====================================================

df.write \
.mode("overwrite") \
.option("compression","snappy") \
.parquet("output/parquet/CDR_PARQUET")

# =====================================================
# Temps d'exécution
# =====================================================

elapsed = time.time() - start_time

print("="*60)
print("Temps d'exécution")
print("="*60)

print(f"{elapsed:.2f} secondes")

print("\nParquet créé : output/parquet/CDR_PARQUET")

spark.stop()