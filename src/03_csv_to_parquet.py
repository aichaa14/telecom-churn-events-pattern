"""
04_csv_to_parquet.py
Conversion optimale CSV → Parquet avec compression Snappy.
Répond au point 4 du cahier des charges :
"Discover a more adequate format for larger datasets — PARQUET
 How to convert csv to parquet format the optimal way — Compression via Snappy"

Ce script :
1. Lit le CSV brut avec un schéma explicite (plus rapide qu'inferSchema)
2. Convertit vers Parquet avec compression Snappy
3. Produit un benchmark comparatif CSV vs Parquet (taille + vitesse de lecture)
4. Documente les gains obtenus
"""

import os
import time
import glob
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, BooleanType
)

os.environ["SPARK_LOCAL_DIRS"] = r"C:\Users\hp\Desktop\STAGE\Telecom_Churn_Project\tmp_spark"
os.makedirs(r"C:\Users\hp\Desktop\STAGE\Telecom_Churn_Project\tmp_spark", exist_ok=True)
os.makedirs("output/parquet", exist_ok=True)
os.makedirs("output/benchmark", exist_ok=True)

# ── SparkSession ──────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("EventsPattern-CSV-to-Parquet")
    .master("local[*]")
    .config("spark.local.dir", r"C:\Users\hp\Desktop\STAGE\Telecom_Churn_Project\tmp_spark")
    .config("spark.sql.shuffle.partitions", "8")
    # Compression Snappy par défaut pour tous les writes Parquet
    .config("spark.sql.parquet.compression.codec", "snappy")
    # Optimisation lecture Parquet : lecture vectorisée
    .config("spark.sql.parquet.enableVectorizedReader", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

CSV_PATH     = "data/CDR.csv"
PARQUET_PATH = "output/parquet/CDR_PARQUET_OPTIMISED"

# ── Schéma explicite (meilleure pratique vs inferSchema) ─────────────────────
# inferSchema lit le fichier deux fois pour deviner les types.
# Un schéma explicite évite ce double passage → 2× plus rapide sur grands fichiers.
schema = StructType([
    StructField("Phone Number",    StringType(),  True),
    StructField("Account Length",  IntegerType(), True),
    StructField("VMail Message",   IntegerType(), True),
    StructField("Day Mins",        DoubleType(),  True),
    StructField("Day Calls",       IntegerType(), True),
    StructField("Day Charge",      DoubleType(),  True),
    StructField("Eve Mins",        DoubleType(),  True),
    StructField("Eve Calls",       IntegerType(), True),
    StructField("Eve Charge",      DoubleType(),  True),
    StructField("Night Mins",      DoubleType(),  True),
    StructField("Night Calls",     IntegerType(), True),
    StructField("Night Charge",    DoubleType(),  True),
    StructField("Intl Mins",       DoubleType(),  True),
    StructField("Intl Calls",      IntegerType(), True),
    StructField("Intl Charge",     DoubleType(),  True),
    StructField("CustServ Calls",  IntegerType(), True),
    StructField("Churn",           BooleanType(), True),
])

print("=" * 60)
print("CONVERSION CSV → PARQUET (Snappy)")
print("=" * 60)

# ── 1. Lecture CSV avec schéma explicite ─────────────────────────────────────
print("\n[1/4] Lecture du CSV avec schéma explicite...")
t0 = time.time()
df = spark.read.csv(CSV_PATH, header=True, schema=schema)
count = df.count()
t_csv_read = time.time() - t0
print(f"      {count} lignes lues en {t_csv_read:.2f}s")
print(f"      Partitions : {df.rdd.getNumPartitions()}")

# ── 2. Écriture Parquet Snappy ────────────────────────────────────────────────
print("\n[2/4] Écriture Parquet (compression Snappy)...")
t0 = time.time()
df.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(PARQUET_PATH)
t_write = time.time() - t0
print(f"      Écrit en {t_write:.2f}s → {PARQUET_PATH}")

# ── 3. Relecture Parquet pour benchmark ───────────────────────────────────────
print("\n[3/4] Benchmark lecture CSV vs Parquet...")

# Lecture CSV (avec inferSchema pour simuler un usage standard)
t0 = time.time()
df_csv_bench = spark.read.csv(CSV_PATH, header=True, inferSchema=True)
_ = df_csv_bench.count()
t_csv_infer = time.time() - t0

# Lecture Parquet
t0 = time.time()
df_parquet_bench = spark.read.parquet(PARQUET_PATH)
_ = df_parquet_bench.count()
t_parquet_read = time.time() - t0

# ── 4. Calcul des tailles ─────────────────────────────────────────────────────
csv_size_kb = os.path.getsize(CSV_PATH) / 1024

parquet_files = glob.glob(os.path.join(PARQUET_PATH, "*.parquet"))
parquet_size_kb = sum(os.path.getsize(f) for f in parquet_files) / 1024

ratio_taille  = csv_size_kb / parquet_size_kb if parquet_size_kb > 0 else 0
gain_taille   = (1 - parquet_size_kb / csv_size_kb) * 100
gain_vitesse  = (1 - t_parquet_read / t_csv_infer) * 100

# ── Résultats ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RÉSULTATS DU BENCHMARK")
print("=" * 60)
print(f"\n{'Métrique':<35} {'CSV':>12} {'Parquet (Snappy)':>18}")
print("-" * 65)
print(f"{'Taille fichier':<35} {csv_size_kb:>10.1f} Ko {parquet_size_kb:>14.1f} Ko")
print(f"{'Temps lecture (inferSchema/schéma)':<35} {t_csv_infer:>10.2f}s  {t_parquet_read:>14.2f}s")
print(f"{'Nombre de fichiers':<35} {'1':>12} {len(parquet_files):>18}")
print(f"{'Schéma intégré':<35} {'Non':>12} {'Oui':>18}")
print(f"{'Lecture par colonne sélective':<35} {'Non':>12} {'Oui':>18}")
print("-" * 65)
print(f"\n  Gain de taille    : {gain_taille:.1f}%  (ratio {ratio_taille:.1f}×)")
print(f"  Gain de vitesse   : {gain_vitesse:.1f}%  (lecture Parquet vs CSV+inferSchema)")
print(f"  Compression       : Snappy (compromis vitesse/taille optimal pour Spark)")

print("\n" + "=" * 60)
print("POURQUOI PARQUET + SNAPPY ?")
print("=" * 60)
print("""
  Format colonnaire : Spark ne lit que les colonnes nécessaires
  → SELECT Day Mins, Churn ne charge pas les 15 autres colonnes

  Schéma intégré : pas d'inférence à la lecture
  → Lecture 2× plus rapide que CSV avec inferSchema

  Compression Snappy : algorithme rapide (priorité vitesse)
  → Meilleur compromis décompression rapide / taille réduite
  → Alternative : gzip (plus compact mais plus lent)

  Statistiques de colonnes : min/max stockés dans les métadonnées
  → Spark peut ignorer des blocs entiers (predicate pushdown)
  → Accélération supplémentaire sur les filtres WHERE
""")

# ── Sauvegarde du rapport benchmark en texte ─────────────────────────────────
rapport = f"""BENCHMARK CSV vs PARQUET (Snappy)
Dataset : CDR Telecom — {count} lignes, 17 colonnes
{'='*50}
CSV    — taille : {csv_size_kb:.1f} Ko | lecture : {t_csv_infer:.2f}s
Parquet— taille : {parquet_size_kb:.1f} Ko | lecture : {t_parquet_read:.2f}s
Gain taille    : {gain_taille:.1f}% (ratio {ratio_taille:.1f}x)
Gain vitesse   : {gain_vitesse:.1f}%
Compression    : Snappy
"""
with open("output/benchmark/csv_vs_parquet.txt", "w") as f:
    f.write(rapport)
print("Rapport benchmark sauvegardé : output/benchmark/csv_vs_parquet.txt")

spark.stop()
print("\nConversion terminée ✓")