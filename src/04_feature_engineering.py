"""
03_feature_engineering.py
Création de variables métier supplémentaires à partir des données CDR.
Entrée  : output/parquet/train | val | test  (issus du preprocessing)
Sorties : output/parquet/train_fe | val_fe | test_fe
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.environ["SPARK_LOCAL_DIRS"] = r"C:\Users\hp\Desktop\STAGE\Telecom_Churn_Project\tmp_spark"
os.makedirs(r"C:\Users\hp\Desktop\STAGE\Telecom_Churn_Project\tmp_spark", exist_ok=True)

# ── SparkSession ──────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("EventsPattern-FeatureEngineering")
    .master("local[*]")
    .config("spark.local.dir", r"C:\Users\hp\Desktop\STAGE\Telecom_Churn_Project\tmp_spark")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── Chargement des splits ─────────────────────────────────────────────────────
train = spark.read.parquet("output/parquet/train")
val   = spark.read.parquet("output/parquet/val")
test  = spark.read.parquet("output/parquet/test")

print(f"Train : {train.count()} | Val : {val.count()} | Test : {test.count()}")

# ── Feature engineering métier ───────────────────────────────────────────────
def add_features(df):
    # 1. Usage total toutes périodes confondues
    df = df.withColumn("Total Mins",
        F.col("Day Mins") + F.col("Eve Mins") +
        F.col("Night Mins") + F.col("Intl Mins")
    )
    df = df.withColumn("Total Calls",
        F.col("Day Calls") + F.col("Eve Calls") +
        F.col("Night Calls") + F.col("Intl Calls")
    )
    df = df.withColumn("Total Charge",
        F.col("Day Charge") + F.col("Eve Charge") +
        F.col("Night Charge") + F.col("Intl Charge")
    )
    # 2. Ratio charge internationale / charge totale
    df = df.withColumn("Intl Charge Ratio",
        F.when(F.col("Total Charge") > 0,
               F.col("Intl Charge") / F.col("Total Charge"))
        .otherwise(0.0)
    )
    # 3. Durée moyenne par appel
    df = df.withColumn("Avg Call Duration",
        F.when(F.col("Total Calls") > 0,
               F.col("Total Mins") / F.col("Total Calls"))
        .otherwise(0.0)
    )
    # 4. Ratio appels service client / ancienneté
    df = df.withColumn("CustServ Rate",
        F.when(F.col("Account Length") > 0,
               F.col("CustServ Calls") / F.col("Account Length"))
        .otherwise(0.0)
    )
    # 5. Usage nocturne vs diurne
    df = df.withColumn("Night Day Ratio",
        F.when(F.col("Day Mins") > 0,
               F.col("Night Mins") / F.col("Day Mins"))
        .otherwise(0.0)
    )
    # 6. Charge par minute (coût perçu)
    df = df.withColumn("Charge Per Min",
        F.when(F.col("Total Mins") > 0,
               F.col("Total Charge") / F.col("Total Mins"))
        .otherwise(0.0)
    )
    return df

train_fe = add_features(train)
val_fe   = add_features(val)
test_fe  = add_features(test)

# ── Vérification ──────────────────────────────────────────────────────────────
new_features = [
    "Total Mins", "Total Calls", "Total Charge",
    "Intl Charge Ratio", "Avg Call Duration",
    "CustServ Rate", "Night Day Ratio", "Charge Per Min"
]

print(f"\nNouvelles features créées ({len(new_features)}) :")
for f in new_features:
    print(f"  - {f}")

print(f"\nTotal colonnes : {len(train.columns)} → {len(train_fe.columns)} (+ {len(new_features)})")

train_fe.select(new_features + ["Churn"]).describe().show()

# ── Sauvegarde Parquet Snappy ─────────────────────────────────────────────────
train_fe.write.mode("overwrite").option("compression", "snappy").parquet("output/parquet/train_fe")
val_fe.write.mode("overwrite").option("compression", "snappy").parquet("output/parquet/val_fe")
test_fe.write.mode("overwrite").option("compression", "snappy").parquet("output/parquet/test_fe")

print("\nSauvegarde terminée :")
print("  output/parquet/train_fe")
print("  output/parquet/val_fe")
print("  output/parquet/test_fe")

spark.stop()
print("\nFeature engineering terminé ✓")