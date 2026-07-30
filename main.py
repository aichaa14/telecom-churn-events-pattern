import subprocess
import sys

scripts = [
    "src/01_csv_to_parquet.py",
    "src/02_preprocessing.py",
    "src/03_feature_engineering.py",
    "src/04_train_model.py",
    "src/05_evaluation.py",
]

for script in scripts:
    print(f"\n=== Exécution : {script} ===")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"Erreur dans {script}")
        break

print("\nPipeline terminé.")