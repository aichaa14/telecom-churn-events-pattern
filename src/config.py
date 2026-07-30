from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossiers
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "reports"

# Fichiers
RAW_DATA = RAW_DIR / "CDR_Dataset.csv"
CLEAN_DATA = PROCESSED_DIR / "clean_data.parquet"
FEATURE_DATA = PROCESSED_DIR / "features.parquet"