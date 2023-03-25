from pathlib import Path

DATA_ROOT = Path("data").absolute()  # .mkdir(parents=True, exist_ok=True)

(DATA_ROOT / "raw").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "preprocessed").mkdir(parents=True, exist_ok=True)
DATA_RAW = DATA_ROOT / "raw"
DATA_PREPROCESSED = DATA_ROOT / "preprocessed"

MODEL_ROOT = Path("models").absolute()
(DATA_ROOT / "svm").mkdir(parents=True, exist_ok=True)

GNN_DATA_ROOT = Path("data_gnn").absolute()

NO_BACKUP = Path("/mount/arbeitsdaten42/studenten2/mustaffn")
