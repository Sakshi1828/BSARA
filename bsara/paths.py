"""Project paths, resolved relative to the repository root (no hard-coded user paths)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
INDEX_CSV = DATA / "index.csv"
CV_DIR = DATA / "cv"          # generated per-fold folders (git-ignored)
RESULTS = ROOT / "results"    # metrics/figures that ARE committed

CLASSES = ["drip", "impact_beating", "gunshot"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
