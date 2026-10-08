import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# tests never touch the network: the orchestrator must not pull history from the public archive
os.environ["ARCHIVE_BACKFILL"] = "false"
