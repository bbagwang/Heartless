import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# tests never touch the network: the orchestrator must not pull history from the public archive
os.environ["ARCHIVE_BACKFILL"] = "false"

# tests never read a local .env: the CLI tests call load_settings() from the working directory (the repo root), where an
# owner's .env (keys, bot token, HEARTLESS_MODE, CONTEXT_*) would otherwise leak into test state and make the suite pass
# or fail depending on the machine. The tests of the .env loading itself opt back in with monkeypatch.setitem.
from heartless.config import Settings  # noqa: E402

Settings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def _sandboxed_seal_ledger(tmp_path, monkeypatch):
    """No test may append to the real docs/results/seal_ledger.jsonl: that file is the audit trail of deliberate
    unseals, and a stray line from a test run would read as an opened final exam. Every test gets its own ledger."""
    from heartless.learning import seal

    monkeypatch.setattr(seal, "LEDGER", tmp_path / "seal_ledger.jsonl")
