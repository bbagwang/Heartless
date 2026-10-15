import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# tests never touch the network: the orchestrator must not pull history from the public archive
os.environ["ARCHIVE_BACKFILL"] = "false"


@pytest.fixture(autouse=True, scope="session")
def _session_tempdir(tmp_path_factory):
    """Point tempfile.mkdtemp() at pytest's own base temp dir for the whole session.

    Many tests build stores and apps in tempfile.mkdtemp() and never remove them; a full run leaked thousands of
    /tmp/tmp* directories holding copied databases (19 GB after a day of research runs). Under pytest's basetemp
    they are removed with the run (pytest keeps only the last few basetemps), so nothing accumulates."""
    import tempfile

    base = str(tmp_path_factory.mktemp("mkdtemp"))
    old = tempfile.tempdir
    tempfile.tempdir = base
    try:
        yield base
    finally:
        tempfile.tempdir = old


@pytest.fixture(autouse=True)
def _sandboxed_seal_ledger(tmp_path, monkeypatch):
    """No test may append to the real docs/results/seal_ledger.jsonl: that file is the audit trail of deliberate
    unseals, and a stray line from a test run would read as an opened final exam. Every test gets its own ledger."""
    from heartless.learning import seal

    monkeypatch.setattr(seal, "LEDGER", tmp_path / "seal_ledger.jsonl")
