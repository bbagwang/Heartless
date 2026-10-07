"""Regression tests for heartless/config.py (fix round 16).

1. The shipped `.env.example` contains `TELEGRAM_OWNER_ID=` (blank, "leave empty to use the pairing flow").
   pydantic-settings passes "" to the `int | None` field, which used to raise ValidationError and abort every
   CLI command before logging was even set up. Both the dotenv path and the plain-environment path
   (docker `env_file:` / systemd `EnvironmentFile=`) must resolve a blank value to None.
2. The web dashboard (kill / close_all / go-live controls over plain HTTP) must bind to loopback by default.
"""
from pathlib import Path

import pytest

from heartless.config import Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # make sure the operator's shell does not leak these into the assertions below
    for k in ("TELEGRAM_OWNER_ID", "WEB_HOST", "HEARTLESS_MODE", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.delenv(k, raising=False)


def test_env_example_shipped_as_dotenv_loads(tmp_path):
    assert "TELEGRAM_OWNER_ID=" in ENV_EXAMPLE.read_text(encoding="utf-8"), ".env.example no longer ships the blank line"
    env = tmp_path / ".env"
    env.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.telegram_owner_id is None
    assert s.mode == "paper"
    assert s.telegram_bot_token == ""


def test_blank_owner_id_from_process_environment(monkeypatch):
    # docker-compose `env_file: .env` and systemd `EnvironmentFile=` export TELEGRAM_OWNER_ID="" into os.environ
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "")
    assert Settings(_env_file=None).telegram_owner_id is None
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "   ")
    assert Settings(_env_file=None).telegram_owner_id is None


def test_numeric_owner_id_still_parsed(monkeypatch):
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "42")
    assert Settings(_env_file=None).telegram_owner_id == 42
    assert Settings(_env_file=None, TELEGRAM_OWNER_ID=7).telegram_owner_id == 7
    assert Settings(_env_file=None, TELEGRAM_OWNER_ID="").telegram_owner_id is None
    assert Settings(_env_file=None, TELEGRAM_OWNER_ID=None).telegram_owner_id is None


def test_garbage_owner_id_still_rejected(monkeypatch):
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "not-a-number")
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_load_settings_from_repo_style_cwd(tmp_path, monkeypatch):
    # `heartless run` / `doctor` / `params` all start with load_settings() from the working directory
    (tmp_path / ".env").write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    s = load_settings(HEARTLESS_DATA_DIR=str(tmp_path / "data"))
    assert s.telegram_owner_id is None
    assert (tmp_path / "data").is_dir()


def test_web_host_defaults_to_loopback():
    s = Settings(_env_file=None)
    assert s.web_host == "127.0.0.1"
    assert s.web_port == 8080
    # explicit opt-in (e.g. inside a container behind the compose 127.0.0.1:8080 mapping) still works
    assert Settings(_env_file=None, WEB_HOST="0.0.0.0").web_host == "0.0.0.0"


def test_web_host_env_override(monkeypatch):
    monkeypatch.setenv("WEB_HOST", "0.0.0.0")
    assert Settings(_env_file=None).web_host == "0.0.0.0"
