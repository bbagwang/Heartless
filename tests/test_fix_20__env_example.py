"""Regression tests for `.env.example` (fix round 20).

README step 2 and the Docker section tell operators to `cp .env.example .env` and only fill in the keys.
`.env.example` used to ship a blank `TELEGRAM_OWNER_ID=` ("leave empty to use the pairing flow") although
`Settings.telegram_owner_id` is `int | None`; the empty string reached the int validator and `load_settings()`
raised, so every CLI subcommand (`doctor`, `params`, `run`, ...) and every compose / systemd start crashed before
logging was set up. The example file must therefore never ship a blank assignment for a non-string field, and a
verbatim copy must always construct a `Settings`.
"""
import re
from pathlib import Path

import pytest
from dotenv import dotenv_values

from heartless.config import Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"

# Settings keys that an operator might also have exported in their shell; they must not leak into the assertions.
_SHELL_KEYS = ("TELEGRAM_OWNER_ID", "TELEGRAM_BOT_TOKEN", "WEB_HOST", "WEB_PORT", "HEARTLESS_MODE",
               "BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_TESTNET", "ANTHROPIC_API_KEY", "TIMEZONE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _SHELL_KEYS:
        monkeypatch.delenv(k, raising=False)


def _example_text() -> str:
    return ENV_EXAMPLE.read_text(encoding="utf-8")


def _active_values() -> dict[str, str | None]:
    """Keys that a verbatim copy of .env.example actually assigns (comments are skipped, like pydantic-settings)."""
    return dotenv_values(ENV_EXAMPLE, encoding="utf-8")


def _aliases() -> dict[str, object]:
    """env alias -> field annotation for every Settings field."""
    out = {}
    for name, field in Settings.model_fields.items():
        out[field.alias or name] = field.annotation
    return out


def test_env_example_exists_and_is_documented_entrypoint():
    assert ENV_EXAMPLE.is_file()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "cp .env.example .env" in readme  # the operator flow under test


def test_owner_id_is_not_shipped_as_blank_assignment():
    active = _active_values()
    # A commented-out example line is fine (it is documentation), an active blank assignment is not.
    assert "TELEGRAM_OWNER_ID" not in active, ".env.example ships an active TELEGRAM_OWNER_ID assignment again"
    # The variable must still be documented so operators know how to pin the owner.
    assert "TELEGRAM_OWNER_ID" in _example_text()


def test_no_blank_assignment_for_non_string_fields():
    """Generalisation of the owner-id crash: every active key in .env.example that maps onto a non-`str` Settings
    field must carry a value; an empty value would be handed to the int/bool/float validator and abort startup."""
    aliases = _aliases()
    for key, value in _active_values().items():
        assert key in aliases, f"{key} in .env.example is not a Settings field (extra='ignore' would drop it silently)"
        if aliases[key] is not str:
            assert value not in (None, ""), f"{key} is a {aliases[key]} field but .env.example ships it blank"


def test_verbatim_copy_constructs_settings(tmp_path):
    env = tmp_path / ".env"
    env.write_text(_example_text(), encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.telegram_owner_id is None  # pairing flow stays active
    assert s.mode == "paper"
    assert s.binance_api_key == ""
    assert s.binance_api_secret == ""
    assert s.binance_testnet is False
    assert s.telegram_bot_token == ""
    assert s.anthropic_api_key == ""
    assert s.web_port == 8080
    assert s.timezone == "Asia/Seoul"
    assert s.live_capable is False


def test_load_settings_from_copied_example_in_cwd(tmp_path, monkeypatch):
    """`heartless doctor` / `params` / `run` all call load_settings() first, with the .env found in the cwd."""
    (tmp_path / ".env").write_text(_example_text(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    s = load_settings(HEARTLESS_DATA_DIR=str(tmp_path / "data"))
    assert s.telegram_owner_id is None
    assert (tmp_path / "data").is_dir()


def test_uncommenting_owner_id_pins_an_integer_owner(tmp_path):
    """The documented way to pin the owner (uncomment the example line and put a real id) yields an int."""
    text = _example_text()
    pinned, n = re.subn(r"^#\s*TELEGRAM_OWNER_ID=.*$", "TELEGRAM_OWNER_ID=987654321", text, flags=re.MULTILINE)
    assert n == 1, "expected exactly one commented-out TELEGRAM_OWNER_ID example line"
    env = tmp_path / ".env"
    env.write_text(pinned, encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.telegram_owner_id == 987654321
    assert isinstance(s.telegram_owner_id, int)


def test_example_contains_no_real_secrets():
    """The example is committed; credential-looking values must never be present (only blank/placeholder)."""
    for key in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY"):
        assert _active_values().get(key, "") in ("", None), f"{key} must ship blank in .env.example"
    # Telegram bot tokens look like `<digits>:<35 chars>`; none may appear anywhere in the file, even in comments.
    assert not re.search(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b", _example_text())
