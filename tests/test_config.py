"""Consistency guards: config and .env.example must not drift apart
(BRIEF ground rule 4: ".env.example kept current")."""

from pathlib import Path

from src.config import Settings


def test_every_setting_is_documented_in_env_example():
    env_example = Path(".env.example").read_text(encoding="utf-8")
    documented = {
        line.split("=")[0].strip()
        for line in env_example.splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }
    missing = [
        name.upper()
        for name in Settings.model_fields
        if name.upper() not in documented
    ]
    assert not missing, f".env.example is missing settings: {missing}"


def test_env_example_has_no_unknown_settings():
    env_example = Path(".env.example").read_text(encoding="utf-8")
    documented = {
        line.split("=")[0].strip()
        for line in env_example.splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }
    known = {name.upper() for name in Settings.model_fields}
    unknown = documented - known
    assert not unknown, f".env.example documents settings that no longer exist: {unknown}"


def test_no_secret_values_in_env_example():
    env_example = Path(".env.example").read_text(encoding="utf-8")
    for line in env_example.splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            value = line.split("=", 1)[1].strip()
            assert value == "", "ANTHROPIC_API_KEY must be empty in .env.example"
