from __future__ import annotations

from pathlib import Path

import pytest

from plva_proxy import credentials


def test_env_file_value_reads_quoted_values_and_skips_other_lines(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nOTHER=1\nHAI_API_KEY="quoted-key"\n', "utf-8")

    assert credentials.env_file_value(env_file, "HAI_API_KEY") == "quoted-key"
    assert credentials.env_file_value(env_file, "MISSING") is None
    assert credentials.env_file_value(tmp_path / "absent.env", "HAI_API_KEY") is None

    env_file.write_text("HAI_API_KEY=\n", "utf-8")
    assert credentials.env_file_value(env_file, "HAI_API_KEY") is None


def test_resolve_provider_key_prefers_environment_then_holo_cli_then_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    holo_env = tmp_path / "holo.env"
    project_root = tmp_path / "project"
    project_root.mkdir()
    holo_env.write_text("HAI_API_KEY=from-holo-cli\n", encoding="utf-8")
    (project_root / ".env").write_text("HAI_API_KEY=from-project\n", encoding="utf-8")
    monkeypatch.setattr(credentials, "HOLO_USER_ENV", holo_env)
    monkeypatch.delenv("HAI_API_KEY", raising=False)

    assert credentials.resolve_provider_key(provider="hcompany", project_root=project_root) == (
        "from-holo-cli",
        "holo_cli",
    )

    monkeypatch.setenv("HAI_API_KEY", "from-shell")
    assert credentials.resolve_provider_key(provider="hcompany", project_root=project_root) == (
        "from-shell",
        "environment",
    )

    monkeypatch.delenv("HAI_API_KEY", raising=False)
    holo_env.write_text("\n", encoding="utf-8")
    assert credentials.resolve_provider_key(provider="hcompany", project_root=project_root) == (
        "from-project",
        "project",
    )


def test_credential_source_preference_limits_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    holo_env = tmp_path / "holo.env"
    project_root = tmp_path / "project"
    project_root.mkdir()
    holo_env.write_text("HAI_API_KEY=from-holo-cli\n", encoding="utf-8")
    (project_root / ".env").write_text("HAI_API_KEY=from-project\n", encoding="utf-8")
    monkeypatch.setattr(credentials, "HOLO_USER_ENV", holo_env)
    monkeypatch.delenv("HAI_API_KEY", raising=False)

    assert credentials.resolve_provider_key(
        provider="hcompany",
        source="project",
        project_root=project_root,
    ) == ("from-project", "project")
    assert credentials.resolve_provider_key(
        provider="hcompany",
        source="holo_cli",
        project_root=project_root,
    ) == ("from-holo-cli", "holo_cli")


def test_env_file_value_reads_export_and_quoted_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('export HAI_API_KEY="quoted-key"\n', "utf-8")

    assert credentials.env_file_value(env_file, "HAI_API_KEY") == "quoted-key"


def test_credential_status_reports_holo_cli_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    holo_env = tmp_path / "holo.env"
    profile = tmp_path / "profile.json"
    holo_env.write_text("HAI_API_KEY=from-holo-cli\n", encoding="utf-8")
    profile.write_text(
        '{"email":"user@example.com","key_label":"HoloDesktop CLI (test)"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(credentials, "HOLO_USER_ENV", holo_env)
    monkeypatch.setattr(credentials, "HOLO_PROFILE", profile)
    monkeypatch.delenv("HAI_API_KEY", raising=False)

    status = credentials.credential_status(provider="hcompany", project_root=tmp_path)

    assert status["configured"] is True
    assert status["source"] == "holo_cli"
    assert status["key_label"] == "HoloDesktop CLI (test)"
    assert status["account_email"] == "user@example.com"
    assert "from-holo-cli" not in str(status)


def test_inject_provider_keys_sets_canonical_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    holo_env = tmp_path / "holo.env"
    holo_env.write_text("HAI_API_KEY=from-holo-cli\n", encoding="utf-8")
    monkeypatch.setattr(credentials, "HOLO_USER_ENV", holo_env)
    environment: dict[str, str] = {}
    credentials.inject_provider_keys(environment, provider="hcompany", project_root=tmp_path)
    assert environment["HAI_API_KEY"] == "from-holo-cli"
