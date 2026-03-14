"""Tests for literavore.config."""

from pathlib import Path

import yaml

from literavore.config import LiteravoreConfig, load_config


class TestDefaults:
    def test_default_config_has_no_conferences(self):
        config = LiteravoreConfig()
        assert config.conferences == []

    def test_default_pdf_settings(self):
        config = LiteravoreConfig()
        assert config.pdf.max_concurrent == 10
        assert config.pdf.keep_pdfs is False

    def test_default_storage_backend(self):
        config = LiteravoreConfig()
        assert config.storage.backend == "local"
        assert config.storage.data_dir == "data"

    def test_default_summary_model(self):
        config = LiteravoreConfig()
        assert config.summary.model == "gpt-4o-mini"

    def test_default_embedding_views(self):
        config = LiteravoreConfig()
        assert "title_abstract" in config.embedding.views


class TestLoadFromYaml:
    def test_load_from_yaml(self, tmp_path: Path):
        cfg = {
            "conferences": [{"name": "Test-2025", "year": 2025, "max_papers": 5}],
            "storage": {"data_dir": str(tmp_path / "data")},
        }
        yml = tmp_path / "config.yml"
        yml.write_text(yaml.dump(cfg))
        config = load_config(yml)
        assert len(config.conferences) == 1
        assert config.conferences[0].name == "Test-2025"

    def test_load_missing_file_gives_defaults(self, tmp_path: Path):
        config = load_config(tmp_path / "nonexistent.yml")
        assert config.conferences == []

    def test_load_empty_yaml(self, tmp_path: Path):
        yml = tmp_path / "empty.yml"
        yml.write_text("")
        config = load_config(yml)
        assert config.conferences == []

    def test_load_from_env_var(self, tmp_path: Path, monkeypatch):
        cfg = {"conferences": [{"name": "Env-2025", "year": 2025}]}
        yml = tmp_path / "env_config.yml"
        yml.write_text(yaml.dump(cfg))
        monkeypatch.setenv("LITERAVORE_CONFIG", str(yml))
        config = load_config()
        assert config.conferences[0].name == "Env-2025"


class TestEnvVarOverrides:
    def test_dev_mode_true(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("LITERAVORE_DEV_MODE", "1")
        config = load_config(tmp_path / "nope.yml")
        assert config.pdf.keep_pdfs is True

    def test_dev_mode_true_yes(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("LITERAVORE_DEV_MODE", "true")
        config = load_config(tmp_path / "nope.yml")
        assert config.pdf.keep_pdfs is True

    def test_dev_mode_false(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("LITERAVORE_DEV_MODE", "false")
        config = load_config(tmp_path / "nope.yml")
        assert config.pdf.keep_pdfs is False

    def test_data_dir_override(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("LITERAVORE_DATA_DIR", "/custom/data")
        config = load_config(tmp_path / "nope.yml")
        assert config.storage.data_dir == "/custom/data"

    def test_storage_backend_override(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("LITERAVORE_STORAGE_BACKEND", "s3")
        config = load_config(tmp_path / "nope.yml")
        assert config.storage.backend == "s3"

    def test_invalid_storage_backend_ignored(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("LITERAVORE_STORAGE_BACKEND", "gcs")
        config = load_config(tmp_path / "nope.yml")
        assert config.storage.backend == "local"
