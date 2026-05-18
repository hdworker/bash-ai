import os
import yaml
import pytest
import ai


class TestDefaultConfig:
    def test_default_config_has_required_keys(self):
        assert "api_key" in ai.DEFAULT_CONFIG
        assert "base_url" in ai.DEFAULT_CONFIG
        assert "model" in ai.DEFAULT_CONFIG
        assert "prompts" in ai.DEFAULT_CONFIG

    def test_default_model(self):
        assert ai.DEFAULT_CONFIG["model"] == "gpt-4o-mini"

    def test_default_api_key_empty(self):
        assert ai.DEFAULT_CONFIG["api_key"] == ""

    def test_default_base_url_empty(self):
        assert ai.DEFAULT_CONFIG["base_url"] == ""

    def test_default_prompts(self):
        prompts = ai.DEFAULT_CONFIG["prompts"]
        assert "chat" in prompts
        assert "cmd" in prompts
        assert "explain" in prompts
        assert "explain_text" in prompts

    def test_default_config_is_independent(self):
        config = ai.DEFAULT_CONFIG.copy()
        config["model"] = "changed"
        assert ai.DEFAULT_CONFIG["model"] != "changed"


class TestConfigPath:
    def test_config_path_uses_config_yaml(self):
        assert ai.CONFIG_PATH.endswith("config.yaml")

    def test_config_path_in_config_directory(self):
        assert "bashai" in ai.CONFIG_PATH
        assert ".config" in ai.CONFIG_PATH


class TestLoadConfig:
    def test_load_config_returns_dict(self, mock_config_path):
        mock_config_path.write_text(yaml.dump({"api_key": "key"}))
        ai._config = None
        config = ai.load_config()
        assert isinstance(config, dict)

    def test_load_config_uses_defaults_when_no_file(self, mock_config_path, monkeypatch):
        mock_config_path.write_text(yaml.dump({"api_key": "test-key"}))
        monkeypatch.setenv("BASHAI_API", "http://test-url")
        ai._config = None
        config = ai.load_config()
        assert config["model"] == ai.DEFAULT_CONFIG["model"]

    def test_load_config_reads_yaml_file(self, mock_config_path):
        mock_config_path.write_text(yaml.dump({
            "api_key": "test-key",
            "model": "gpt-4",
            "base_url": "http://test.com"
        }))
        ai._config = None
        config = ai.load_config()
        assert config["api_key"] == "test-key"
        assert config["model"] == "gpt-4"
        assert config["base_url"] == "http://test.com"

    def test_load_config_reads_partial_yaml(self, mock_config_path):
        mock_config_path.write_text(yaml.dump({
            "api_key": "key",
            "model": "gpt-4-turbo"
        }))
        ai._config = None
        config = ai.load_config()
        assert config["model"] == "gpt-4-turbo"

    def test_load_config_prompts_override(self, mock_config_path):
        mock_config_path.write_text(yaml.dump({
            "api_key": "key",
            "prompts": {
                "cmd": "Custom cmd prompt"
            }
        }))
        ai._config = None
        config = ai.load_config()
        assert config["prompts"]["cmd"] == "Custom cmd prompt"
        assert config["prompts"]["chat"] == ai.DEFAULT_CONFIG["prompts"]["chat"]

    def test_load_config_model_from_env(self, mock_config_path, monkeypatch):
        monkeypatch.setenv("BASHAI_MODEL", "env-model")
        monkeypatch.setenv("BASHAI_API", "http://env-url")
        mock_config_path.write_text(yaml.dump({"api_key": "key", "model": "file-model"}))
        ai._config = None
        config = ai.load_config()
        assert config["model"] == "env-model"
        assert config["base_url"] == "http://env-url"

    def test_load_config_empty_yaml_returns_defaults(self, mock_config_path, monkeypatch):
        mock_config_path.write_text(yaml.dump({"api_key": "test-key"}))
        monkeypatch.setenv("BASHAI_API", "http://test-url")
        ai._config = None
        config = ai.load_config()
        assert config["api_key"] == "test-key"
        assert config["model"] == ai.DEFAULT_CONFIG["model"]


class TestGetConfig:
    def test_get_config_caches_result(self, mock_config_path):
        mock_config_path.write_text(yaml.dump({"api_key": "key"}))
        ai._config = None
        config1 = ai.get_config()
        config2 = ai.get_config()
        assert config1 is config2

    def test_get_config_returns_same_instance(self, mock_config_path):
        mock_config_path.write_text(yaml.dump({"api_key": "key"}))
        ai._config = None
        assert ai.get_config() is ai.get_config()
