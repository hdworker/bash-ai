import os
import openai
import pytest
import ai


class TestGetCmd:
    def test_get_cmd_returns_stripped_cmd(self, mock_client):
        client = mock_client("  ls -la  ")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmd = ai.get_cmd(client, "list files", "gpt-4")
        assert cmd == "ls -la"

    def test_get_cmd_removes_backticks(self, mock_client):
        client = mock_client("```bash\nls -la\n```")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmd = ai.get_cmd(client, "list files", "gpt-4")
        assert cmd == "ls -la"

    def test_get_cmd_handles_none_response(self, mock_client):
        client = mock_client(None)
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmd = ai.get_cmd(client, "list files", "gpt-4")
        assert cmd == ""

    def test_get_cmd_removes_only_opening_backticks(self, mock_client):
        client = mock_client("```bash\necho hello")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmd = ai.get_cmd(client, "print hello", "gpt-4")
        assert cmd == "echo hello"

    def test_get_cmd_removes_only_closing_backticks(self, mock_client):
        client = mock_client("echo hello\n```")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmd = ai.get_cmd(client, "print hello", "gpt-4")
        assert cmd == "echo hello"


class TestGetCmdList:
    def test_get_cmd_list_returns_multiple(self, mock_client):
        client = mock_client("cmd1", choices=["cmd1", "cmd2", "cmd3"])
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmds = ai.get_cmd_list(client, "test", "gpt-4", n=3)
        assert len(cmds) == 3

    def test_get_cmd_list_handles_none_content(self, mock_client):
        client = mock_client("cmd1", choices=[None, "cmd2", None])
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmds = ai.get_cmd_list(client, "test", "gpt-4", n=3)
        assert isinstance(cmds, list)

    def test_get_cmd_list_removes_duplicates(self, mock_client):
        client = mock_client("cmd1", choices=["cmd1", "cmd1", "cmd2"])
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmds = ai.get_cmd_list(client, "test", "gpt-4", n=3)
        assert len(set(cmds)) == len(cmds)

    def test_get_cmd_list_strips_backticks(self, mock_client):
        client = mock_client("cmd1", choices=["```bash\nls\n```", "echo test"])
        os.environ["NOCACHE"] = "1"
        ai._config = None
        cmds = ai.get_cmd_list(client, "test", "gpt-4", n=2)
        assert "ls" in cmds
        assert "echo test" in cmds


class TestChat:
    def test_chat_returns_content(self, mock_client, mock_cache_folder):
        client = mock_client("test response")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        response = ai.chat(client, "hello", "gpt-4")
        assert response == "test response"

    def test_chat_returns_stripped_content(self, mock_client, mock_cache_folder):
        client = mock_client("  stripped  ")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        response = ai.chat(client, "hello", "gpt-4")
        assert response == "stripped"

    def test_chat_handles_none_content(self, mock_client, mock_cache_folder):
        client = mock_client(None)
        os.environ["NOCACHE"] = "1"
        ai._config = None
        response = ai.chat(client, "hello", "gpt-4")
        assert response == ""

    def test_chat_saves_history(self, mock_client, mock_cache_folder):
        client = mock_client("response")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        ai.chat(client, "hello", "gpt-4")
        history = ai.load_history()
        assert len(history) >= 2


class TestGetNeededContext:
    def test_get_needed_context_returns_int(self, mock_client):
        client = mock_client("2")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        result = ai.get_needed_context("test cmd", client, "gpt-4")
        assert isinstance(result, int)

    def test_get_needed_context_returns_negative_one_on_error(self, mock_client, capsys):
        client = mock_client("invalid")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        result = ai.get_needed_context("test cmd", client, "gpt-4")
        assert result == -1

    def test_get_needed_context_returns_negative_one(self, mock_client):
        client = mock_client("-1")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        result = ai.get_needed_context("test cmd", client, "gpt-4")
        assert result == -1


class TestGetExplanation:
    def test_get_explanation_returns_text(self, mock_client):
        client = mock_client("This is an explanation")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        result = ai.get_explaination(client, "ls -la", "gpt-4")
        assert result == "This is an explanation"

    def test_get_explanation_removes_double_newlines(self, mock_client):
        client = mock_client("line1\n\nline2\n\nline3")
        os.environ["NOCACHE"] = "1"
        ai._config = None
        result = ai.get_explaination(client, "ls", "gpt-4")
        assert "\n\n" not in result


class TestGenerateContextHelp:
    def test_returns_string(self):
        result = ai.generate_context_help()
        assert isinstance(result, str)

    def test_contains_context_count(self):
        result = ai.generate_context_help()
        count = len(ai.CONTEXT)
        for i in range(count):
            assert str(i) in result


@pytest.mark.integration
class TestRealAPI:
    def test_echo_current_time(self, tmp_path, mocker):
        mocker.patch("ai.CACHE_FOLDER", str(tmp_path))
        os.environ["NOCACHE"] = "1"
        ai._config = None

        config = ai.get_config()
        client = openai.OpenAI(api_key=config["api_key"], base_url=config["base_url"] if config["base_url"] else None)
        model = config["model"]

        cmd = ai.get_cmd(client, "how to echo current time", model, context_prompt="")

        assert isinstance(cmd, str)
        if len(cmd) > 0:
            assert any(kw in cmd.lower() for kw in ["date", "time", "echo"])
