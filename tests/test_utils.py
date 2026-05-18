import os
import pytest
import ai


class TestHighlight:
    def test_highlight_returns_string(self):
        result = ai.highlight("ls -la", "ls lists files in -la format")
        assert isinstance(result, str)

    def test_highlight_preserves_explanation(self):
        cmd = "ls -la"
        explanation = "Lists files"
        result = ai.highlight(cmd, explanation)
        assert "Lists" in result or "files" in result

    def test_highlight_with_empty_cmd(self):
        result = ai.highlight("", "Some explanation")
        assert result == "Some explanation"

    def test_highlight_with_empty_explanation(self):
        result = ai.highlight("ls -la", "")
        assert result == ""


class TestSquareText:
    def test_square_text_returns_string(self):
        try:
            result = ai.square_text("test")
            assert isinstance(result, str)
        except OSError:
            pytest.skip("No terminal available")

    def test_square_text_contains_borders(self):
        try:
            result = ai.square_text("test")
            assert "|" in result
        except OSError:
            pytest.skip("No terminal available")

    def test_square_text_contains_dashes(self):
        try:
            result = ai.square_text("test")
            assert "-" in result
        except OSError:
            pytest.skip("No terminal available")

    def test_square_text_multiline(self):
        try:
            result = ai.square_text("line1\nline2\nline3")
            assert "line1" in result
            assert "line2" in result
            assert "line3" in result
        except OSError:
            pytest.skip("No terminal available")


class TestContextFunctions:
    def test_get_context_files_returns_string(self):
        result = ai.get_context_files()
        assert isinstance(result, str)

    def test_get_context_process_list_returns_string(self):
        result = ai.get_context_process_list()
        assert isinstance(result, str)
        assert "processes" in result.lower()

    def test_get_context_users_returns_string(self):
        result = ai.get_context_users()
        assert isinstance(result, str)
        assert "users" in result.lower()

    def test_get_context_groups_returns_string(self):
        result = ai.get_context_groups()
        assert isinstance(result, str)
        assert "groups" in result.lower()

    def test_get_context_network_interfaces_returns_string(self):
        result = ai.get_context_network_interfaces()
        assert isinstance(result, str)
        assert "interfaces" in result.lower()

    def test_get_context_network_routes_returns_string(self):
        result = ai.get_context_network_routes()
        assert isinstance(result, str)
        assert "routes" in result.lower()


class TestContextList:
    def test_context_is_list(self):
        assert isinstance(ai.CONTEXT, list)

    def test_context_entries_have_name_and_function(self):
        for entry in ai.CONTEXT:
            assert "name" in entry
            assert "function" in entry
            assert callable(entry["function"])

    def test_context_function_env_excluded(self):
        names = [entry["name"] for entry in ai.CONTEXT]
        assert "List of environment variables" not in names


class TestSignalHandler:
    def test_signal_handler_exits(self, mocker):
        mock_exit = mocker.patch("sys.exit")
        ai.signal_handler(None, None)
        mock_exit.assert_called_once_with(0)
