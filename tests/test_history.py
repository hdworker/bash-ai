import os
import pickle
import pytest
import ai


class TestLoadHistory:
    def test_load_history_returns_empty_list_when_no_file(self, mock_cache_folder):
        history = ai.load_history()
        assert history == []

    def test_load_history_returns_existing_history(self, mock_cache_folder):
        history_file = os.path.join(str(mock_cache_folder), "chat_history")
        test_history = [{"role": "user", "content": "test"}]
        with open(history_file, "wb") as f:
            pickle.dump(test_history, f)
        history = ai.load_history()
        assert history == test_history

    def test_load_history_creates_directory(self, mock_cache_folder):
        ai.load_history()
        assert mock_cache_folder.exists()

    def test_load_history_handles_corrupted_file(self, mock_cache_folder):
        history_file = os.path.join(str(mock_cache_folder), "chat_history")
        with open(history_file, "wb") as f:
            f.write(b"not valid pickle")
        with pytest.raises(Exception):
            ai.load_history()


class TestSaveHistory:
    def test_save_history_writes_file(self, mock_cache_folder):
        history = [{"role": "user", "content": "hello"}]
        ai.save_history(history)
        history_file = os.path.join(str(mock_cache_folder), "chat_history")
        assert os.path.exists(history_file)

    def test_save_history_preserves_data(self, mock_cache_folder):
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        ai.save_history(history)
        loaded = ai.load_history()
        assert loaded == history

    def test_save_history_applies_limit(self, mock_cache_folder):
        history = [{"role": "user", "content": str(i)} for i in range(60)]
        ai.save_history(history, limit=50)
        loaded = ai.load_history()
        assert len(loaded) == 50
        assert loaded[0]["content"] == "10"

    def test_save_history_creates_directory(self, mock_cache_folder):
        ai.save_history([])
        assert mock_cache_folder.exists()


class TestCleanHistory:
    def test_clean_history_removes_file(self, mock_cache_folder):
        history_file = os.path.join(str(mock_cache_folder), "chat_history")
        with open(history_file, "wb") as f:
            pickle.dump([], f)
        ai.clean_history()
        assert not os.path.exists(history_file)

    def test_clean_history_no_file_no_error(self, mock_cache_folder):
        ai.clean_history()
        assert not os.path.exists(os.path.join(str(mock_cache_folder), "chat_history"))

    def test_clean_history_creates_directory(self, mock_cache_folder):
        ai.clean_history()
        assert mock_cache_folder.exists()
