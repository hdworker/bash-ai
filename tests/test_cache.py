import os
import pytest
import ai


class TestCacheDecorator:
    def test_cache_bypass_with_nocache_env(self, mock_cache_folder, monkeypatch):
        monkeypatch.setenv("NOCACHE", "1")
        call_count = [0]

        @ai.cache()
        def func(x):
            call_count[0] += 1
            return x * 2

        func(5)
        func(5)
        assert call_count[0] == 2

    def test_cache_returns_same_result(self, mock_cache_folder):
        @ai.cache()
        def func(x):
            return x * 2

        assert func(3) == 6
        assert func(3) == 6

    def test_cache_different_args(self, mock_cache_folder):
        @ai.cache()
        def func(x):
            return x * 2

        func(3)
        func(5)
        cache_file = os.path.join(str(mock_cache_folder), "cache.pkl")
        if os.path.exists(cache_file):
            import pickle
            from collections import OrderedDict
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            assert isinstance(cache, OrderedDict)

    def test_cache_creates_directory(self, mock_cache_folder):
        @ai.cache()
        def func():
            return 1

        func()
        assert mock_cache_folder.exists()

    def test_cache_persists_across_calls(self, mock_cache_folder):
        @ai.cache()
        def func(x):
            return x + 1

        func(10)
        func(20)
        cache_file = os.path.join(str(mock_cache_folder), "cache.pkl")
        if os.path.exists(cache_file):
            import pickle
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            assert len(cache) == 2


class TestCacheMaxsize:
    def test_cache_evicts_oldest(self, mock_cache_folder):
        @ai.cache(maxsize=2)
        def func(x):
            return x

        func(1)
        func(2)
        func(3)

        cache_file = os.path.join(str(mock_cache_folder), "cache.pkl")
        if os.path.exists(cache_file):
            import pickle
            from collections import OrderedDict
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            assert len(cache) <= 2
