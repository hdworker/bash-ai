import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai as ai_module


@pytest.fixture(autouse=True)
def reset_module():
    ai_module._config = None
    yield
    ai_module._config = None


@pytest.fixture
def mock_config_path(tmp_path):
    config_file = tmp_path / "config.yaml"
    original = ai_module.CONFIG_PATH
    ai_module.CONFIG_PATH = str(config_file)
    yield config_file
    ai_module.CONFIG_PATH = original


@pytest.fixture
def mock_cache_folder(tmp_path):
    original = ai_module.CACHE_FOLDER
    ai_module.CACHE_FOLDER = str(tmp_path)
    yield tmp_path
    ai_module.CACHE_FOLDER = original


@pytest.fixture
def mock_client():
    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)

    class MockResponse:
        def __init__(self, content, choices=None):
            if choices is None:
                self.choices = [MockChoice(content)]
            else:
                self.choices = [MockChoice(c) for c in choices]

    class MockCreate:
        def __init__(self, content, choices=None):
            self._content = content
            self._choices = choices

        def __call__(self, **kwargs):
            return MockResponse(self._content, self._choices)

    class MockCompletions:
        def __init__(self, content, choices=None):
            self.create = MockCreate(content, choices)

    class MockChat:
        def __init__(self, content, choices=None):
            self.completions = MockCompletions(content, choices)

    class MockClient:
        def __init__(self, content, choices=None):
            self.chat = MockChat(content, choices)

    return MockClient
