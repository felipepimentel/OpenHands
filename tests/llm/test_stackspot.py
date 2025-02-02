from unittest.mock import Mock, patch

import pytest
import requests

from openhands.core.config import LLMConfig
from openhands.core.message import Message
from openhands.llm import StackSpotLLM


@pytest.fixture
def config():
    return LLMConfig(
        model="stackspot-test-model",
        api_key="test-api-key",
        base_url="https://api.stackspot.com",
        temperature=0.7,
        max_output_tokens=1000,
    )


@pytest.fixture
def llm(config):
    return StackSpotLLM(config)


def test_init_requires_api_key():
    config = LLMConfig(model="test-model")
    with pytest.raises(ValueError, match="API key is required"):
        StackSpotLLM(config)


def test_format_messages_for_llm(llm):
    # Test single message
    message = Message(role="user", content="Hello")
    formatted = llm.format_messages_for_llm(message)
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"

    # Test list of messages
    messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="Hi"),
    ]
    formatted = llm.format_messages_for_llm(messages)
    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[1]["role"] == "user"


def test_get_token_count(llm):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    count = llm.get_token_count(messages)
    assert isinstance(count, int)
    assert count > 0


@patch("requests.post")
def test_completion_success(mock_post, llm):
    # Mock response
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": "test-id",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 10},
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    # Test completion
    messages = [{"role": "user", "content": "Hi"}]
    response = llm.completion(messages)

    # Verify response
    assert response.id == "test-id"
    assert len(response.choices) == 1
    assert response.choices[0]["message"]["content"] == "Hello!"
    assert response.choices[0]["finish_reason"] == "stop"

    # Verify API call
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.stackspot.com/v1/chat/completions"
    assert "Authorization" in kwargs["headers"]
    assert kwargs["json"]["messages"] == messages


@patch("requests.post")
def test_completion_error(mock_post, llm):
    # Mock error response
    mock_post.side_effect = requests.exceptions.RequestException("API Error")

    # Test completion with error
    messages = [{"role": "user", "content": "Hi"}]
    with pytest.raises(requests.exceptions.RequestException):
        llm.completion(messages)


def test_str_representation(llm):
    assert str(llm) == f"StackSpotLLM(model={llm.config.model})"
    assert repr(llm) == str(llm)


@patch("requests.post")
def test_completion_with_optional_params(mock_post, llm):
    # Mock response
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": "test-id",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 10},
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    # Update config with optional parameters
    llm.config.top_p = 0.95

    # Test completion
    messages = [{"role": "user", "content": "Hi"}]
    response = llm.completion(messages)

    # Verify API call includes optional parameters
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["top_p"] == 0.95


def test_reset(llm):
    # Test that reset doesn't raise any errors
    llm.reset()
