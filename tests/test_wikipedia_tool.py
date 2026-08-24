import unittest.mock as mock
import pytest
from utils.tools import search_wikipedia


def test_search_wikipedia_ddgs_success():
    mock_results = [
        {
            "title": "Transformer (deep learning architecture)",
            "href": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
            "body": "In deep learning, the transformer is a family of artificial neural network architectures based on self-attention.",
        }
    ]
    with mock.patch("ddgs.DDGS") as mock_ddgs_cls:
        mock_instance = mock.MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.text.return_value = mock_results
        mock_ddgs_cls.return_value = mock_instance

        res = search_wikipedia("Transformer architecture")
        assert "Transformer (deep learning architecture)" in res
        assert "self-attention" in res


def test_search_wikipedia_fallback_on_empty():
    with mock.patch("ddgs.DDGS") as mock_ddgs_cls:
        mock_instance = mock.MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.text.return_value = []
        mock_ddgs_cls.return_value = mock_instance

        with mock.patch("urllib.request.urlopen", side_effect=Exception("API blocked")):
            res = search_wikipedia("nonexistent_query_xyz_123")
            assert "No Wikipedia reference found" in res
