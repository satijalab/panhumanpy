"""
Tests for model download and cache behaviour in load_inference_model.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from panhumanpy.ANNotate_tools import InferenceTools, CACHE_DIR, MODEL_URLS
from panhumanpy.ANNotate_tools import _available_model_versions


def _make_inference_tools(version):
    """Helper: instantiate InferenceTools without triggering autoload."""
    return InferenceTools.__new__(InferenceTools)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR to a temporary directory for all tests."""
    monkeypatch.setattr(
        "panhumanpy.ANNotate_tools.CACHE_DIR", tmp_path
    )
    return tmp_path


def test_model_urls_exist_for_all_versions():
    """Every version in _tools has a corresponding entry in MODEL_URLS."""
    from panhumanpy.ANNotate_tools import _available_model_versions
    versions = _available_model_versions()
    for v in versions:
        assert v in MODEL_URLS, (
            f"MODEL_URLS missing entry for version '{v}'"
        )


def test_download_triggered_on_first_use(tmp_path, monkeypatch):
    """load_inference_model downloads the file when cache is empty."""
    monkeypatch.setattr("panhumanpy.ANNotate_tools.CACHE_DIR", tmp_path)

    fake_model = MagicMock()

    def fake_urlretrieve(url, dest):
        Path(dest).touch()

    with patch("panhumanpy.ANNotate_tools.urllib.request.urlretrieve",
               side_effect=fake_urlretrieve) as mock_dl, \
         patch("panhumanpy.ANNotate_tools.load_model",
               return_value=fake_model):

        it = InferenceTools.__new__(InferenceTools)
        it._model_version = "v0"
        it._inference_model_filename = "inference_model.keras"
        it.load_inference_model()

        mock_dl.assert_called_once()
        call_url = mock_dl.call_args[0][0]
        assert call_url == MODEL_URLS["v0"], (
            "urlretrieve called with wrong URL"
        )


def test_no_download_on_cache_hit(tmp_path, monkeypatch):
    """load_inference_model skips download when model already cached."""
    monkeypatch.setattr("panhumanpy.ANNotate_tools.CACHE_DIR", tmp_path)

    cache_path = tmp_path / "v0" / "inference_model" / "inference_model.keras"
    cache_path.parent.mkdir(parents=True)
    cache_path.touch()

    fake_model = MagicMock()

    with patch("panhumanpy.ANNotate_tools.urllib.request.urlretrieve") as mock_dl, \
         patch("panhumanpy.ANNotate_tools.load_model",
               return_value=fake_model):

        it = InferenceTools.__new__(InferenceTools)
        it._model_version = "v0"
        it._inference_model_filename = "inference_model.keras"
        it.load_inference_model()

        mock_dl.assert_not_called()


def test_invalid_version_raises(tmp_path, monkeypatch):
    """load_inference_model raises ValueError for unknown version."""
    monkeypatch.setattr("panhumanpy.ANNotate_tools.CACHE_DIR", tmp_path)

    it = InferenceTools.__new__(InferenceTools)
    it._model_version = "v999"
    it._inference_model_filename = "inference_model.keras"

    with pytest.raises(ValueError, match="No download URL found"):
        it.load_inference_model()


def test_model_saved_with_correct_filename(tmp_path, monkeypatch):
    """Downloaded model is saved as inference_model.keras regardless of URL filename."""
    monkeypatch.setattr("panhumanpy.ANNotate_tools.CACHE_DIR", tmp_path)

    fake_model = MagicMock()
    saved_path = []

    def fake_urlretrieve(url, dest):
        saved_path.append(Path(dest))
        Path(dest).touch()

    with patch("panhumanpy.ANNotate_tools.urllib.request.urlretrieve",
               side_effect=fake_urlretrieve), \
         patch("panhumanpy.ANNotate_tools.load_model",
               return_value=fake_model):

        it = InferenceTools.__new__(InferenceTools)
        it._model_version = "v0"
        it._inference_model_filename = "inference_model.keras"
        it.load_inference_model()

        assert saved_path[0].name == "inference_model.keras", (
            "Model should be saved as inference_model.keras regardless of URL filename"
        )