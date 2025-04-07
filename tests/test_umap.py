import numpy as np
import pytest

import azimuthpy.umap as umap


def test_embed_default():
    """Test that embed returns a numpy array with the default 2 components."""
    np.random.seed(0)
    data = np.random.rand(10, 5)  # 10 samples, 5 features
    result = umap.embed(data)
    # Default UMAPConfig uses n_components=2.
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 2)


def test_embed_custom_config():
    """Test that a custom UMAPConfig with n_components=3 produces an embedding
    of shape (n_samples, 3).
    """
    np.random.seed(0)
    data = np.random.rand(20, 10)
    config = umap.UMAPConfig(n_components=3)
    result = umap.embed(data, config)
    assert isinstance(result, np.ndarray)
    assert result.shape == (20, 3)


def test_embed_kw_override():
    """Test that keyword arguments override the default configuration.

    Here, overriding n_components via kwargs should change the output embedding
    shape.
    """
    np.random.seed(0)
    data = np.random.rand(15, 8)
    # Override n_components to 4 even if default config is used.
    result = umap.embed(data, n_components=4)
    assert isinstance(result, np.ndarray)
    assert result.shape == (15, 4)


def test_invalid_input():
    """Test that providing a 1D array (invalid input) raises an exception."""
    data = np.array([1, 2, 3, 4])
    with pytest.raises(Exception):
        # UMAP expects a 2D array; this should raise an error.
        umap.embed(data)
