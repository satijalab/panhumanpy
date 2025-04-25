from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import umap as umap_learn

__all__ = ["embed", "UMAPConfig"]


@dataclass
class UMAPConfig:
    """Configuration for UMAP embeddings.

    Attributes:
        n_neighbors (int): Number of neighbors considered for each point. Default is 30.
        n_components (int): Number of dimensions in the output embedding. Default is 2.
        metric (str): Distance metric to use. Default is 'cosine'.
        min_dist (float): Minimum distance between points in the low-dimensional space. Default is 0.3.
        learning_rate (float): Learning rate for optimization. Default is 1.0.
        random_state (int): Seed used by the random number generator for reproducibility. Default is 42.
        spread (float): Effective scale of embedded points. Default is 1.0.
        init (str): Method used for initialization of the embedding. Default is 'spectral'.
    """

    n_neighbors: int = 30
    n_components: int = 2
    metric: str = "cosine"
    min_dist: float = 0.3
    learning_rate: float = 1.0
    random_state: int = 42
    spread: float = 1.0
    init: str = "spectral"


def embed(
    query: npt.NDArray,
    config: Optional[UMAPConfig] = None,
    **kwargs,
) -> npt.NDArray:
    """Compute a UMAP embedding for the given high-dimensional data.

    This function applies the UMAP algorithm to reduce the dimensionality of the input data.
    If no configuration is provided, it uses a default UMAPConfig instance. Additional keyword
    arguments can override the default configuration parameters.

    Args:
        query (numpy.ndarray): A NumPy array of shape (n_samples, n_features) representing the input data.
        config (Optional[UMAPConfig]): An optional configuration for UMAP parameters. Defaults to None,
            in which case a default UMAPConfig is used.
        **kwargs: Additional keyword arguments that override the configuration parameters.

    Returns:
        numpy.ndarray: A NumPy array representing the low-dimensional embedding of the input data.
    """

    if config is None:
        config = UMAPConfig()

    kwargs = dict(asdict(config), **kwargs)
    umap_model = umap_learn.UMAP(**kwargs)
    result = umap_model.fit_transform(query)

    assert isinstance(result, np.ndarray)

    return result
