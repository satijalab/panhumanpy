from __future__ import annotations

from typing import Generic, Optional, Protocol, TypeVar

import anndata as ad
import numpy.typing as npt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


class ModelRunner(Protocol):
    """Protocol defining the interface for the package's main prediction engine.

    Implementations should provide methods for aligning features, normalizing
    data, generating embeddings, and producing cell type predictions.
    """

    def align_features(
        self, query: csr_matrix, features: list[str]
    ) -> tuple[csr_matrix, list[str]]:
        ...

    def normalize(self, query: csr_matrix) -> csr_matrix:
        ...

    def embed(self, query: csr_matrix) -> npt.NDArray:
        ...

    def annotate(
        self, query: csr_matrix
    ) -> tuple[list[npt.NDArray], list[LabelEncoder]]:
        ...


class AnnotationHandler(Protocol):
    """Protocol defining the interface for annotation handlers.

    Implementations should take in softmax probabilities, label encoders,
    and cell identifiers.
    """

    @classmethod
    def build(
        cls,
        softmax_probabilities: list[npt.NDArray],
        encoders: list[LabelEncoder],
        cells: list[str],
    ) -> AnnotationHandler:
        ...


AnnotationType = TypeVar("AnnotationType", bound=AnnotationHandler, contravariant=True)


class AnnDataHandler(Protocol, Generic[AnnotationType]):
    """Generic protocol defining the interface for serializing final data
    structures from intermediate annotations.

    Implementations should define how to assemble counts, embeddings, and
    cell type annotations into the desired output format.
    """

    def update(
        self,
        query: ad.AnnData,
        azimuth_embeddings: Optional[npt.NDArray],
        umap_embeddings: Optional[npt.NDArray],
        azimuth_annotations: Optional[AnnotationType],
    ) -> ad.AnnData:
        ...
