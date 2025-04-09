from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Optional, Type, TypeVar

import anndata as ad
from scipy.sparse import csr_matrix

import azimuthpy.umap as umap
from azimuthpy.datatypes import AnnDataHandler, AnnotationType, ModelRunner

OutputHandler = TypeVar("OutputType", bound=AnnDataHandler)


@dataclass
class Azimuth(Generic[AnnotationType, OutputHandler]):
    """Main entrypoint for running analysis workflows on AnnData inputs
    containing scRNA-seq data.

    An `Azimuth` instance delegates all the analysis logic to it's composite
    `model_runner`, `annotation_type` and `output_handler` attributes.

    Attributes:
        model (ModelRunner):
            The prediction engine responsible for aligning input features to the
            reference panel, normalizing counts, generating low-dimensional
            embeddings, and producing softmax-based cell type probabilities.
        annotation_type (Type[AnnotationType]):
            The handler class used to construct hierarchical annotation objects
            from model-generated softmax outputs and
            corresponding label encoders.
        output (OutputHandler):
            The AnnData handler implementation that integrates computed
            embeddings, UMAP projections, and cell type annotations back into
            the provided AnnData object.
        azimuth_embeddings (bool):
            Flag indicating whether or not to output Azimuth embeddings.
            If `umap_embeddings` is `True` azimuth embeddings will always be
            calculated. Defaults to `True`.
        umap_embeddings (bool):
            Flag indicating whether or not to calculate umap embeddings.
            Defaults to True.
        umap_config (umap.UMAPConfig):
            Config for the umap embedding.
        azimuth_annotations (bool):
            Flag indicating whether or not to calculate Azimuth annotations.
            Defaults to True.
    """

    model: ModelRunner
    annotation_type: Type[AnnotationType]
    output: OutputHandler

    azimuth_embeddings: bool = True
    umap_embeddings: bool = True
    umap_config: umap.UMAPConfig = field(default_factory=umap.UMAPConfig)

    azimuth_annotations: bool = True

    def run(
        self,
        query: ad.AnnData,
        cell_col: Optional[str] = None,
        feature_col: Optional[str] = None,
    ) -> ad.AnnData:
        """Run the analysis workflow on the provided AnnData object.

        The method performs the following steps:
          1. Convert the count matrix to a compressed sparse row matrix.
          2. Retrieve the list of cells and features (using provided column
          names if available).
          3. Align the features and normalize the counts.
          4. Compute embeddings (optional).
          5. Compute annotations (optional).
          6. Update the AnnData object with the computed results.

        Args:
            adata (ad.AnnData):
                The query object.
            cell_col (Optional[str]):
                Column in adata.obs to use for cell names. If None, use default.
            feature_col (Optional[str]):
                Column in adata.obs to use for feature names. If None,
                use default.

        Returns:
            ad.AnnData:
            The updated AnnData object with new embeddings and annotations.
        """

        # Make sure that `counts` is sparse.
        counts = csr_matrix(query.X)
        # Extract cell and feature names using the specified columns.
        cells = self.get_cells(query, cell_col)
        features = self.get_features(query, feature_col)

        # Align `counts` with the expected feature panel and then normalize.
        aligned_counts, _ = self.model.align_features(counts, features)
        normalized_counts = self.model.normalize(aligned_counts)

        azimuth_embeddings = None
        umap_embeddings = None
        # Compute Azimuth embeddings if either azimuth or umap embeddings
        # are enabled.
        if self.azimuth_embeddings or self.umap_embeddings:
            azimuth_embeddings = self.model.embed(normalized_counts)
            # If umap embeddings are enabled, compute them as well.
            if self.umap_embeddings:
                umap_embeddings = umap.embed(azimuth_embeddings, self.umap_config)

        azimuth_annotations = None
        # Compute annotations if enabled.
        if self.azimuth_annotations:
            softmax_probabilities, encoders = self.model.annotate(normalized_counts)
            azimuth_annotations = self.annotation_type.build(
                softmax_probabilities=softmax_probabilities,
                encoders=encoders,
                cells=cells,
            )

        # Update the original AnnData object with the computed embeddings
        # and annotations.
        return self.output.update(
            adata=query,
            # It's possible to output umap embeddings without the underlying
            # azimuth embeddings.
            azimuth_embeddings=azimuth_embeddings if self.azimuth_embeddings else None,
            umap_embeddings=umap_embeddings,
            azimuth_annotations=azimuth_annotations,
        )

    @staticmethod
    def get_cells(adata: ad.AnnData, cell_col: str | None) -> list[str]:
        """Retrieve a list of cell identifiers from the AnnData object.

        Args:
            adata (ad.AnnData):
                The AnnData object containing cell data.
            cell_col (Optional[str]):
                Column name to extract cell identifiers. If None, use the
                default observation names.

        Returns:
            list[str]: A list of cell names.
        """

        if cell_col is None:
            return adata.obs_names.tolist()

        return adata.obs[cell_col].tolist()

    @staticmethod
    def get_features(adata: ad.AnnData, feature_col: str | None) -> list[str]:
        """Retrieve a list of feature identifiers from the AnnData object.

        Args:
            adata (ad.AnnData):
                The AnnData object containing feature data.
            feature_col (Optional[str]):
                Column name to extract feature identifiers. If None, use the
                default observation names.

        Returns:
            list[str]: A list of feature names.
        """

        if feature_col is None:
            return adata.obs_names.tolist()

        return adata.obs[feature_col].tolist()
