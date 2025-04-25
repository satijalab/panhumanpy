from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import anndata as ad
import numpy as np

from azimuthpy.cell_type_hierarchy import CellTypeHierarchy
from azimuthpy.datatypes import AnnDataHandler
from azimuthpy.panhuman.postprocessing.label_correction import LabelCorrections


@dataclass
class PanHumanAnnotator(AnnDataHandler[CellTypeHierarchy]):
    """Provides a convenient way to add Azimuth-specific embeddings and cell
    type annotations to an AnnData object, tailored for a panhuman
    analysis context.

    Attributes:
        label_resolution (Literal["fine", "medium"]):
            Specifies the resolution for label corrections.
    """

    label_resolution: Literal["fine", "medium"]

    def update(
        self,
        adata: ad.AnnData,
        azimuth_embeddings: Optional[np.ndarray],
        umap_embeddings: Optional[np.ndarray],
        azimuth_annotations: Optional[CellTypeHierarchy],
    ) -> ad.AnnData:
        """Update the given `anndata.AnnData` object with Azimuth embeddings and
        annotations.

        Embeddings are stored in the `.obsm` attribute and annotations are
        assigned to the `.obs` attribute with specific keys for different
        label levels and correction.

        Args:
            adata (ad.AnnData):
                The AnnData object to update.
            azimuth_embeddings (Optional[np.ndarray]):
                Array of Azimuth embeddings; stored as "X_azimuth" if provided.
            umap_embeddings (Optional[np.ndarray]):
                Array of UMAP embeddings based on Azimuth embeddings; stored
                as "X_azimuth-umap" if provided.
            azimuth_annotations (Optional[CellTypeHierarchy]):
                Hierarchichal cell type annotations providing labels at
                varying degrees of granularity; used to populate multiple
                observation fields.

        Returns:
            ad.AnnData:
                A copy of the input AnnData object updated with any provided
                embeddings and labels.
        """

        updated = adata.copy()

        if azimuth_embeddings is not None:
            updated.obsm["X_azimuth"] = azimuth_embeddings

        if umap_embeddings is not None:
            updated.obsm["X_azimuth-umap"] = umap_embeddings

        if azimuth_annotations is not None:
            updated.obs[
                "azimuth_labels"
            ] = azimuth_annotations.combined_labels.values.tolist()
            updated.obs[
                "azimuth_labels_consistent"
            ] = azimuth_annotations.strict_labels.values.tolist()

            # Load label corrections based on the specified label resolution
            # and apply them to refine the cell type annotations.
            label_corrections = LabelCorrections.load(self.label_resolution)
            updated.obs[
                "azimuth_labels_refined"
            ] = label_corrections.get_corrected_labels(azimuth_annotations)

            # Retrieve the cell type labels from the first and last levels
            # in the hierarchical annotations and add them to the output.
            root_level = (levels := azimuth_annotations.levels)[0]
            leaf_level = levels[-1]
            updated.obs["azimuth_labels_root"] = azimuth_annotations.labels_by_level[
                root_level
            ].values.tolist()
            updated.obs["azimuth_labels_leaf"] = azimuth_annotations.labels_by_level[
                leaf_level
            ].values.tolist()

        return updated
