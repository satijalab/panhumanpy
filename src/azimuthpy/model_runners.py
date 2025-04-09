from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model  # type: ignore [reportMissingImports]

from azimuthpy.datatypes import ModelRunner
from azimuthpy.loss_functions import LvlWtFocalLoss  # noqa: F401
from azimuthpy.utils import align_features, normalize


@dataclass
class KerasModelRunner(ModelRunner):
    model_name: str

    embedding_model: Model
    annotation_model: Model

    feature_panel: list[str]

    encoders: list[LabelEncoder]

    def align_features(self, query, features) -> tuple[csr_matrix, list[str]]:
        """Align the input matrix columns to match the expected feature panel.

        Parameters:
            query (csr_matrix):
                Sparse matrix of input data.
            features (list[str]):
                List of feature names corresponding to the columns in query.

        Returns:
            tuple[csr_matrix, list[str]]:
                A tuple containing the aligned sparse matrix and the updated
                list of features matching the model's feature panel.
        """
        return align_features(query, features, self.feature_panel)

    def normalize(self, query) -> csr_matrix:
        """Normalize the input data using a predefined normalization function.

        Parameters:
            query (csr_matrix): Sparse matrix of raw counts.

        Returns:
            csr_matrix: The normalized sparse matrix.
        """
        return normalize(query)

    def embed(self, query: csr_matrix) -> npt.NDArray:
        """Generate a lower-dimensional embedding of the input data.

        This method uses the embedding model's prediction function to compute
        an embedded representation from the input matrix.

        Parameters:
            query (csr_matrix): Sparse matrix of normalized data.

        Returns:
            np.array: Array representing the embedded features.
        """
        model_output = self.embedding_model.predict(query)

        return model_output

    def annotate(
        self,
        query: csr_matrix,
    ) -> tuple[list[npt.NDArray], list[LabelEncoder]]:
        """
        Annotate the input data with predictions and associated label encoders.

        This method employs the annotation model to generate predictions from
        the input data and returns these predictions alongside the
        corresponding label encoders.

        Parameters:
            query (csr_matrix): Sparse matrix of input data.

        Returns:
            tuple[list[npt.NDArray], list[LabelEncoder]]:
                A tuple where the first element is a list of prediction arrays
                and the second element is a list of LabelEncoder instances.
        """
        model_output = self.annotation_model.predict(query)

        assert len(self.encoders) == len(model_output), (
            len(self.encoders),
            len(model_output),
        )

        return model_output, self.encoders
