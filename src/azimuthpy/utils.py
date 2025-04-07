import numpy as np
from scipy.sparse import csr_matrix, hstack

__all__ = ["align_features", "normalize"]


def align_features(
    query, query_features, reference_features
) -> tuple[csr_matrix, list[str]]:
    """Align the query matrix columns to match the reference feature order.

    This function reorders the columns of the input sparse matrix so that they
    align with the order specified in `reference_features`. If the query matrix
    lacks some features present in the reference, new zero-filled columns are
    appended for those features.

    Parameters:
        query (csr_matrix): Sparse matrix (shape: [n_samples, n_query_features])
            where each column corresponds to a feature in `query_features`.
        query_features (list[str]): List of feature names corresponding to the
            columns in `query`.
        reference_features (list[str]): Desired list of features defining the
            target column order. Missing features in the query will be added as
            zero columns.

    Returns:
        tuple[csr_matrix, list[str]]:
            - csr_matrix: A new sparse matrix with columns arranged to follow
            the order in `reference_features` (with zeros added for missing features).
            - list[str]: Updated list of feature names corresponding to the new
            column order.
    """

    common_features = set(query_features).intersection(set(reference_features))
    extra_features = set(reference_features) - common_features

    if extra_features:
        zeros = csr_matrix((query.shape[0], len(extra_features)))
        query = hstack([query, zeros])
        query_features = list(query_features) + list(extra_features)
    else:
        query_features = list(query_features)

    indices = [query_features.index(feat) for feat in reference_features]

    return query[:, indices], query_features  # type: ignore [reportIndexIssue]


def normalize(counts: csr_matrix, scale_factor: float = 10000) -> csr_matrix:
    """Normalize a sparse count matrix by scaling and log-transforming
    its values.

    This function computes the total counts per row, scales each row so that
    its sum equals the specified scale factor (default 10,000), applies a
    natural logarithm transformation (log1p) to the scaled values, and returns
    the final values in a sparse matrix.

    Parameters:
        counts (csr_matrix):
            A sparse matrix of raw counts (rows: samples, columns: features).
        scale_factor (float, optional):
            The target sum for each row after scaling. Defaults to 10000.

    Returns:
        csr_matrix:
            A normalized sparse matrix with log-transformed values.
    """

    row_sums = np.array(counts.sum(axis=1)).reshape(-1, 1)

    normalized = counts.multiply(scale_factor / row_sums)
    normalized.data = np.log1p(normalized.data)

    return normalized.tocsr()
