from scipy.sparse import csr_matrix, hstack

__all__ = ["align_features"]


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
