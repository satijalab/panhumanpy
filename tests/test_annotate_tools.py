"""
Tests for functions and classes in ANNotate_tools.
"""

import os
import warnings
import pytest
import pandas as pd
import numpy as np
import anndata
from scipy.sparse import csr_matrix
from pathlib import Path


from panhumanpy.ANNotate_tools import (
    if_full_consistent_hierarchy,
    insert_col,
    coerce_metadata_types,
    _available_model_versions,
    _load_versioned_ontology_map,
    _make_cl_output_path,
    map_to_cell_ontology,
)


##### test the concordance detection fn if_full_consistent_hierarchy under different conditions

def test_single_level_hierarchy():
    """Test with max_depth=1 (single node) - edge case"""
    cell_label = ['A']
    result = if_full_consistent_hierarchy(cell_label, 1)
    assert result, "Single level hierarchy should always be consistent"

def test_two_level_consistent():
    """Test basic two-level consistent hierarchy"""
    cell_label = ['A', 'A|B']
    result = if_full_consistent_hierarchy(cell_label, 2)
    assert result, "A -> A|B should be consistent"

def test_three_level_consistent():
    """Test basic three-level consistent hierarchy"""
    cell_label = ['A', 'A|B', 'A|B|C']
    result = if_full_consistent_hierarchy(cell_label, 3)
    assert result, "A -> A|B -> A|B|C should be consistent"

def test_deep_consistent_hierarchy():
    """Test deeper consistent hierarchy"""
    cell_label = ['A', 'A|B', 'A|B|C', 'A|B|C|D', 'A|B|C|D|E']
    result = if_full_consistent_hierarchy(cell_label, 5)
    assert result, "Deep consistent hierarchy should return True"

def test_shared_node_path_jump_bug():
    """Test an important bug case - jumping between different paths to shared node"""
    cell_label = ['A', 'A|B', 'A|B|X', 'A|C|X|Y']
    result = if_full_consistent_hierarchy(cell_label, 4)
    assert not result, "Jumping from A|B|X to A|C|X|Y should be inconsistent"

def test_inconsistent_at_level_one():
    """Test inconsistency at first level transition"""
    cell_label = ['A', 'B|C']
    result = if_full_consistent_hierarchy(cell_label, 2)
    assert not result, "A -> B|C should be inconsistent"

def test_inconsistent_at_level_two():
    """Test inconsistency at second level transition"""
    cell_label = ['A', 'A|B', 'A|C|D']
    result = if_full_consistent_hierarchy(cell_label, 3)
    assert not result, "A|B -> A|C|D should be inconsistent"

def test_inconsistent_deep_hierarchy():
    """Test inconsistency deep in hierarchy"""
    cell_label = ['A', 'A|B', 'A|B|C', 'A|B|X|D']
    result = if_full_consistent_hierarchy(cell_label, 4)
    assert not result, "A|B|C -> A|B|X|D should be inconsistent"


####### insert_col tests ################################################
#########################################################################


def test_insert_col_basic():
    """Test that insert_col inserts a column at the specified location."""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df = insert_col(df, 1, 'c', [5, 6])
    assert list(df.columns) == ['a', 'c', 'b'], (
        "insert_col: column not inserted at correct position"
    )
    assert list(df['c']) == [5, 6], (
        "insert_col: column values incorrect"
    )


def test_insert_col_overwrites_existing():
    """Test that insert_col overwrites an existing column of the same name."""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df = insert_col(df, 0, 'b', [9, 9])
    assert list(df.columns).count('b') == 1, (
        "insert_col: duplicate column after overwrite"
    )
    assert list(df['b']) == [9, 9], (
        "insert_col: overwritten column has incorrect values"
    )


def test_insert_col_at_end():
    """Test inserting a column at the end of the DataFrame."""
    df = pd.DataFrame({'a': [1], 'b': [2]})
    df = insert_col(df, 2, 'c', [3])
    assert df.columns[-1] == 'c', (
        "insert_col: column not inserted at end"
    )


def test_insert_col_at_start():
    """Test inserting a column at the start of the DataFrame."""
    df = pd.DataFrame({'a': [1], 'b': [2]})
    df = insert_col(df, 0, 'z', [0])
    assert df.columns[0] == 'z', (
        "insert_col: column not inserted at start"
    )


####### coerce_metadata_types tests ####################################
#########################################################################


def test_coerce_metadata_types_numeric_unchanged():
    """Test that numeric columns are left unchanged."""
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3, 4]})
    result = coerce_metadata_types(df)
    assert result['a'].dtype == df['a'].dtype, (
        "coerce_metadata_types: numeric column dtype changed"
    )


def test_coerce_metadata_types_nan_replaced():
    """Test that NaN-like strings are replaced with 'NA'."""
    df = pd.DataFrame({'a': ['hello', None, 'world']})
    result = coerce_metadata_types(df)
    assert 'NA' in result['a'].values, (
        "coerce_metadata_types: None not replaced with 'NA'"
    )


def test_coerce_metadata_types_returns_copy():
    """Test that coerce_metadata_types returns a copy, not modifying in place."""
    df = pd.DataFrame({'a': ['x', None]})
    result = coerce_metadata_types(df)
    assert df['a'].isna().any(), (
        "coerce_metadata_types: original DataFrame was modified"
    )


####### _available_model_versions tests ################################
#########################################################################


def test_available_model_versions_returns_list():
    """Test that _available_model_versions returns a list."""
    versions = _available_model_versions()
    assert isinstance(versions, list), (
        "_available_model_versions: should return a list"
    )


def test_available_model_versions_nonempty():
    """Test that at least one model version is available."""
    versions = _available_model_versions()
    assert len(versions) > 0, (
        "_available_model_versions: no versions found"
    )


def test_available_model_versions_format():
    """Test that all returned versions follow the v{i} format."""
    import re
    versions = _available_model_versions()
    pattern = re.compile(r'^v\d+$')
    for v in versions:
        assert pattern.match(v), (
            f"_available_model_versions: version '{v}' does not follow v{{i}} format"
        )


def test_available_model_versions_sorted():
    """Test that returned versions are sorted."""
    versions = _available_model_versions()
    assert versions == sorted(versions), (
        "_available_model_versions: versions are not sorted"
    )


####### _load_versioned_ontology_map tests #############################
#########################################################################


def test_load_versioned_ontology_map_returns_dataframe():
    """Test that _load_versioned_ontology_map returns a DataFrame for
    each available version."""
    versions = _available_model_versions()
    for v in versions:
        result = _load_versioned_ontology_map(v)
        assert isinstance(result, pd.DataFrame), (
            f"_load_versioned_ontology_map: did not return a DataFrame "
            f"for version '{v}'"
        )


def test_load_versioned_ontology_map_required_columns():
    """Test that the loaded map has the required columns for each
    available version."""
    required_cols = ['Annotation_Label', 'CL_Label', 'CL_ID']
    versions = _available_model_versions()
    for v in versions:
        df = _load_versioned_ontology_map(v)
        for col in required_cols:
            assert col in df.columns, (
                f"_load_versioned_ontology_map: missing column '{col}' "
                f"for version '{v}'"
            )


def test_load_versioned_ontology_map_nonempty():
    """Test that the loaded map is not empty for each available version."""
    versions = _available_model_versions()
    for v in versions:
        df = _load_versioned_ontology_map(v)
        assert len(df) > 0, (
            f"_load_versioned_ontology_map: empty DataFrame for version '{v}'"
        )


def test_load_versioned_ontology_map_invalid_version_raises():
    """Test that _load_versioned_ontology_map raises ValueError for an
    invalid version."""
    with pytest.raises(ValueError):
        _load_versioned_ontology_map('v999')


####### _make_cl_output_path tests #####################################
#########################################################################


def test_make_cl_output_path_basic(tmp_path):
    """Test that _make_cl_output_path produces a path with _CL suffix."""
    input_path = str(tmp_path / "mydata.h5ad")
    result = _make_cl_output_path(input_path)
    assert result == str(tmp_path / "mydata_CL.h5ad"), (
        "_make_cl_output_path: output path incorrect"
    )


def test_make_cl_output_path_collision(tmp_path):
    """Test that _make_cl_output_path adds a timestamp when output
    already exists."""
    input_path = str(tmp_path / "mydata.h5ad")
    # create the would-be output file so it already exists
    collision_path = tmp_path / "mydata_CL.h5ad"
    collision_path.touch()
    result = _make_cl_output_path(input_path)
    assert result != str(collision_path), (
        "_make_cl_output_path: should not return existing path on collision"
    )
    assert "mydata_CL_" in result, (
        "_make_cl_output_path: timestamp suffix not added on collision"
    )


def test_make_cl_output_path_csv(tmp_path):
    """Test that _make_cl_output_path works correctly for CSV files."""
    input_path = str(tmp_path / "metadata.csv")
    result = _make_cl_output_path(input_path)
    assert result == str(tmp_path / "metadata_CL.csv"), (
        "_make_cl_output_path: CSV output path incorrect"
    )


def test_make_cl_output_path_custom_suffix(tmp_path):
    """Test that _make_cl_output_path respects a custom suffix."""
    input_path = str(tmp_path / "mydata.h5ad")
    result = _make_cl_output_path(input_path, suffix="_mapped")
    assert result == str(tmp_path / "mydata_mapped.h5ad"), (
        "_make_cl_output_path: custom suffix not applied correctly"
    )


####### map_to_cell_ontology tests #####################################
#########################################################################


def _make_test_df(labels):
    """Helper: make a small metadata DataFrame with a label column."""
    return pd.DataFrame({'cell_id': range(len(labels)), 'label': labels})


def _get_valid_labels(n=3):
    """Helper: get n real labels from the default version's ontology map."""
    from panhumanpy.ANNotate import model_version_default
    df = _load_versioned_ontology_map(model_version_default)
    return df['Annotation_Label'].dropna().unique()[:n].tolist()


def test_map_to_cell_ontology_list_input():
    """Test map_to_cell_ontology with a list of valid labels returns a list."""
    labels = _get_valid_labels(3)
    result = map_to_cell_ontology(labels)
    assert isinstance(result, list), (
        "map_to_cell_ontology (list): should return a list"
    )
    assert len(result) == len(labels), (
        "map_to_cell_ontology (list): output length should match input"
    )


def test_map_to_cell_ontology_list_no_unmapped():
    """Test that valid labels produce no 'unmapped' entries."""
    labels = _get_valid_labels(3)
    result = map_to_cell_ontology(labels)
    assert 'unmapped' not in result, (
        "map_to_cell_ontology (list): valid labels should not produce 'unmapped'"
    )


def test_map_to_cell_ontology_list_include_cl_id():
    """Test list input with include_cl_id=True returns a tuple of two lists."""
    labels = _get_valid_labels(3)
    cl_labels, cl_ids = map_to_cell_ontology(labels, include_cl_id=True)
    assert isinstance(cl_labels, list), (
        "map_to_cell_ontology (list, include_cl_id): cl_labels should be a list"
    )
    assert isinstance(cl_ids, list), (
        "map_to_cell_ontology (list, include_cl_id): cl_ids should be a list"
    )
    assert len(cl_labels) == len(labels), (
        "map_to_cell_ontology (list, include_cl_id): cl_labels length mismatch"
    )
    assert len(cl_ids) == len(labels), (
        "map_to_cell_ontology (list, include_cl_id): cl_ids length mismatch"
    )


def test_map_to_cell_ontology_list_unmapped_label():
    """Test that an unknown label is set to 'unmapped' and a warning is emitted."""
    labels = ['ThisLabelDoesNotExistInTheMap_XYZ123']
    with pytest.warns(UserWarning):
        result = map_to_cell_ontology(labels)
    assert result == ['unmapped'], (
        "map_to_cell_ontology (list): unknown label should map to 'unmapped'"
    )


def test_map_to_cell_ontology_list_unmapped_warning_emitted_once():
    """Test that a single warning is emitted even for multiple unmapped labels."""
    labels = ['FakeLabel_A', 'FakeLabel_B', 'FakeLabel_A']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        map_to_cell_ontology(labels)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1, (
        "map_to_cell_ontology (list): should emit exactly one warning for "
        "all unmapped labels combined"
    )


def test_map_to_cell_ontology_dataframe_input():
    """Test map_to_cell_ontology with a DataFrame adds _CL column."""
    labels = _get_valid_labels(3)
    df = _make_test_df(labels)
    result = map_to_cell_ontology(df, src_col='label')
    assert isinstance(result, pd.DataFrame), (
        "map_to_cell_ontology (DataFrame): should return a DataFrame"
    )
    assert 'label_CL' in result.columns, (
        "map_to_cell_ontology (DataFrame): 'label_CL' column not found"
    )


def test_map_to_cell_ontology_dataframe_column_position():
    """Test that _CL column is inserted immediately after src_col."""
    labels = _get_valid_labels(3)
    df = _make_test_df(labels)
    result = map_to_cell_ontology(df, src_col='label')
    cols = list(result.columns)
    src_idx = cols.index('label')
    cl_idx = cols.index('label_CL')
    assert cl_idx == src_idx + 1, (
        "map_to_cell_ontology (DataFrame): 'label_CL' should be immediately "
        "after 'label'"
    )


def test_map_to_cell_ontology_dataframe_include_cl_id():
    """Test DataFrame input with include_cl_id=True adds both columns
    in correct order."""
    labels = _get_valid_labels(3)
    df = _make_test_df(labels)
    result = map_to_cell_ontology(df, src_col='label', include_cl_id=True)
    cols = list(result.columns)
    assert 'label_CL' in cols, (
        "map_to_cell_ontology (DataFrame, include_cl_id): 'label_CL' not found"
    )
    assert 'label_CL_ID' in cols, (
        "map_to_cell_ontology (DataFrame, include_cl_id): 'label_CL_ID' not found"
    )
    cl_idx = cols.index('label_CL')
    cl_id_idx = cols.index('label_CL_ID')
    assert cl_id_idx == cl_idx + 1, (
        "map_to_cell_ontology (DataFrame, include_cl_id): 'label_CL_ID' should "
        "be immediately after 'label_CL'"
    )


def test_map_to_cell_ontology_dataframe_invalid_src_col_raises():
    """Test that a missing src_col raises ValueError."""
    df = pd.DataFrame({'a': [1, 2]})
    with pytest.raises(ValueError):
        map_to_cell_ontology(df, src_col='nonexistent')


def test_map_to_cell_ontology_dataframe_no_src_col_raises():
    """Test that not providing src_col for a DataFrame raises ValueError."""
    df = pd.DataFrame({'a': [1, 2]})
    with pytest.raises(ValueError):
        map_to_cell_ontology(df)


def test_map_to_cell_ontology_anndata_input():
    """Test map_to_cell_ontology with an AnnData object updates obs."""
    labels = _get_valid_labels(3)
    obs = pd.DataFrame({'label': labels}, index=[f'cell_{i}' for i in range(len(labels))])
    adata = anndata.AnnData(
        X=csr_matrix(np.zeros((len(labels), 5))),
        obs=obs
    )
    result = map_to_cell_ontology(adata, src_col='label')
    assert isinstance(result, anndata.AnnData), (
        "map_to_cell_ontology (AnnData): should return an AnnData object"
    )
    assert 'label_CL' in result.obs.columns, (
        "map_to_cell_ontology (AnnData): 'label_CL' not found in obs"
    )


def test_map_to_cell_ontology_h5ad_input(tmp_path):
    """Test map_to_cell_ontology with an h5ad file path reads, annotates,
    and writes a new file."""
    labels = _get_valid_labels(3)
    obs = pd.DataFrame({'label': labels}, index=[f'cell_{i}' for i in range(len(labels))])
    adata = anndata.AnnData(
        X=csr_matrix(np.zeros((len(labels), 5))),
        obs=obs
    )
    input_path = str(tmp_path / "test_input.h5ad")
    adata.write_h5ad(input_path)

    result = map_to_cell_ontology(input_path, src_col='label')

    assert isinstance(result, anndata.AnnData), (
        "map_to_cell_ontology (h5ad): should return an AnnData object"
    )
    assert 'label_CL' in result.obs.columns, (
        "map_to_cell_ontology (h5ad): 'label_CL' not found in obs"
    )
    expected_output = str(tmp_path / "test_input_CL.h5ad")
    assert os.path.exists(expected_output), (
        "map_to_cell_ontology (h5ad): output file not written to disk"
    )


def test_map_to_cell_ontology_csv_input(tmp_path):
    """Test map_to_cell_ontology with a CSV file path reads, annotates,
    and writes a new file."""
    labels = _get_valid_labels(3)
    df = _make_test_df(labels)
    input_path = str(tmp_path / "test_input.csv")
    df.to_csv(input_path, index=False)

    result = map_to_cell_ontology(input_path, src_col='label')

    assert isinstance(result, pd.DataFrame), (
        "map_to_cell_ontology (csv): should return a DataFrame"
    )
    assert 'label_CL' in result.columns, (
        "map_to_cell_ontology (csv): 'label_CL' not found in result"
    )
    expected_output = str(tmp_path / "test_input_CL.csv")
    assert os.path.exists(expected_output), (
        "map_to_cell_ontology (csv): output file not written to disk"
    )


def test_map_to_cell_ontology_invalid_file_extension_raises():
    """Test that a string path with an unsupported extension raises TypeError."""
    with pytest.raises(TypeError):
        map_to_cell_ontology('/some/path/file.txt', src_col='label')


def test_map_to_cell_ontology_nonexistent_h5ad_raises():
    """Test that a non-existent h5ad path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        map_to_cell_ontology('/nonexistent/path/file.h5ad', src_col='label')


def test_map_to_cell_ontology_nonexistent_csv_raises():
    """Test that a non-existent CSV path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        map_to_cell_ontology('/nonexistent/path/file.csv', src_col='label')


def test_map_to_cell_ontology_invalid_version_raises():
    """Test that an invalid model_version raises ValueError."""
    labels = _get_valid_labels(3)
    with pytest.raises(ValueError):
        map_to_cell_ontology(labels, model_version='v999')


def test_map_to_cell_ontology_invalid_data_type_raises():
    """Test that an unsupported data type raises TypeError."""
    with pytest.raises(TypeError):
        map_to_cell_ontology(12345, src_col='label')