"""
Test annotation functionality in panhumanpy.
"""

import re
import os
import pytest
import anndata
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path



###### test consistency in model version ###############################
########################################################################

def test_model_version_default_format():
    """
    Test that model_version_default in ANNotate.py exists and follows v{i} format.
    """
    annotate_file = Path(__file__).parent.parent / "src" / "panhumanpy" / "ANNotate.py"
    
    with open(annotate_file, 'r') as f:
        content = f.read()
    
    pattern = r"model_version_default\s*=\s*['\"]([^'\"]+)['\"]"
    match = re.search(pattern, content)
    
    assert match, "model_version_default variable not found in ANNotate.py"
    
    model_version_default = match.group(1)
    version_pattern = re.compile(r'^v\d+$')
    
    assert version_pattern.match(model_version_default), (
        f"model_version_default '{model_version_default}' does not follow v{{i}} format"
    )


def test_model_version_directory_exists():
    """
    Test that the directory corresponding to model_version_default exists in _tools.
    """
    annotate_file = Path(__file__).parent.parent / "src" / "panhumanpy" / "ANNotate.py"
    
    with open(annotate_file, 'r') as f:
        content = f.read()
    
    pattern = r"model_version_default\s*=\s*['\"]([^'\"]+)['\"]"
    match = re.search(pattern, content)
    
    assert match, "model_version_default variable not found in ANNotate.py"
    model_version_default = match.group(1)
    
    tools_dir = Path(__file__).parent.parent / "src" / "panhumanpy" / "_tools"
    version_dir = tools_dir / model_version_default
    
    assert version_dir.exists(), (
        f"Directory {model_version_default} not found in _tools directory"
    )
    assert version_dir.is_dir(), (
        f"{model_version_default} exists but is not a directory"
    )


def test_model_version_matches_package_major_version():
    """
    Test that the version number in model_version_default matches the major version 
    of the package as defined in __init__.py.
    """
    annotate_file = Path(__file__).parent.parent / "src" / "panhumanpy" / "ANNotate.py"
    
    with open(annotate_file, 'r') as f:
        content = f.read()
    
    pattern = r"model_version_default\s*=\s*['\"]([^'\"]+)['\"]"
    match = re.search(pattern, content)
    
    assert match, "model_version_default variable not found in ANNotate.py"
    model_version_default = match.group(1)
    
    version_match = re.match(r'v(\d+)', model_version_default)
    assert version_match, f"Could not extract version number from {model_version_default}"
    model_major_version = int(version_match.group(1))
    
    init_file = Path(__file__).parent.parent / "src" / "panhumanpy" / "__init__.py"
    
    with open(init_file, 'r') as f:
        init_content = f.read()
    
    version_pattern = r'__version__\s*=\s*["\']([^"\']+)["\']'
    version_match = re.search(version_pattern, init_content)
    
    assert version_match, "__version__ not found in __init__.py"
    package_version = version_match.group(1)
    
    package_major_version = int(package_version.split('.')[0])
    
    assert model_major_version == package_major_version, (
        f"Model version major number ({model_major_version}) does not match "
        f"package major version ({package_major_version}). "
        f"model_version_default='{model_version_default}', package __version__='{package_version}'"
    )






####### functionality tests in default mode ############################################
#############################################################################


def test_azimuthnn_class():
    """Test that AzimuthNN class works on test data."""
    try:
        from panhumanpy import AzimuthNN
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Load the test object
        test_obj = anndata.read_h5ad(test_file)
        
        # Initialize AzimuthNN with minimal settings to avoid lengthy processing
        # Use small batch size for faster testing
        azimuth = AzimuthNN(
            query_arg=test_obj,
            eval_batch_size=32
        )
        
        # Verify that we got some basic results
        assert hasattr(azimuth, 'annotations'), (
            "test_azimuthnn_class: AzimuthNN object missing 'annotations' "
            "attribute after processing"
        )
        assert hasattr(azimuth, 'cells_meta'), (
            "test_azimuthnn_class: AzimuthNN object missing 'cells_meta' "
            "attribute after processing"
        )
    except ImportError:
        assert False, (
            "test_azimuthnn_class: Failed to import AzimuthNN from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_azimuthnn_class: Error running AzimuthNN on test data: {e}"
        )


def test_azimuthnn_base_with_h5ad():
    """Test AzimuthNN_base with the test h5ad file."""
    try:
        from panhumanpy import AzimuthNN_base
        import anndata
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Initialize the base class
        azimuth_base = AzimuthNN_base()
        
        # Load the test h5ad file
        azimuth_base.query_h5ad(test_file)
        
        # Process the query with minimal settings
        azimuth_base.process_query()
        
        # Run the inference model
        _ = azimuth_base.run_inference_model()
        _ = azimuth_base.calibrate_predictions()

        inference_outputs = azimuth_base._inference_outputs_unprocessed
        
        # Verify that inference outputs have expected structure
        assert isinstance(inference_outputs, dict), (
            "test_azimuthnn_base_with_h5ad: Inference outputs should be "
            "a dictionary"
        )
        
        expected_keys = [
            'hierarchical_label_preds', 
            'class_preds', 
            'probability_of_preds',
            'softmax_vals_all'
        ]
        for key in expected_keys:
            assert key in inference_outputs, (
                f"test_azimuthnn_base_with_h5ad: Missing key '{key}' in "
                "inference outputs"
            )
        
        # Process outputs, detailed mode is more general
        processed_outputs = azimuth_base.process_outputs(mode = 'detailed')
        
        # Verify processed outputs
        assert isinstance(processed_outputs, dict), (
            "test_azimuthnn_base_with_h5ad: Processed outputs should be "
            "a dictionary"
        )
        
        expected_keys_minimal_mode = [
            'full_hierarchical_labels',
            'level_zero_labels',
            'final_level_labels',
            'final_level_confidence',
            'full_consistent_hierarchy'
        ]

        extra_keys_detailed_mode = [
            f'level_{i+1}_labels' for i in range(azimuth_base.max_depth)
        ]
        
        expected_keys = expected_keys_minimal_mode+extra_keys_detailed_mode

        for key in expected_keys:
            assert key in processed_outputs, (
                f"test_azimuthnn_base_with_h5ad: Missing key '{key}' in "
                "processed outputs"
            )
            
    except ImportError:
        assert False, (
            "test_azimuthnn_base_with_h5ad: Failed to import AzimuthNN_base "
            "from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_azimuthnn_base_with_h5ad: Error testing with h5ad: {e}"
        )


def test_annotate_core_with_h5ad():
    """Test annotate_core function with the test h5ad file."""
    try:
        from panhumanpy import annotate_core
        import anndata
        from scipy.sparse import csr_matrix
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Load the test object
        test_obj = anndata.read_h5ad(test_file)
        
        # Extract required inputs for annotate_core
        X_query = csr_matrix(test_obj.X)
        query_features = test_obj.var_names.tolist()
        cells_meta = test_obj.obs
        
        # Call annotate_core with minimal settings
        results = annotate_core(
            X_query=X_query,
            query_features=query_features,
            cells_meta=cells_meta,
            annotation_pipeline='supervised',
            eval_batch_size=32,
            normalization_override=False,
            norm_check_batch_size=32,
            output_mode='minimal',
            refine_labels=False,
            map_to_cl=None,
            include_cl_id = False,
            extract_embeddings=False,
            umap_embeddings=False,
            n_neighbors=5, 
            n_components=2, 
            metric='cosine', 
            min_dist=0.1, 
            umap_lr=1.0, 
            umap_seed=42, 
            spread=1.0,
            verbose=False,
            init='spectral'
        )
        
        # Verify the return structure
        assert isinstance(results, dict), (
            "test_annotate_core_with_h5ad: Function should return a dictionary"
        )
        
        expected_keys = [
            'azimuth_object', 'embeddings_dict', 
            'umap_dict', 'cells_meta'
        ]
        for key in expected_keys:
            assert key in results, (
                f"test_annotate_core_with_h5ad: Missing expected key '{key}' "
                "in results"
            )
        
        # Check that cell metadata has been updated with annotations
        assert 'level_zero_labels' in results['cells_meta'].columns, (
            "test_annotate_core_with_h5ad: Cell metadata should contain "
            "level_zero_labels column"
        )
            
    except ImportError:
        assert False, (
            "test_annotate_core_with_h5ad: Failed to import annotate_core "
            "from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_annotate_core_with_h5ad: Error testing with h5ad: {e}"
        )


def test_embeddings_and_umap_with_h5ad():
    """Test embeddings and UMAP generation with test h5ad file."""
    try:
        from panhumanpy import AzimuthNN
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Load the test object with minimal settings but enable embeddings
        # Use small batch size for faster processing
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32
        )

        _ = azimuth.azimuth_embed()
        _ = azimuth.azimuth_umap()
        
        # Verify that embeddings and UMAP were generated
        assert 'azimuth_embed' in azimuth.embeddings, (
            "test_embeddings_and_umap_with_h5ad: 'azimuth_embed' not found in "
            "embeddings dictionary"
        )
        
        assert 'azimuth_umap' in azimuth.umaps, (
            "test_embeddings_and_umap_with_h5ad: 'azimuth_umap' not found in "
            "umaps dictionary"
        )
        
        # Check embeddings shape
        embeddings = azimuth.embeddings['azimuth_embed']
        assert isinstance(embeddings, np.ndarray), (
            "test_embeddings_and_umap_with_h5ad: Embeddings should be a "
            "numpy array"
        )
        assert embeddings.shape[0] == azimuth.num_cells, (
            "test_embeddings_and_umap_with_h5ad: Embeddings first dimension "
            "should match number of cells"
        )
        
        # Check UMAP shape
        umap_coords = azimuth.umaps['azimuth_umap']
        assert isinstance(umap_coords, np.ndarray), (
            "test_embeddings_and_umap_with_h5ad: UMAP should be a numpy array"
        )
        assert umap_coords.shape[0] == azimuth.num_cells, (
            "test_embeddings_and_umap_with_h5ad: UMAP first dimension "
            "should match number of cells"
        )
        assert umap_coords.shape[1] == 2, (
            "test_embeddings_and_umap_with_h5ad: UMAP second dimension "
            "should be 2 by default"
        )
        
    except ImportError:
        assert False, (
            "test_embeddings_and_umap_with_h5ad: Failed to import AzimuthNN "
            "from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_embeddings_and_umap_with_h5ad: Error testing embeddings "
            f"and UMAP: {e}"
        )


def test_refine_labels_with_h5ad():
    """Test label refinement with test h5ad file."""
    try:
        from panhumanpy import AzimuthNN_base
        import anndata
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Initialize the base class
        azimuth_base = AzimuthNN_base()
        
        # Load the test h5ad file
        azimuth_base.query_h5ad(test_file)
        
        # Process the query
        azimuth_base.process_query()
        
        # Run the inference model
        _ = azimuth_base.run_inference_model()
        _ = azimuth_base.calibrate_predictions()
        
        # Process outputs
        _ = azimuth_base.process_outputs()
        
        # Test refine_labels with all three levels
        for level in ['broad', 'medium', 'fine']:
            refined_labels = azimuth_base.refine_labels(level)
            
            # Check that we got labels
            assert isinstance(refined_labels, list), (
                f"test_refine_labels_with_h5ad: Refined labels for {level} "
                "level should be a list"
            )
            
            assert len(refined_labels) == azimuth_base.num_cells, (
                f"test_refine_labels_with_h5ad: Number of {level} labels "
                "should match number of cells"
            )
            
            # Check that labels were added to the azimuth_refined_labels dict
            assert f'azimuth_{level}' in azimuth_base._azimuth_refined_labels, (
                f"test_refine_labels_with_h5ad: 'azimuth_{level}' not found "
                "in _azimuth_refined_labels dictionary"
            )
        
        # Test update_cells_meta
        updated_meta = azimuth_base.update_cells_meta()
        
        # Check that refined labels are in the updated metadata
        for level in ['broad', 'medium', 'fine']:
            assert f'azimuth_{level}' in updated_meta.columns, (
                f"test_refine_labels_with_h5ad: 'azimuth_{level}' column "
                "not found in updated cell metadata"
            )
        
    except ImportError:
        assert False, (
            "test_refine_labels_with_h5ad: Failed to import AzimuthNN_base "
            "from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_refine_labels_with_h5ad: Error testing label refinement: {e}"
        )



####### Helper functions for model version testing ######################
##########################################################################

def get_default_model_version():
    """
    Get the default model version from ANNotate.py.
    """
    annotate_file = Path(__file__).parent.parent / "src" / "panhumanpy" / "ANNotate.py"
    
    with open(annotate_file, 'r') as f:
        content = f.read()
    
    pattern = r"model_version_default\s*=\s*['\"]([^'\"]+)['\"]"
    match = re.search(pattern, content)
    
    if not match:
        raise ValueError("model_version_default variable not found in ANNotate.py")
    
    return match.group(1)


def get_available_model_versions():
    """
    Get all available model versions by scanning the _tools directory.
    """
    tools_dir = Path(__file__).parent.parent / "src" / "panhumanpy" / "_tools"
    
    if not tools_dir.exists():
        return []
    
    versions = []
    for item in tools_dir.iterdir():
        if item.is_dir() and item.name.startswith('v') and item.name[1:].isdigit():
            versions.append(item.name)
    
    return sorted(versions)


def get_alternate_model_versions():
    """
    Get all model versions except the default one.
    """
    default_version = get_default_model_version()
    all_versions = get_available_model_versions()
    
    return [v for v in all_versions if v != default_version]




####### functionality tests with alternate models ###################
#####################################################################


@pytest.mark.parametrize("model_version", get_alternate_model_versions())
def test_azimuthnn_class_alternate_models(model_version):
    """Test that AzimuthNN class works on test data with alternate model versions."""
    try:
        from panhumanpy import AzimuthNN
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Load the test object
        test_obj = anndata.read_h5ad(test_file)
        
        # Initialize AzimuthNN with alternate model version
        azimuth = AzimuthNN(
            query_arg=test_obj,
            model_version=model_version,
            eval_batch_size=32
        )
        
        # Verify that we got some basic results
        assert hasattr(azimuth, 'annotations'), (
            f"test_azimuthnn_class_alternate_models ({model_version}): "
            "AzimuthNN object missing 'annotations' attribute after processing"
        )
        assert hasattr(azimuth, 'cells_meta'), (
            f"test_azimuthnn_class_alternate_models ({model_version}): "
            "AzimuthNN object missing 'cells_meta' attribute after processing"
        )
        
        # Verify the model version was set correctly
        assert azimuth._model_version == model_version, (
            f"test_azimuthnn_class_alternate_models ({model_version}): "
            f"Model version not set correctly. Expected {model_version}, "
            f"got {azimuth._model_version}"
        )
        
    except ImportError:
        assert False, (
            f"test_azimuthnn_class_alternate_models ({model_version}): "
            "Failed to import AzimuthNN from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_azimuthnn_class_alternate_models ({model_version}): "
            f"Error running AzimuthNN on test data: {e}"
        )


@pytest.mark.parametrize("model_version", get_alternate_model_versions())
def test_azimuthnn_base_with_h5ad_alternate_models(model_version):
    """Test AzimuthNN_base with the test h5ad file using alternate model versions."""
    try:
        from panhumanpy import AzimuthNN_base
        import anndata
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Initialize the base class with alternate model version
        azimuth_base = AzimuthNN_base(model_version=model_version)
        
        # Load the test h5ad file
        azimuth_base.query_h5ad(test_file)
        
        # Process the query with minimal settings
        azimuth_base.process_query()
        
        # Run the inference model
        _ = azimuth_base.run_inference_model()
        _ = azimuth_base.calibrate_predictions()

        inference_outputs = azimuth_base._inference_outputs_unprocessed
        
        # Verify that inference outputs have expected structure
        assert isinstance(inference_outputs, dict), (
            f"test_azimuthnn_base_with_h5ad_alternate_models ({model_version}): "
            "Inference outputs should be a dictionary"
        )
        
        expected_keys = [
            'hierarchical_label_preds', 
            'class_preds', 
            'probability_of_preds',
            'softmax_vals_all'
        ]
        for key in expected_keys:
            assert key in inference_outputs, (
                f"test_azimuthnn_base_with_h5ad_alternate_models ({model_version}): "
                f"Missing key '{key}' in inference outputs"
            )
        
        # Process outputs, detailed mode is more general
        processed_outputs = azimuth_base.process_outputs(mode = 'detailed')
        
        # Verify processed outputs
        assert isinstance(processed_outputs, dict), (
            f"test_azimuthnn_base_with_h5ad_alternate_models ({model_version}): "
            "Processed outputs should be a dictionary"
        )
        
        expected_keys_minimal_mode = [
            'full_hierarchical_labels',
            'level_zero_labels',
            'final_level_labels',
            'final_level_confidence',
            'full_consistent_hierarchy'
        ]

        extra_keys_detailed_mode = [
            f'level_{i+1}_labels' for i in range(azimuth_base.max_depth)
        ]
        
        expected_keys = expected_keys_minimal_mode+extra_keys_detailed_mode

        for key in expected_keys:
            assert key in processed_outputs, (
                f"test_azimuthnn_base_with_h5ad_alternate_models ({model_version}): "
                f"Missing key '{key}' in processed outputs"
            )
            
        # Verify the model version was set correctly
        assert azimuth_base._model_version == model_version, (
            f"test_azimuthnn_base_with_h5ad_alternate_models ({model_version}): "
            f"Model version not set correctly. Expected {model_version}, "
            f"got {azimuth_base._model_version}"
        )
            
    except ImportError:
        assert False, (
            f"test_azimuthnn_base_with_h5ad_alternate_models ({model_version}): "
            "Failed to import AzimuthNN_base from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_azimuthnn_base_with_h5ad_alternate_models ({model_version}): "
            f"Error testing with h5ad: {e}"
        )


@pytest.mark.parametrize("model_version", get_alternate_model_versions())
def test_annotate_core_with_h5ad_alternate_models(model_version):
    """Test annotate_core function with the test h5ad file using alternate model versions."""
    try:
        from panhumanpy import annotate_core
        import anndata
        from scipy.sparse import csr_matrix
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Load the test object
        test_obj = anndata.read_h5ad(test_file)
        
        # Extract required inputs for annotate_core
        X_query = csr_matrix(test_obj.X)
        query_features = test_obj.var_names.tolist()
        cells_meta = test_obj.obs
        
        # Call annotate_core with minimal settings and alternate model version
        results = annotate_core(
            X_query=X_query,
            query_features=query_features,
            cells_meta=cells_meta,
            annotation_pipeline='supervised',
            eval_batch_size=32,
            normalization_override=False,
            norm_check_batch_size=32,
            output_mode='minimal',
            refine_labels=False,
            map_to_cl = None,
            include_cl_id = False,
            extract_embeddings=False,
            umap_embeddings=False,
            n_neighbors=5, 
            n_components=2, 
            metric='cosine', 
            min_dist=0.1, 
            umap_lr=1.0, 
            umap_seed=42, 
            spread=1.0,
            verbose=False,
            init='spectral',
            model_version=model_version
        )
        
        # Verify the return structure
        assert isinstance(results, dict), (
            f"test_annotate_core_with_h5ad_alternate_models ({model_version}): "
            "Function should return a dictionary"
        )
        
        expected_keys = [
            'azimuth_object', 'embeddings_dict', 
            'umap_dict', 'cells_meta'
        ]
        for key in expected_keys:
            assert key in results, (
                f"test_annotate_core_with_h5ad_alternate_models ({model_version}): "
                f"Missing expected key '{key}' in results"
            )
        
        # Check that cell metadata has been updated with annotations
        assert 'level_zero_labels' in results['cells_meta'].columns, (
            f"test_annotate_core_with_h5ad_alternate_models ({model_version}): "
            "Cell metadata should contain level_zero_labels column"
        )
        
        # Verify the model version was set correctly in the azimuth object
        azimuth_obj = results['azimuth_object']
        assert azimuth_obj._model_version == model_version, (
            f"test_annotate_core_with_h5ad_alternate_models ({model_version}): "
            f"Model version not set correctly. Expected {model_version}, "
            f"got {azimuth_obj._model_version}"
        )
            
    except ImportError:
        assert False, (
            f"test_annotate_core_with_h5ad_alternate_models ({model_version}): "
            "Failed to import annotate_core from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_annotate_core_with_h5ad_alternate_models ({model_version}): "
            f"Error testing with h5ad: {e}"
        )


@pytest.mark.parametrize("model_version", get_alternate_model_versions())
def test_embeddings_and_umap_with_h5ad_alternate_models(model_version):
    """Test embeddings and UMAP generation with test h5ad file using alternate model versions."""
    try:
        from panhumanpy import AzimuthNN
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Load the test object with minimal settings but enable embeddings
        # Use small batch size for faster processing and alternate model version
        azimuth = AzimuthNN(
            query_arg=test_file,
            model_version=model_version,
            eval_batch_size=32
        )

        _ = azimuth.azimuth_embed()
        _ = azimuth.azimuth_umap()
        
        # Verify that embeddings and UMAP were generated
        assert 'azimuth_embed' in azimuth.embeddings, (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            "'azimuth_embed' not found in embeddings dictionary"
        )
        
        assert 'azimuth_umap' in azimuth.umaps, (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            "'azimuth_umap' not found in umaps dictionary"
        )
        
        # Check embeddings shape
        embeddings = azimuth.embeddings['azimuth_embed']
        assert isinstance(embeddings, np.ndarray), (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            "Embeddings should be a numpy array"
        )
        assert embeddings.shape[0] == azimuth.num_cells, (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            "Embeddings first dimension should match number of cells"
        )
        
        # Check UMAP shape
        umap_coords = azimuth.umaps['azimuth_umap']
        assert isinstance(umap_coords, np.ndarray), (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            "UMAP should be a numpy array"
        )
        assert umap_coords.shape[0] == azimuth.num_cells, (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            "UMAP first dimension should match number of cells"
        )
        assert umap_coords.shape[1] == 2, (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            "UMAP second dimension should be 2 by default"
        )
        
        # Verify the model version was set correctly
        assert azimuth._model_version == model_version, (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            f"Model version not set correctly. Expected {model_version}, "
            f"got {azimuth._model_version}"
        )
        
    except ImportError:
        assert False, (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            "Failed to import AzimuthNN from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_embeddings_and_umap_with_h5ad_alternate_models ({model_version}): "
            f"Error testing embeddings and UMAP: {e}"
        )


@pytest.mark.parametrize("model_version", get_alternate_model_versions())
def test_refine_labels_with_h5ad_alternate_models(model_version):
    """Test label refinement with test h5ad file using alternate model versions."""
    try:
        from panhumanpy import AzimuthNN_base
        import anndata
        
        # Path to test file
        test_file = os.path.join("queries", "test_obj.h5ad")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found, skipping test")
        
        # Initialize the base class with alternate model version
        azimuth_base = AzimuthNN_base(model_version=model_version)
        
        # Load the test h5ad file
        azimuth_base.query_h5ad(test_file)
        
        # Process the query
        azimuth_base.process_query()
        
        # Run the inference model
        _ = azimuth_base.run_inference_model()
        _ = azimuth_base.calibrate_predictions()
        
        # Process outputs
        _ = azimuth_base.process_outputs()
        
        # Test refine_labels with all three levels
        for level in ['broad', 'medium', 'fine']:
            refined_labels = azimuth_base.refine_labels(level)
            
            # Check that we got labels
            assert isinstance(refined_labels, list), (
                f"test_refine_labels_with_h5ad_alternate_models ({model_version}): "
                f"Refined labels for {level} level should be a list"
            )
            
            assert len(refined_labels) == azimuth_base.num_cells, (
                f"test_refine_labels_with_h5ad_alternate_models ({model_version}): "
                f"Number of {level} labels should match number of cells"
            )
            
            # Check that labels were added to the azimuth_refined_labels dict
            assert f'azimuth_{level}' in azimuth_base._azimuth_refined_labels, (
                f"test_refine_labels_with_h5ad_alternate_models ({model_version}): "
                f"'azimuth_{level}' not found in _azimuth_refined_labels dictionary"
            )
        
        # Test update_cells_meta
        updated_meta = azimuth_base.update_cells_meta()
        
        # Check that refined labels are in the updated metadata
        for level in ['broad', 'medium', 'fine']:
            assert f'azimuth_{level}' in updated_meta.columns, (
                f"test_refine_labels_with_h5ad_alternate_models ({model_version}): "
                f"'azimuth_{level}' column not found in updated cell metadata"
            )
        
        # Verify the model version was set correctly
        assert azimuth_base._model_version == model_version, (
            f"test_refine_labels_with_h5ad_alternate_models ({model_version}): "
            f"Model version not set correctly. Expected {model_version}, "
            f"got {azimuth_base._model_version}"
        )
        
    except ImportError:
        assert False, (
            f"test_refine_labels_with_h5ad_alternate_models ({model_version}): "
            "Failed to import AzimuthNN_base from panhumanpy"
        )
    except Exception as e:
        assert False, (
            f"test_refine_labels_with_h5ad_alternate_models ({model_version}): "
            f"Error testing label refinement: {e}"
        )


####### Additional tests for model version functionality ###################
############################################################################



def test_default_version_is_available():
    """Test that the default model version is available in the _tools directory."""
    default_version = get_default_model_version()
    available_versions = get_available_model_versions()
    
    assert default_version in available_versions, (
        f"Default model version '{default_version}' not found in available "
        f"versions: {available_versions}"
    )


def test_parametrized_tests_skip_when_no_alternates():
    """Test that parametrized tests will skip gracefully when no alternate versions exist."""
    alternate_versions = get_alternate_model_versions()
    
    # This test documents the behavior - if there are no alternate versions,
    # the parametrized tests will be skipped automatically by pytest
    if len(alternate_versions) == 0:
        pytest.skip("No alternate model versions available for testing")
    
    # If we reach here, there are alternate versions available
    assert len(alternate_versions) > 0, (
        "This assertion should not fail if we reach this point"
    )



####### AzimuthNN refine parameter tests ###################################
############################################################################
 
 
def test_azimuthnn_refine_default():
    """Test that AzimuthNN with default refine=True produces all three 
    refinement levels in cells_meta."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32
        )
 
        for level in ['broad', 'medium', 'fine']:
            assert f'azimuth_{level}' in azimuth.cells_meta.columns, (
                f"test_azimuthnn_refine_default: 'azimuth_{level}' "
                f"column not found in cells_meta with default refine=True"
            )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_refine_default: {e}"
        )
 
 
def test_azimuthnn_refine_false():
    """Test that AzimuthNN with refine=False produces no refinement 
    columns."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=False
        )
 
        for level in ['broad', 'medium', 'fine']:
            assert f'azimuth_{level}' not in azimuth.cells_meta.columns, (
                f"test_azimuthnn_refine_false: 'azimuth_{level}' "
                f"should not be in cells_meta with refine=False"
            )
 
        # processed outputs should still exist
        assert azimuth.processed_outputs is not None, (
            "test_azimuthnn_refine_false: processed_outputs should "
            "exist even without refinement"
        )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_refine_false: {e}"
        )
 
 
def test_azimuthnn_refine_none():
    """Test that AzimuthNN with refine=None behaves same as 
    refine=False."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=None
        )
 
        for level in ['broad', 'medium', 'fine']:
            assert f'azimuth_{level}' not in azimuth.cells_meta.columns, (
                f"test_azimuthnn_refine_none: 'azimuth_{level}' "
                f"should not be in cells_meta with refine=None"
            )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_refine_none: {e}"
        )
 
 
def test_azimuthnn_refine_broad_only():
    """Test that AzimuthNN with refine=['broad'] produces only broad 
    refinement."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=['broad']
        )
 
        assert 'azimuth_broad' in azimuth.cells_meta.columns, (
            "test_azimuthnn_refine_broad_only: 'azimuth_broad' "
            "should be in cells_meta"
        )
        for level in ['medium', 'fine']:
            assert f'azimuth_{level}' not in azimuth.cells_meta.columns, (
                f"test_azimuthnn_refine_broad_only: 'azimuth_{level}' "
                f"should not be in cells_meta with refine=['broad']"
            )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_refine_broad_only: {e}"
        )
 
 
def test_azimuthnn_refine_fine_auto_prepends_broad():
    """Test that refine=['fine'] automatically includes broad as a 
    prerequisite."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=['fine']
        )
 
        assert 'azimuth_broad' in azimuth.cells_meta.columns, (
            "test_azimuthnn_refine_fine_auto_prepends_broad: "
            "'azimuth_broad' should be auto-included when "
            "refine=['fine']"
        )
        assert 'azimuth_fine' in azimuth.cells_meta.columns, (
            "test_azimuthnn_refine_fine_auto_prepends_broad: "
            "'azimuth_fine' should be in cells_meta"
        )
        assert 'azimuth_medium' not in azimuth.cells_meta.columns, (
            "test_azimuthnn_refine_fine_auto_prepends_broad: "
            "'azimuth_medium' should not be in cells_meta with "
            "refine=['fine']"
        )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_refine_fine_auto_prepends_broad: {e}"
        )
 
 
def test_azimuthnn_refine_empty_list():
    """Test that refine=[] behaves same as refine=False."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=[]
        )
 
        for level in ['broad', 'medium', 'fine']:
            assert f'azimuth_{level}' not in azimuth.cells_meta.columns, (
                f"test_azimuthnn_refine_empty_list: 'azimuth_{level}' "
                f"should not be in cells_meta with refine=[]"
            )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_refine_empty_list: {e}"
        )
 
 
def test_azimuthnn_refine_invalid_raises():
    """Test that invalid refinement levels raise ValueError."""
    from panhumanpy import AzimuthNN
 
    test_file = os.path.join("queries", "test_obj.h5ad")
    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found")
 
    with pytest.raises(ValueError):
        AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=['coarse']
        )
 
 
def test_azimuthnn_refine_invalid_type_raises():
    """Test that non-list/bool/None refine raises TypeError."""
    from panhumanpy import AzimuthNN
 
    test_file = os.path.join("queries", "test_obj.h5ad")
    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found")
 
    with pytest.raises(TypeError):
        AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine='broad'
        )
 
 
def test_azimuthnn_refine_deprecation_warning():
    """Test that calling azimuth_refine() emits a DeprecationWarning."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32
        )
 
        with pytest.warns(DeprecationWarning):
            azimuth.azimuth_refine()
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_refine_deprecation_warning: {e}"
        )



####### AzimuthNN output consistency with AzimuthNN_base ##################
############################################################################
 
 
def test_azimuthnn_vs_base_processed_outputs():
    """Test that AzimuthNN produces the same processed outputs as 
    AzimuthNN_base for the same input data."""
    try:
        from panhumanpy import AzimuthNN, AzimuthNN_base
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        # run via AzimuthNN_base (old path)
        base = AzimuthNN_base(eval_batch_size=32)
        base.query_h5ad(test_file)
        base.process_query()
        _ = base.run_inference_model()
        _ = base.calibrate_predictions()
        _ = base.process_outputs(mode='minimal')
 
        # run via AzimuthNN (new minibatched path)
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=False
        )
 
        # compare processed output keys
        assert set(base.processed_outputs.keys()) == set(
            azimuth.processed_outputs.keys()
        ), (
            "test_azimuthnn_vs_base_processed_outputs: "
            "processed_outputs keys differ"
        )
 
        # compare values
        for key in base.processed_outputs.keys():
            base_vals = base.processed_outputs[key]
            nn_vals = azimuth.processed_outputs[key]
 
            if isinstance(base_vals, np.ndarray):
                assert np.allclose(
                    base_vals, nn_vals, equal_nan=True
                ), (
                    f"test_azimuthnn_vs_base_processed_outputs: "
                    f"values differ for key '{key}'"
                )
            else:
                assert base_vals == nn_vals, (
                    f"test_azimuthnn_vs_base_processed_outputs: "
                    f"values differ for key '{key}'"
                )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_vs_base_processed_outputs: {e}"
        )
 
 
def test_azimuthnn_vs_base_refined_labels():
    """Test that AzimuthNN produces the same refined labels as 
    AzimuthNN_base for the same input data."""
    try:
        from panhumanpy import AzimuthNN, AzimuthNN_base
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        # run via AzimuthNN_base (old path)
        base = AzimuthNN_base(eval_batch_size=32)
        base.query_h5ad(test_file)
        base.process_query()
        _ = base.run_inference_model()
        _ = base.calibrate_predictions()
        _ = base.process_outputs(mode='minimal')
        _ = base.refine_labels('broad')
        _ = base.refine_labels('medium')
        _ = base.refine_labels('fine')
 
        # run via AzimuthNN (new minibatched path)
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=True
        )
 
        for level in ['broad', 'medium', 'fine']:
            key = f'azimuth_{level}'
            assert key in base._azimuth_refined_labels, (
                f"test_azimuthnn_vs_base_refined_labels: "
                f"'{key}' missing from base._azimuth_refined_labels"
            )
            assert key in azimuth._azimuth_refined_labels, (
                f"test_azimuthnn_vs_base_refined_labels: "
                f"'{key}' missing from azimuth._azimuth_refined_labels"
            )
 
            base_labels = base._azimuth_refined_labels[key]
            nn_labels = azimuth._azimuth_refined_labels[key]
 
            assert base_labels == nn_labels, (
                f"test_azimuthnn_vs_base_refined_labels: "
                f"refined labels differ for '{key}'"
            )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_vs_base_refined_labels: {e}"
        )
 
 
def test_azimuthnn_vs_base_cells_meta_columns():
    """Test that AzimuthNN produces the same cells_meta columns as 
    AzimuthNN_base after full pipeline."""
    try:
        from panhumanpy import AzimuthNN, AzimuthNN_base
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        # run via AzimuthNN_base (old path)
        base = AzimuthNN_base(eval_batch_size=32)
        base.query_h5ad(test_file)
        base.process_query()
        _ = base.run_inference_model()
        _ = base.calibrate_predictions()
        _ = base.process_outputs(mode='minimal')
        _ = base.refine_labels('broad')
        _ = base.refine_labels('medium')
        _ = base.refine_labels('fine')
        _ = base.update_cells_meta()
 
        # run via AzimuthNN (new minibatched path)
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=True
        )
 
        base_cols = set(base.cells_meta.columns)
        nn_cols = set(azimuth.cells_meta.columns)
 
        assert base_cols == nn_cols, (
            f"test_azimuthnn_vs_base_cells_meta_columns: "
            f"cells_meta columns differ.\n"
            f"In base only: {base_cols - nn_cols}\n"
            f"In AzimuthNN only: {nn_cols - base_cols}"
        )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_vs_base_cells_meta_columns: {e}"
        )
 
 
def test_azimuthnn_num_cells_restored_after_pipeline():
    """Test that num_cells is correctly restored to full dataset size 
    after the minibatched pipeline completes."""
    try:
        from panhumanpy import AzimuthNN
        import anndata
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        test_obj = anndata.read_h5ad(test_file)
        expected_cells = test_obj.n_obs
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32
        )
 
        assert azimuth.num_cells == expected_cells, (
            f"test_azimuthnn_num_cells_restored_after_pipeline: "
            f"num_cells is {azimuth.num_cells}, expected "
            f"{expected_cells}"
        )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_num_cells_restored_after_pipeline: {e}"
        )
 
 
def test_azimuthnn_inference_outputs_cleared():
    """Test that _inference_outputs_unprocessed is None after the 
    minibatched pipeline completes, confirming softmax arrays were 
    released."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32
        )
 
        assert azimuth._inference_outputs_unprocessed is None, (
            "test_azimuthnn_inference_outputs_cleared: "
            "_inference_outputs_unprocessed should be None after "
            "minibatched pipeline"
        )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_inference_outputs_cleared: {e}"
        )
 
 
def test_azimuthnn_embeddings_still_work_after_pipeline():
    """Test that embeddings can still be extracted after the 
    minibatched pipeline completes, confirming _inference_input_matrix 
    was properly restored."""
    try:
        from panhumanpy import AzimuthNN
 
        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")
 
        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32
        )
 
        embeddings = azimuth.azimuth_embed()
 
        assert embeddings is not None, (
            "test_azimuthnn_embeddings_still_work_after_pipeline: "
            "azimuth_embed() returned None"
        )
        assert embeddings.shape[0] == azimuth.num_cells, (
            "test_azimuthnn_embeddings_still_work_after_pipeline: "
            "embeddings row count does not match num_cells"
        )
 
    except Exception as e:
        assert False, (
            f"test_azimuthnn_embeddings_still_work_after_pipeline: {e}"
        )



####### AzimuthNN cell ontology mapping tests ##############################
############################################################################


def test_map_to_cell_ontology_class_method():
    """Test that AzimuthNN_base.map_to_cell_ontology adds a _CL column
    immediately after the source column in cells_meta."""
    try:
        from panhumanpy import AzimuthNN

        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")

        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=['broad']
        )

        azimuth.map_to_cell_ontology('azimuth_broad')

        cols = list(azimuth.cells_meta.columns)

        assert 'azimuth_broad_CL' in cols, (
            "test_map_to_cell_ontology_class_method: "
            "'azimuth_broad_CL' not found in cells_meta"
        )

        # verify column is immediately after src_col
        src_idx = cols.index('azimuth_broad')
        cl_idx = cols.index('azimuth_broad_CL')
        assert cl_idx == src_idx + 1, (
            "test_map_to_cell_ontology_class_method: "
            "'azimuth_broad_CL' should be immediately after 'azimuth_broad'"
        )

        # verify no None values
        assert azimuth.cells_meta['azimuth_broad_CL'].notna().all(), (
            "test_map_to_cell_ontology_class_method: "
            "'azimuth_broad_CL' should have no NaN values"
        )

    except Exception as e:
        assert False, (
            f"test_map_to_cell_ontology_class_method: {e}"
        )


def test_map_to_cell_ontology_class_method_include_cl_id():
    """Test that AzimuthNN_base.map_to_cell_ontology with include_cl_id=True
    adds both _CL and _CL_ID columns in the correct order."""
    try:
        from panhumanpy import AzimuthNN

        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")

        azimuth = AzimuthNN(
            query_arg=test_file,
            eval_batch_size=32,
            refine=['broad']
        )

        azimuth.map_to_cell_ontology('azimuth_broad', include_cl_id=True)

        cols = list(azimuth.cells_meta.columns)

        assert 'azimuth_broad_CL' in cols, (
            "test_map_to_cell_ontology_class_method_include_cl_id: "
            "'azimuth_broad_CL' not found in cells_meta"
        )
        assert 'azimuth_broad_CL_ID' in cols, (
            "test_map_to_cell_ontology_class_method_include_cl_id: "
            "'azimuth_broad_CL_ID' not found in cells_meta"
        )

        # verify ordering: src_col, _CL, _CL_ID
        src_idx = cols.index('azimuth_broad')
        cl_idx = cols.index('azimuth_broad_CL')
        cl_id_idx = cols.index('azimuth_broad_CL_ID')

        assert cl_idx == src_idx + 1, (
            "test_map_to_cell_ontology_class_method_include_cl_id: "
            "'azimuth_broad_CL' should be immediately after 'azimuth_broad'"
        )
        assert cl_id_idx == cl_idx + 1, (
            "test_map_to_cell_ontology_class_method_include_cl_id: "
            "'azimuth_broad_CL_ID' should be immediately after "
            "'azimuth_broad_CL'"
        )

        # verify no None values in either column
        for col in ['azimuth_broad_CL', 'azimuth_broad_CL_ID']:
            assert azimuth.cells_meta[col].notna().all(), (
                f"test_map_to_cell_ontology_class_method_include_cl_id: "
                f"'{col}' should have no NaN values"
            )

    except Exception as e:
        assert False, (
            f"test_map_to_cell_ontology_class_method_include_cl_id: {e}"
        )


def test_map_to_cell_ontology_invalid_src_col_raises():
    """Test that map_to_cell_ontology raises ValueError when src_col
    does not exist in cells_meta."""
    from panhumanpy import AzimuthNN

    test_file = os.path.join("queries", "test_obj.h5ad")
    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found")

    azimuth = AzimuthNN(
        query_arg=test_file,
        eval_batch_size=32,
        refine=False
    )

    with pytest.raises(ValueError):
        azimuth.map_to_cell_ontology('nonexistent_column')


def test_annotate_core_with_map_to_cl():
    """Test that annotate_core with map_to_cl adds CL columns to the
    returned cells_meta."""
    try:
        from panhumanpy import annotate_core

        test_file = os.path.join("queries", "test_obj.h5ad")
        if not os.path.exists(test_file):
            pytest.skip(f"Test file {test_file} not found")

        test_obj = anndata.read_h5ad(test_file)
        X_query = csr_matrix(test_obj.X)
        query_features = test_obj.var_names.tolist()
        cells_meta = test_obj.obs

        results = annotate_core(
            X_query=X_query,
            query_features=query_features,
            cells_meta=cells_meta,
            annotation_pipeline='supervised',
            eval_batch_size=32,
            normalization_override=False,
            norm_check_batch_size=32,
            output_mode='minimal',
            refine_labels=True,
            map_to_cl=['azimuth_broad'],
            include_cl_id=False,
            extract_embeddings=False,
            umap_embeddings=False,
            n_neighbors=5,
            n_components=2,
            metric='cosine',
            min_dist=0.1,
            umap_lr=1.0,
            umap_seed=42,
            spread=1.0,
            verbose=False,
            init='spectral'
        )

        assert 'azimuth_broad_CL' in results['cells_meta'].columns, (
            "test_annotate_core_with_map_to_cl: "
            "'azimuth_broad_CL' not found in returned cells_meta"
        )

        assert results['cells_meta']['azimuth_broad_CL'].notna().all(), (
            "test_annotate_core_with_map_to_cl: "
            "'azimuth_broad_CL' should have no NaN values"
        )

    except Exception as e:
        assert False, (
            f"test_annotate_core_with_map_to_cl: {e}"
        )


def test_annotate_core_map_to_cl_invalid_type_raises():
    """Test that annotate_core raises TypeError when map_to_cl is a
    string instead of a list."""
    from panhumanpy import annotate_core

    test_file = os.path.join("queries", "test_obj.h5ad")
    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found")

    test_obj = anndata.read_h5ad(test_file)
    X_query = csr_matrix(test_obj.X)
    query_features = test_obj.var_names.tolist()
    cells_meta = test_obj.obs

    with pytest.raises(TypeError):
        annotate_core(
            X_query=X_query,
            query_features=query_features,
            cells_meta=cells_meta,
            annotation_pipeline='supervised',
            eval_batch_size=32,
            normalization_override=False,
            norm_check_batch_size=32,
            output_mode='minimal',
            refine_labels=False,
            map_to_cl='azimuth_broad',
            include_cl_id=False,
            extract_embeddings=False,
            umap_embeddings=False,
            n_neighbors=5,
            n_components=2,
            metric='cosine',
            min_dist=0.1,
            umap_lr=1.0,
            umap_seed=42,
            spread=1.0,
            verbose=False,
            init='spectral'
        )


def test_annotate_core_include_cl_id_invalid_type_raises():
    """Test that annotate_core raises TypeError when include_cl_id is
    not a bool."""
    from panhumanpy import annotate_core

    test_file = os.path.join("queries", "test_obj.h5ad")
    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found")

    test_obj = anndata.read_h5ad(test_file)
    X_query = csr_matrix(test_obj.X)
    query_features = test_obj.var_names.tolist()
    cells_meta = test_obj.obs

    with pytest.raises(TypeError):
        annotate_core(
            X_query=X_query,
            query_features=query_features,
            cells_meta=cells_meta,
            annotation_pipeline='supervised',
            eval_batch_size=32,
            normalization_override=False,
            norm_check_batch_size=32,
            output_mode='minimal',
            refine_labels=False,
            map_to_cl=None,
            include_cl_id='yes',
            extract_embeddings=False,
            umap_embeddings=False,
            n_neighbors=5,
            n_components=2,
            metric='cosine',
            min_dist=0.1,
            umap_lr=1.0,
            umap_seed=42,
            spread=1.0,
            verbose=False,
            init='spectral'
        )
 