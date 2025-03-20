"""
Test annotation functionality in panhumanpy.
"""

import os
import pytest
import anndata
import numpy as np
from scipy.sparse import csr_matrix


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
        inference_outputs = azimuth_base.run_inference_model()
        
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
        
        # Process outputs
        processed_outputs = azimuth_base.process_outputs()
        
        # Verify processed outputs
        assert isinstance(processed_outputs, dict), (
            "test_azimuthnn_base_with_h5ad: Processed outputs should be "
            "a dictionary"
        )
        
        expected_keys = [
            'full_hierarchical_labels',
            'level_zero_labels',
            'final_level_labels',
            'final_level_softmax_prob',
            'full_consistent_hierarchy'
        ]
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