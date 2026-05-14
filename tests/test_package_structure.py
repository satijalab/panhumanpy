"""
Tests for package structure.
"""

import os
import re
from pathlib import Path
import importlib.util


def test_package_structure():
   """
   Test that the panhumanpy package structure is correct.
   
   Verifies:
   - Core files exist in the package root
   - _tools directory exists and contains at least one version subdirectory
   - Version subdirectories follow v{i} naming convention
   """
   package_root = Path(__file__).parent.parent/ "src" / "panhumanpy"
   
   assert package_root.exists(), f"Package root not found at {package_root}"
   
   core_files = ["__init__.py", "ANNotate.py", "ANNotate_tools.py", "loss_fn.py"]
   for file in core_files:
       file_path = package_root / file
       assert file_path.exists(), f"Core file {file} not found at {file_path}"
   
   tools_dir = package_root / "_tools"
   assert tools_dir.exists(), f"_tools directory not found at {tools_dir}"
   assert tools_dir.is_dir(), f"_tools is not a directory"
   
   tools_init = tools_dir / "__init__.py"
   assert tools_init.exists(), f"_tools/__init__.py not found"
   
   subdirs = [d for d in tools_dir.iterdir() if d.is_dir() and not d.name.startswith("__")]
   
   assert len(subdirs) >= 1, f"_tools directory must contain at least one subdirectory, found: {[d.name for d in subdirs]}"
   
   version_pattern = re.compile(r'^v\d+$')
   for subdir in subdirs:
       assert version_pattern.match(subdir.name), f"Subdirectory {subdir.name} does not follow v{{i}} naming convention"




def test_version_directory_structure():
    """
    Test that each version directory (v{i}) in _tools has the correct structure and contents.

    Verifies:
    - Each v{i} directory has required subdirectories and files
    - __init__.py files exist where required
    - model_meta dictionary exists with required keys
    - feature_panel_size matches actual feature count
    - loss function exists in loss_fn.py
    """
    package_root = Path(__file__).parent.parent / "src" / "panhumanpy"
    tools_dir = package_root / "_tools"

    version_pattern = re.compile(r'^v\d+$')
    version_dirs = [d for d in tools_dir.iterdir() 
                    if d.is_dir() and version_pattern.match(d.name)]

    assert len(version_dirs) >= 1, "At least one version directory should exist"

    for version_dir in version_dirs:
        version_name = version_dir.name
        
        # Check version directory has __init__.py
        version_init = version_dir / "__init__.py"
        assert version_init.exists(), f"{version_name}/__init__.py not found"
        
        # Required subdirectories and their contents
        required_subdirs = {
            "inference_encoders": ["__init__.py", "inference_encoders.pkl"],
            "inference_feature_panel": ["__init__.py", "inference_feature_panel.txt"],
            "inference_model": ["__init__.py", "inference_model.keras"],
            "postprocessing": ["__init__.py", "panhuman_annotate_fine.csv", "panhuman_annotate_medium.csv"],
            "calibration":["__init__.py"],
            "cell_ontology_map": ["__init__.py", "cell_ontology_map.csv"]
        }
        
        # Check each required subdirectory and its contents
        for subdir_name, required_files in required_subdirs.items():
            subdir = version_dir / subdir_name
            assert subdir.exists(), f"{version_name}/{subdir_name} directory not found"
            assert subdir.is_dir(), f"{version_name}/{subdir_name} is not a directory"
            
            for required_file in required_files:
                file_path = subdir / required_file
                assert file_path.exists(), f"{version_name}/{subdir_name}/{required_file} not found"
        
        # Load and check model_meta from version __init__.py
        spec = importlib.util.spec_from_file_location(f"panhumanpy._tools.{version_name}", version_init)
        version_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(version_module)
        
        assert hasattr(version_module, 'model_meta'), f"{version_name}/__init__.py missing model_meta dictionary"
        model_meta = version_module.model_meta
        assert isinstance(model_meta, dict), f"{version_name}/model_meta is not a dictionary"
        
        # Check required keys in model_meta
        required_keys = [
            'inference_model_name',
            'inference_model_loss_function', 
            'max_depth',
            'inference_model_embedding_layer',
            'feature_panel_size',
            'calibration',
            'calibrator_filenames'
        ]
        
        for key in required_keys:
            assert key in model_meta, f"{version_name}/model_meta missing required key: {key}"
        
        # Verify feature_panel_size matches actual feature count
        feature_panel_file = version_dir / "inference_feature_panel" / "inference_feature_panel.txt"
        with open(feature_panel_file, 'r') as f:
            feature_lines = [line.strip() for line in f if line.strip()]
        
        actual_feature_count = len(feature_lines)
        expected_feature_count = model_meta['feature_panel_size']
        assert actual_feature_count == expected_feature_count, (
            f"{version_name}: feature_panel_size ({expected_feature_count}) does not match "
            f"actual feature count ({actual_feature_count}) in inference_feature_panel.txt"
        )
        
        # Verify loss function exists in loss_fn.py
        loss_fn_file = package_root / "loss_fn.py"
        spec = importlib.util.spec_from_file_location("loss_fn", loss_fn_file)
        loss_fn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loss_fn_module)
        
        assert hasattr(loss_fn_module, 'list_of_loss_fns'), "loss_fn.py missing list_of_loss_fns"
        available_loss_fns = loss_fn_module.list_of_loss_fns
        specified_loss_fn = model_meta['inference_model_loss_function']
        
        assert specified_loss_fn in available_loss_fns, (
            f"{version_name}: loss function '{specified_loss_fn}' not found in "
            f"loss_fn.py list_of_loss_fns: {available_loss_fns}"
        )

        # Verify calibration consistency
        calibration_type = model_meta['calibration']
        calibrators_list = model_meta['calibrator_filenames']

        if calibration_type is None:
            assert len(calibrators_list) == 0, (
                f"{version_name}: When calibration is None, calibrators must be an empty list. "
                f"Found: {calibrators_list}"
            )
        else:
            assert len(calibrators_list) == model_meta['max_depth'], (
                f"{version_name}: When calibration is not None ('{calibration_type}'), "
                f"calibrators list must be max_depth long."
            )

        if len(calibrators_list) > 0:
            calibration_dir = version_dir / "calibration"
            for calibrator_filename in calibrators_list:
                calibrator_path = calibration_dir / calibrator_filename
                assert calibrator_path.exists(), (
                    f"{version_name}: Calibrator file '{calibrator_filename}' not found in "
                    f"calibration directory. Expected at: {calibrator_path}"
                )
                assert calibrator_path.is_file(), (
                    f"{version_name}: Calibrator '{calibrator_filename}' exists but is not a file"
                )
                assert calibrator_path.suffix == '.keras', f"Expected .keras file, got: {calibrator_path}"
        


