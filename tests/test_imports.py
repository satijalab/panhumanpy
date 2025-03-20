"""
Test module imports for panhumanpy.
"""

import inspect


def test_package_import():
    """Test that the panhumanpy package can be imported."""
    try:
        import panhumanpy
    except ImportError:
        assert False, (
            "test_package_import: Failed to import panhumanpy package"
        )


def test_version():
    """Test that the package has a version."""
    import panhumanpy
    assert hasattr(panhumanpy, "__version__")
    assert isinstance(panhumanpy.__version__, str)
    assert panhumanpy.__version__ != ""


def test_exposed_classes_and_functions():
    """Test that all expected classes and functions are exposed."""
    import panhumanpy
    
    # Check for the main exposed classes and functions
    assert hasattr(panhumanpy, "AzimuthNN")
    assert hasattr(panhumanpy, "AzimuthNN_base")
    assert hasattr(panhumanpy, "annotate_core")
    assert hasattr(panhumanpy, "configure")
    
    # Verify AzimuthNN and AzimuthNN_base are classes
    assert inspect.isclass(panhumanpy.AzimuthNN)
    assert inspect.isclass(panhumanpy.AzimuthNN_base)
    
    # Verify annotate_core and configure are functions
    assert callable(panhumanpy.annotate_core)
    assert callable(panhumanpy.configure)


def test_class_inheritance():
    """Test that AzimuthNN inherits from AzimuthNN_base."""
    import panhumanpy
    assert issubclass(panhumanpy.AzimuthNN, panhumanpy.AzimuthNN_base)


def test_all_variable():
    """Test that __all__ contains all the expected names."""
    import panhumanpy
    expected_names = [
        'AzimuthNN',
        'AzimuthNN_base',
        'annotate_core',
        'configure'
    ]
    assert hasattr(panhumanpy, "__all__")
    assert set(panhumanpy.__all__) == set(expected_names), (
        "test_all_variable: __all__ is incorrect. Expected: "
        f"{expected_names}, Got: {panhumanpy.__all__}"
    )


def test_internal_imports():
    """Test that internal modules can be imported."""
    try:
        from panhumanpy import ANNotate
        from panhumanpy import ANNotate_tools
        assert True
    except ImportError:
        assert False, (
            "test_internal_imports: Failed to import panhumanpy.ANNotate "
            "or panhumanpy.ANNotate_tools modules"
        )


def test_detailed_annotations_import():
    """Test that detailed annotation classes and functions can be imported."""
    try:
        # Import specific tools and functions that should be available
        from panhumanpy.ANNotate_tools import (
            QueryObj, ReadQueryObj, InferenceInputData, OutputLabels, 
            Inference, AutoloadInferenceTools, Embeddings, Umaps
        )
        assert True
    except ImportError:
        assert False, (
            "test_detailed_annotations_import: Failed to import helper "
            "functions for Azimuth from panhumanpy.ANNotate_tools"
        )