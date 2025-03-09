def test_import():
    """Test that the package can be imported."""
    try:
        # import panhumanpy.core
        from panhumanpy.core.ANNotate import annotate

        assert True
    except ImportError:
        assert False, "Failed to import panhumanpy"
