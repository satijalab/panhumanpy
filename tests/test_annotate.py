def test_annotate():
    "Test that the package can be run on simple input."
    try:
        # import panhumanpy.core
        import os
        import sys

        from panhumanpy.core.ANNotate import annotate

        test_file = "./queries/test_obj.h5ad"
        sys.argv = ["annotate", test_file]
        annotate()
        assert True
    except ImportError:
        assert False, "Failed to run query test"
