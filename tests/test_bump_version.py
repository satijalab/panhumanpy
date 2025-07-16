import tempfile
import shutil
from pathlib import Path
import pytest
import sys
import os
import unittest.mock


sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import bump_version

class TestBumpVersion:
    
    def setup_method(self):
        """Create a temporary directory with test files"""
        self.test_dir = Path(tempfile.mkdtemp())
        os.chdir(self.test_dir)
        
        # Create directory structure
        (self.test_dir / "src" / "panhumanpy").mkdir(parents=True)
        
        # Create test files with initial versions
        self.create_test_files()
        
    def teardown_method(self):
        """Clean up temporary directory"""
        os.chdir(Path.home())  # Change out of temp dir
        shutil.rmtree(self.test_dir)
    
    def create_test_files(self):
        """Create test files with version 1.0.0 (TestConstellation)"""
        
        # src/panhumanpy/__init__.py
        init_content = '''"""Test package"""
__version__ = "1.0.0"
__version_name__ = "TestConstellation"
__version_full__ = f"{__version__} ({__version_name__})"
'''
        (self.test_dir / "src" / "panhumanpy" / "__init__.py").write_text(init_content)
        
        # pyproject.toml
        pyproject_content = '''[project]
name = "test-package"
version = "1.0.0"
description = "Test package"
'''
        (self.test_dir / "pyproject.toml").write_text(pyproject_content)
        
        # README.md
        readme_content = '''# Test Package

**Current version: 1.0.0 (TestConstellation)**

Description here.
'''
        (self.test_dir / "README.md").write_text(readme_content)
        
        # CONTRIBUTING.md
        contributing_content = '''## Developer's Guide

### Version Types:
- MAJOR: New model

#### Current Version:
- 1.0.0 (TestConstellation)

More content here.
'''
        (self.test_dir / "CONTRIBUTING.md").write_text(contributing_content)

    def test_get_current_version(self):
        """Test getting current version from __init__.py"""
        version, constellation = bump_version.get_current_version()
        assert version == (1, 0, 0)
        assert constellation == "TestConstellation"

    def test_check_consistency_when_consistent(self):
        """Test consistency check passes when all files match"""
        version, constellation = bump_version.get_current_version()
        assert bump_version.check_all_consistency(version, constellation) == True

    def test_check_consistency_when_inconsistent(self):
        """Test consistency check fails when files don't match"""
        # Modify pyproject.toml to have wrong version
        pyproject_path = self.test_dir / "pyproject.toml"
        content = pyproject_path.read_text()
        content = content.replace('version = "1.0.0"', 'version = "2.0.0"')
        pyproject_path.write_text(content)
        
        version, constellation = bump_version.get_current_version()
        assert bump_version.check_all_consistency(version, constellation) == False

    def test_init_missing_version(self):
        init_path = self.test_dir / "src" / "panhumanpy" / "__init__.py"
        init_path.write_text('__version_name__ = "TestConstellation"') # Missing __version__
        version, constellation = bump_version.get_current_version()
        assert version is None
        assert constellation is None

    def test_init_missing_name(self):
        init_path = self.test_dir / "src" / "panhumanpy" / "__init__.py"
        init_path.write_text('__version__ = "1.0.0"') # Missing __version_name__
        version, constellation = bump_version.get_current_version()
        assert version is None
        assert constellation is None

    def test_patch_version_bump(self):
        """Test patch version bump (1.0.0 -> 1.0.1)"""
        with unittest.mock.patch('builtins.input', return_value='y'):
            result = bump_version.bump_version("patch")
            assert result == 0
        
        # Check version was updated correctly
        version, constellation = bump_version.get_current_version()
        assert version == (1, 0, 1)
        assert constellation == "TestConstellation"
        
        # Check ALL files are consistent
        assert bump_version.check_all_consistency(version, constellation) == True
        
        # Explicitly verify each file was updated
        pyproject_content = (self.test_dir / "pyproject.toml").read_text()
        assert "1.0.1" in pyproject_content
        
        readme_content = (self.test_dir / "README.md").read_text()
        assert "1.0.1 (TestConstellation)" in readme_content
        
        contributing_content = (self.test_dir / "CONTRIBUTING.md").read_text()
        assert "1.0.1 (TestConstellation)" in contributing_content

    def test_minor_version_bump(self):
        """Test minor version bump (1.0.0 -> 1.1.0)"""
        with unittest.mock.patch('builtins.input', return_value='y'):
            result = bump_version.bump_version("minor")
            assert result == 0
        
        # Check version was updated correctly
        version, constellation = bump_version.get_current_version()
        assert version == (1, 1, 0)
        assert constellation == "TestConstellation" 
        
        # Check ALL files are consistent
        assert bump_version.check_all_consistency(version, constellation) == True
        
        # Explicitly verify each file was updated
        pyproject_content = (self.test_dir / "pyproject.toml").read_text()
        assert "1.1.0" in pyproject_content
        
        readme_content = (self.test_dir / "README.md").read_text()
        assert "1.1.0 (TestConstellation)" in readme_content
        
        contributing_content = (self.test_dir / "CONTRIBUTING.md").read_text()
        assert "1.1.0 (TestConstellation)" in contributing_content

    def test_major_version_bump(self):
        """Test major version bump (1.0.0 -> 2.0.0) with new constellation"""
        with unittest.mock.patch('builtins.input', return_value='y'):
            result = bump_version.bump_version("major", "NewConstellation")
            assert result == 0
        
        # Check version was updated correctly
        version, constellation = bump_version.get_current_version()
        assert version == (2, 0, 0)
        assert constellation == "NewConstellation" 
        
        # Check ALL files are consistent
        assert bump_version.check_all_consistency(version, constellation) == True
        
        # Explicitly verify each file was updated
        pyproject_content = (self.test_dir / "pyproject.toml").read_text()
        assert "2.0.0" in pyproject_content
        
        readme_content = (self.test_dir / "README.md").read_text()
        assert "2.0.0 (NewConstellation)" in readme_content
        
        contributing_content = (self.test_dir / "CONTRIBUTING.md").read_text()
        assert "2.0.0 (NewConstellation)" in contributing_content

    def test_major_version_without_constellation_fails(self):
        """Test that major version bump fails without constellation name"""
        result = bump_version.bump_version("major")
        assert result == 1 

    def test_invalid_constellation_name_fails(self):
        """Test that invalid constellation names are rejected"""
        result = bump_version.bump_version("major", "invalidname")  # lowercase
        assert result == 1
        
        result = bump_version.bump_version("major", "ALLCAPS")  # all caps
        assert result == 1
        
        result = bump_version.bump_version("major", "Invalid123")  # numbers
        assert result == 1

    def test_user_cancellation(self):
        """Test that user can cancel version bump"""
        
        with unittest.mock.patch('builtins.input', return_value='n'):
            result = bump_version.bump_version("patch")
            assert result == 1
        
        # Version should not have changed
        version, constellation = bump_version.get_current_version()
        assert version == (1, 0, 0)
        assert constellation == "TestConstellation"

    def test_missing_init_file(self):
        """Test behavior when __init__.py is missing"""
        (self.test_dir / "src" / "panhumanpy" / "__init__.py").unlink()
        
        version, constellation = bump_version.get_current_version()
        assert version is None
        assert constellation is None

    def test_validate_constellation_name(self):
        """Test constellation name validation"""
        assert bump_version.validate_constellation_name("Andromeda") == True
        assert bump_version.validate_constellation_name("Cassiopeia") == True
        assert bump_version.validate_constellation_name("andromeda") == False  # lowercase
        assert bump_version.validate_constellation_name("ANDROMEDA") == False  # all caps
        assert bump_version.validate_constellation_name("Andromeda123") == False  # numbers
        assert bump_version.validate_constellation_name("") == False  # empty
        assert bump_version.validate_constellation_name(None) == False  # None