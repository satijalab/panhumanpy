#!/usr/bin/env python3
"""
Version bumping script for panhumanpy following constellation protocol.
Usage: python scripts/bump_version.py [major|minor|patch] [constellation_name]

Examples:
    python scripts/bump_version.py patch                    # Bug fix
    python scripts/bump_version.py minor                    # New feature
    python scripts/bump_version.py major Cassiopeia         # New model, breaking changes to API
"""

import re
import sys
from pathlib import Path

# File paths for version info
INIT_FILE = Path("src/panhumanpy/__init__.py")
PYPROJECT_FILE = Path("pyproject.toml")
README_FILE = Path("README.md")
CONTRIBUTING_FILE = Path("CONTRIBUTING.md")

# Regex patterns for finding version info
VERSION_PATTERNS = {
    "init": re.compile(r"__version__ = ['\"](\d+)\.(\d+)\.(\d+)['\"]"),
    "name": re.compile(r"__version_name__ = ['\"]([^'\"]+)['\"]"),
    "pyproject": re.compile(r"version = ['\"](\d+\.\d+\.\d+)['\"]"),
    "readme": re.compile(r"\*\*Current version: ([\d.]+) \(([^)]+)\)\*\*"),
    "contributing": re.compile(r"- ([\d.]+)\s*\(([^)]+)\)"), 
}

def get_current_version():
    """Get current version from __init__.py"""
    if not INIT_FILE.exists():
        print("❌ Error: Could not find src/panhumanpy/__init__.py")
        return None, None
    
    content = INIT_FILE.read_text()
    
    version_match = VERSION_PATTERNS["init"].search(content)
    if not version_match:
        print("❌ Error: Could not find version in __init__.py")
        return None, None
    
    name_match = VERSION_PATTERNS["name"].search(content)
    if not name_match:
        print("❌ Error: Could not find version name in __init__.py")
        return None, None
    
    major, minor, patch = map(int, version_match.groups())
    constellation = name_match.group(1)
    
    return (major, minor, patch), constellation

def check_file_consistency(file_path, version_str, constellation_name=None):
    """Check that version info in a single file is consistent."""
    if not file_path.exists():
        print(f"⚠️  Warning: {file_path} not found, skipping consistency check.")
        return True

    content = file_path.read_text()
    
    if file_path == PYPROJECT_FILE:
        match = VERSION_PATTERNS["pyproject"].search(content)
        if match and match.group(1) == version_str:
            return True
    elif file_path == README_FILE:
        match = VERSION_PATTERNS["readme"].search(content)
        if match and match.group(1) == version_str and match.group(2) == constellation_name:
            return True
    elif file_path == CONTRIBUTING_FILE:
        match = VERSION_PATTERNS["contributing"].search(content)
        if match and match.group(1) == version_str and match.group(2) == constellation_name:
            return True
    
    print(f"❌ Version mismatch in {file_path}")
    print(f"   Expected version: {version_str}")
    if constellation_name:
        print(f"   Expected name: {constellation_name}")
    print(f"   Found: {match.group(0) if match else 'None'}")
    return False

def check_all_consistency(current_version, current_constellation):
    """Check version consistency across all files"""
    current_version_str = f"{current_version[0]}.{current_version[1]}.{current_version[2]}"
    
    print("🔍 Checking version consistency across files...")
    
    files_to_check = [
        (PYPROJECT_FILE, current_version_str, None),
        (README_FILE, current_version_str, current_constellation),
        (CONTRIBUTING_FILE, current_version_str, current_constellation)
    ]
    
    for file, version_str, name in files_to_check:
        if not check_file_consistency(file, version_str, name):
            print("\n❌ Version inconsistency detected!")
            return False
            
    print("✅ All files have consistent version information")
    return True

def update_file(file_path, old_version_str, new_version_str, new_name=None):
    """Update version and name in a single file based on its type."""
    if not file_path.exists():
        print(f"⚠️  Warning: {file_path} not found, skipping update.")
        return
        
    content = file_path.read_text()
    
    if file_path == INIT_FILE:
        # Update version number
        content = VERSION_PATTERNS["init"].sub(f'__version__ = "{new_version_str}"', content)
        # Update version name
        if new_name:
            content = VERSION_PATTERNS["name"].sub(f'__version_name__ = "{new_name}"', content)
    
    elif file_path == PYPROJECT_FILE:
        content = VERSION_PATTERNS["pyproject"].sub(f'version = "{new_version_str}"', content)

    elif file_path == README_FILE:
        #old_version_str is being used to build the regex string for robustness
        pattern = re.compile(rf'(\*\*Current version: )({re.escape(old_version_str)}) \(([^)]+)\)\*\*')
        content = pattern.sub(f'**Current version: {new_version_str} ({new_name})**', content)
    
    elif file_path == CONTRIBUTING_FILE:
        pattern = re.compile(rf'- {re.escape(old_version_str)} \([^)]+\)')
        content = pattern.sub(f'- {new_version_str} ({new_name})', content)
        
    file_path.write_text(content)

def validate_constellation_name(constellation):
    """Validate constellation name format"""
    if not constellation:
        return False
    if not constellation.isalpha() or not constellation[0].isupper() or constellation[1:].isupper():
        print("❌ Error: First letter of constellation name must be capitalized and contain only letters")
        return False
    return True

def bump_version(bump_type, constellation=None):
    """Main version bumping logic"""
    current_version, current_constellation = get_current_version()
    if not current_version:
        return 1
    
    if not check_all_consistency(current_version, current_constellation):
        return 1
    
    major, minor, patch = current_version
    print(f"Current version: {major}.{minor}.{patch} ({current_constellation})")
    
    new_constellation = current_constellation
    
    if bump_type == "major":
        new_version = (major + 1, 0, 0)
        if not constellation:
            print("❌ Error: Major version requires a new constellation name.")
            print("Example: python scripts/bump_version.py major Cassiopeia")
            return 1
        if not validate_constellation_name(constellation):
            return 1
        new_constellation = constellation
    elif bump_type == "minor":
        new_version = (major, minor + 1, 0)
    elif bump_type == "patch":
        new_version = (major, minor, patch + 1)
    else:
        print("❌ Error: Invalid bump type. Use 'major', 'minor', or 'patch'")
        return 1
    
    new_version_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    print(f"New version: {new_version_str} ({new_constellation})")
    
    confirm = input("Proceed with version bump? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Version bump cancelled")
        return 1
    
    try:
        update_file(PYPROJECT_FILE, f"{major}.{minor}.{patch}", new_version_str)
        print("✅ Updated pyproject.toml")
        
        update_file(
            INIT_FILE, 
            f"{major}.{minor}.{patch}", 
            new_version_str,
            new_name=new_constellation
        )
        print("✅ Updated src/panhumanpy/__init__.py")
        
        update_file(
            README_FILE, 
            f"{major}.{minor}.{patch}", 
            new_version_str, 
            new_name=new_constellation
        )
        print("✅ Updated README.md")

        update_file(
            CONTRIBUTING_FILE, 
            f"{major}.{minor}.{patch}", 
            new_version_str, 
            new_name=new_constellation
        )
        print("✅ Updated CONTRIBUTING.md")

        print(f"\n🎉 Version successfully bumped to {new_version_str} ({new_constellation})")
        print("\nNext steps:")
        print("1. Review the changes: git diff")
        print("2. Commit the changes: git add . && git commit -m 'Bump version to {}'".format(new_version_str))
        print("3. Tag the release: git tag v{}-{}".format(new_version_str, new_constellation.lower())) 
        print("4. Push changes and tags: git push --follow-tags  # if on branch main") 
        print("..else git push origin version_bump_branch --tags # if on a separate branch")
        
        return 0
    except Exception as e:
        print(f"❌ Error updating files: {e}")
        return 1

def show_help():
    """Show usage help"""
    print(__doc__)
    print("\nNote: Constellation names are chosen on-the-fly for major versions")

def main():
    if len(sys.argv) < 2:
        show_help()
        return 1
    
    command = sys.argv[1]
    
    if command in ["help", "-h", "--help"]:
        show_help()
        return 0
    
    if command == "current":
        version, constellation = get_current_version()
        if version:
            print(f"Current version: {version[0]}.{version[1]}.{version[2]} ({constellation})")
        return 0
    
    if command == "check":
        version, constellation = get_current_version()
        if version:
            check_all_consistency(version, constellation)
        return 0
    
    if command not in ["major", "minor", "patch"]:
        print(f"❌ Error: Invalid command '{command}'")
        show_help()
        return 1
    
    constellation = sys.argv[2] if len(sys.argv) > 2 else None
    
    return bump_version(command, constellation)

if __name__ == "__main__":
    sys.exit(main())