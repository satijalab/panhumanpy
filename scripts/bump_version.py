#!/usr/bin/env python3
"""
Version bumping script for panhumanpy following constellation protocol.
Usage: python scripts/bump_version.py [major|minor|patch] [constellation_name]

Examples:
    python scripts/bump_version.py patch                    # Bug fix
    python scripts/bump_version.py minor                    # New feature
    python scripts/bump_version.py major Cassiopeia         # New model, breaking change
"""

import re
import sys
from pathlib import Path

def get_current_version():
    """Get current version from __init__.py"""
    init_file = Path("src/panhumanpy/__init__.py")
    if not init_file.exists():
        print("❌ Error: Could not find src/panhumanpy/__init__.py")
        return None, None
    
    content = init_file.read_text()
    
    # Extract version
    version_match = re.search(r"__version__ = ['\"](\d+)\.(\d+)\.(\d+)['\"]", content)
    if not version_match:
        print("❌ Error: Could not find version in __init__.py")
        return None, None
    
    # Extract constellation name
    name_match = re.search(r"__version_name__ = ['\"]([^'\"]+)['\"]", content)
    if not name_match:
        print("❌ Error: Could not find version name in __init__.py")
        return None, None
    
    major, minor, patch = map(int, version_match.groups())
    constellation = name_match.group(1)
    
    return (major, minor, patch), constellation

def update_version_in_file(file_path, old_version, new_version, old_name=None, new_name=None):
    """Update version in a file"""
    content = Path(file_path).read_text()
    
    # Create version strings
    old_version_str = f"{old_version[0]}.{old_version[1]}.{old_version[2]}"
    new_version_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    
    # Sanity check: verify current version matches expected old version
    current_version_match = re.search(rf'version = ["\']([^"\']+)["\']', content)
    if current_version_match:
        current_version = current_version_match.group(1)
        if current_version != old_version_str:
            print(f"⚠️  Warning: Expected version {old_version_str} but found {current_version} in {file_path}")
    
    current_version_match = re.search(rf'__version__ = ["\']([^"\']+)["\']', content)
    if current_version_match:
        current_version = current_version_match.group(1)
        if current_version != old_version_str:
            print(f"⚠️  Warning: Expected __version__ {old_version_str} but found {current_version} in {file_path}")
    
    # Sanity check: verify current constellation name matches expected old name
    if old_name:
        current_name_match = re.search(rf'__version_name__ = ["\']([^"\']+)["\']', content)
        if current_name_match:
            current_name = current_name_match.group(1)
            if current_name != old_name:
                print(f"⚠️  Warning: Expected version name {old_name} but found {current_name} in {file_path}")
    
    # Update version number
    content = re.sub(
        rf'version = ["\'].*?["\']',
        f'version = "{new_version_str}"',
        content
    )
    
    content = re.sub(
        rf'__version__ = ["\'].*?["\']',
        f'__version__ = "{new_version_str}"',
        content
    )
    
    # Update constellation name if provided
    if old_name and new_name:
        content = re.sub(
            rf'__version_name__ = ["\'].*?["\']',
            f'__version_name__ = "{new_name}"',
            content
        )
    
    Path(file_path).write_text(content)

def update_contributing_version(old_version, new_version, old_name, new_name):
    """Update version in CONTRIBUTING.md"""
    contributing_path = Path("CONTRIBUTING.md")
    if not contributing_path.exists():
        print("⚠️  Warning: CONTRIBUTING.md not found, skipping")
        return
    
    content = contributing_path.read_text()
    
    old_version_str = f"{old_version[0]}.{old_version[1]}.{old_version[2]}"
    new_version_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    
    # Update the "Current Version:" section
    old_line = f"- {old_version_str} ({old_name})"
    new_line = f"- {new_version_str} ({new_name})"
    
    content = content.replace(old_line, new_line)
    
    contributing_path.write_text(content)

def validate_constellation_name(constellation):
    """Validate constellation name format"""
    if not constellation:
        return False
    
    # Check if it's a valid constellation name (basic validation)
    if not constellation.isalpha() or not constellation[0].isupper():
        print("❌ Error: Constellation name should be capitalized and contain only letters")
        return False
    
    return True

def bump_version(bump_type, constellation=None):
    """Main version bumping logic"""
    # Get current version
    current_version, current_constellation = get_current_version()
    if not current_version:
        return False
    
    major, minor, patch = current_version
    
    print(f"Current version: {major}.{minor}.{patch} ({current_constellation})")
    
    # Calculate new version
    if bump_type == "major":
        new_version = (major + 1, 0, 0)
        if not constellation:
            print("❌ Error: Major version requires constellation name")
            print("Example: python scripts/bump_version.py major Cassiopeia")
            return False
        if not validate_constellation_name(constellation):
            return False
        new_constellation = constellation
    elif bump_type == "minor":
        new_version = (major, minor + 1, 0)
        new_constellation = current_constellation
    elif bump_type == "patch":
        new_version = (major, minor, patch + 1)
        new_constellation = current_constellation
    else:
        print("❌ Error: Invalid bump type. Use 'major', 'minor', or 'patch'")
        return False
    
    new_version_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    
    print(f"New version: {new_version_str} ({new_constellation})")
    
    # Confirm with user
    confirm = input("Proceed with version bump? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Version bump cancelled")
        return False
    
    # Update files
    try:
        # Update pyproject.toml
        update_version_in_file("pyproject.toml", current_version, new_version)
        print("✅ Updated pyproject.toml")
        
        # Update __init__.py
        update_version_in_file(
            "src/panhumanpy/__init__.py", 
            current_version, 
            new_version,
            current_constellation,
            new_constellation
        )
        print("✅ Updated src/panhumanpy/__init__.py")
        
        # Update README.md
        update_readme_version(current_version, new_version, current_constellation, new_constellation)
        print("✅ Updated README.md")
        
        # Update CONTRIBUTING.md
        update_contributing_version(current_version, new_version, current_constellation, new_constellation)
        print("✅ Updated CONTRIBUTING.md")
        
        print(f"\n🎉 Version successfully bumped to {new_version_str} ({new_constellation})")
        print("\nNext steps:")
        print("1. Review the changes: git diff")
        print("2. Commit the changes: git add . && git commit -m 'Bump version to {}'".format(new_version_str))
        print("3. Tag the release: git tag v{}-{}".format(new_version_str, new_constellation.lower()))
        print("3. Push tags: git push origin --tags")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating files: {e}")
        return False

def show_help():
    """Show usage help"""
    print(__doc__)
    print("\nNote: Constellation names are chosen on-the-fly for major versions.")

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
    
    if command not in ["major", "minor", "patch"]:
        print(f"❌ Error: Invalid command '{command}'")
        show_help()
        return 1
    
    constellation = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = bump_version(command, constellation)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())