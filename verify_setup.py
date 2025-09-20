#!/usr/bin/env python3
"""
Verification script to test the package setup.
"""

import sys
import os

def verify_package_structure():
    """Verify that the package structure is correct."""
    print("Verifying package structure...")
    
    required_dirs = [
        'sas_to_python',
        'examples',
        'examples/data_manipulation',
        'examples/visualization', 
        'examples/modeling',
        'examples/preprocessing',
        'examples/clustering',
        'examples/metadata_operations',
        'utils',
        'tests',
        'docs'
    ]
    
    required_files = [
        'pyproject.toml',
        'README.md',
        '.gitignore',
        'sas_to_python/__init__.py',
        'utils/__init__.py',
        'utils/data_utils.py',
        'utils/reporting_utils.py',
        'tests/__init__.py',
        'tests/test_utils.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Package structure is complete!")
    return True


def verify_imports():
    """Test that basic imports work."""
    print("Testing imports...")
    
    try:
        # Test main package import
        import sas_to_python
        print(f"✅ sas_to_python version: {sas_to_python.__version__}")
        
        # Test utils imports
        from utils.data_utils import validate_data, load_sample_data
        from utils.reporting_utils import create_summary_report
        print("✅ Utility imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 50)
    print("SAS to Python Package Setup Verification")
    print("=" * 50)
    
    structure_ok = verify_package_structure()
    imports_ok = verify_imports()
    
    if structure_ok and imports_ok:
        print("\n🎉 Setup verification successful!")
        print("\nNext steps:")
        print("1. Run: poetry install")
        print("2. Run: poetry shell")
        print("3. Run: python -m pytest tests/")
        return 0
    else:
        print("\n❌ Setup verification failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
