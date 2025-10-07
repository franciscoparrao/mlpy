#!/usr/bin/env python
"""
Generate MLPY documentation automatically using Sphinx.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def generate_api_docs():
    """Generate API documentation using sphinx-apidoc."""
    print("[API] Generating API documentation...")
    
    # Paths
    project_root = Path(__file__).parent
    source_dir = project_root / "mlpy"
    docs_dir = project_root / "docs"
    api_dir = docs_dir / "source" / "api"
    
    # Create api directory if it doesn't exist
    api_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate API docs
    cmd = [
        sys.executable, "-m", "sphinx.ext.apidoc",
        "-f",  # Force overwrite
        "-M",  # Module-first
        "-e",  # Separate pages for each module
        "-T",  # No TOC file
        "-o", str(api_dir),  # Output directory
        str(source_dir),  # Source directory
        # Exclude patterns
        "*/tests/*",
        "*/test_*.py",
        "*/__pycache__/*"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("[OK] API documentation generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error generating API docs: {e}")
        return False
    
    return True

def build_html_docs():
    """Build HTML documentation using Sphinx."""
    print("[BUILD] Building HTML documentation...")
    
    # Paths
    project_root = Path(__file__).parent
    docs_dir = project_root / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"
    
    # Clean previous build
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    # Build HTML
    cmd = [
        sys.executable, "-m", "sphinx",
        "-b", "html",  # Build HTML
        "-W",  # Warnings as errors
        "-j", "auto",  # Parallel build
        str(source_dir),  # Source directory
        str(build_dir / "html")  # Output directory
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=docs_dir)
        print("[OK] HTML documentation built successfully!")
        print(f"[PATH] Documentation available at: {build_dir / 'html' / 'index.html'}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error building HTML docs: {e}")
        return False
    
    return True

def create_api_index():
    """Create an index file for API documentation."""
    print("[INDEX] Creating API index...")
    
    project_root = Path(__file__).parent
    api_dir = project_root / "docs" / "source" / "api"
    
    index_content = """
API Reference
=============

This section contains the complete API reference for MLPY.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   mlpy.base
   mlpy.core

Tasks
-----

.. toctree::
   :maxdepth: 2

   mlpy.tasks

Learners
--------

.. toctree::
   :maxdepth: 2

   mlpy.learners
   mlpy.learners.baseline
   mlpy.learners.ensemble
   mlpy.learners.sklearn

Measures
--------

.. toctree::
   :maxdepth: 2

   mlpy.measures

Resamplings
-----------

.. toctree::
   :maxdepth: 2

   mlpy.resamplings

Pipelines
---------

.. toctree::
   :maxdepth: 2

   mlpy.pipelines

AutoML
------

.. toctree::
   :maxdepth: 2

   mlpy.automl

Validation
----------

.. toctree::
   :maxdepth: 2

   mlpy.validation

Visualization
-------------

.. toctree::
   :maxdepth: 2

   mlpy.visualization

Utilities
---------

.. toctree::
   :maxdepth: 2

   mlpy.utils
"""
    
    index_file = api_dir / "index.rst"
    index_file.write_text(index_content.strip())
    print("[OK] API index created!")

def main():
    """Main function to generate all documentation."""
    print("[DOCS] MLPY Documentation Generator")
    print("=" * 50)
    
    # Check if Sphinx is installed
    try:
        import sphinx
        print(f"[OK] Sphinx version: {sphinx.__version__}")
    except ImportError:
        print("[WARNING] Sphinx not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "sphinx", "sphinx-rtd-theme"])
    
    # Generate documentation
    success = True
    
    # Step 1: Generate API docs
    if not generate_api_docs():
        success = False
    
    # Step 2: Create API index
    create_api_index()
    
    # Step 3: Build HTML docs
    if not build_html_docs():
        success = False
    
    if success:
        print("\n" + "=" * 50)
        print("[SUCCESS] Documentation generated successfully!")
        print("[INFO] To view the documentation, open:")
        print(f"   docs/build/html/index.html")
    else:
        print("\n" + "=" * 50)
        print("[WARNING] Documentation generation completed with errors.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())