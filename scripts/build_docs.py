"""Build Sphinx documentation.

This script builds the HTML documentation for the metacompskin package using Sphinx.
It cleans previous builds and generates fresh documentation.

Usage:
    python scripts/build_docs.py

Output:
    docs/_build/html/index.html
"""

import subprocess
import sys
from pathlib import Path


def build_docs():
    """Build Sphinx HTML documentation."""
    # Get project root and docs directory
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    build_dir = docs_dir / "_build"

    if not docs_dir.exists():
        print(f"ERROR: Documentation directory not found: {docs_dir}")
        sys.exit(1)

    print("=" * 70)
    print("Building Sphinx Documentation for metacompskin")
    print("=" * 70)

    # Clean previous build
    print("\nCleaning previous build...")
    clean_result = subprocess.run(
        [sys.executable, "-m", "sphinx.cmd.build", "-M", "clean", str(docs_dir), str(build_dir)],
        capture_output=True,
        text=True
    )

    if clean_result.returncode != 0:
        print(f"Warning: Clean command failed (this is OK if no previous build exists)")
        print(clean_result.stderr)
    else:
        print("✓ Previous build cleaned")

    # Build HTML docs
    print("\nBuilding HTML documentation...")
    build_result = subprocess.run(
        [sys.executable, "-m", "sphinx.cmd.build", "-b", "html", str(docs_dir), str(build_dir / "html")],
        capture_output=True,
        text=True
    )

    # Show output
    if build_result.stdout:
        print(build_result.stdout)

    if build_result.returncode == 0:
        print("\n" + "=" * 70)
        print("✓ Documentation built successfully!")
        print("=" * 70)
        output_file = build_dir / "html" / "index.html"
        print(f"\nOutput: {output_file}")
        print(f"\nTo view: Open {output_file} in your browser")
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ Documentation build failed!")
        print("=" * 70)
        if build_result.stderr:
            print("\nErrors:")
            print(build_result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(build_docs())

