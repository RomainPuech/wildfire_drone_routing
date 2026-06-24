#!/usr/bin/env python3
"""Export paper_2021_dataset_creation_explainer.md to PDF using pandoc + XeLaTeX."""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "paper_2021_dataset_creation_explainer.md"
PDF = HERE / "paper_2021_dataset_creation_explainer.pdf"


def main() -> int:
    if not MD.is_file():
        print(f"Missing: {MD}", file=sys.stderr)
        return 1
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("pandoc not found in PATH. Install pandoc (and a LaTeX with xelatex).", file=sys.stderr)
        return 1

    cmd = [
        pandoc,
        str(MD),
        "-o",
        str(PDF),
        "--resource-path",
        str(HERE),
        "-V",
        "geometry:margin=1in",
        "--pdf-engine",
        "xelatex",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HERE))
    print(f"Wrote {PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
