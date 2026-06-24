#!/usr/bin/env python3
"""Build sn-article.pdf (pdflatex + BibTeX).

Compiles in paper/Nature_Wildfires/ (figures, class, bib paths) and copies the
PDF to paper/sn-article.pdf on success.

Requires a TeX distribution (e.g. MacTeX / TeX Live) with pdflatex and bibtex.
If latexmk is on PATH, it is used (recommended).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "Nature_Wildfires"
MAIN = "sn-article"


def _copy_pdf_to_paper() -> None:
    built = SRC / f"{MAIN}.pdf"
    dest = ROOT / f"{MAIN}.pdf"
    if built.is_file():
        shutil.copy2(built, dest)
        print(f"PDF: {dest}", file=sys.stderr)


def main() -> int:
    if not SRC.is_dir():
        print(f"Missing source directory: {SRC}", file=sys.stderr)
        return 1

    tex = f"{MAIN}.tex"
    if shutil.which("latexmk"):
        r = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex,
            ],
            cwd=SRC,
        )
        if r.returncode == 0:
            _copy_pdf_to_paper()
        return r.returncode

    print(
        "latexmk not found; using pdflatex + bibtex (install latexmk for faster rebuilds).",
        file=sys.stderr,
    )
    pdflatex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex]
    for cmd in (pdflatex, ["bibtex", MAIN], pdflatex, pdflatex):
        r = subprocess.run(cmd, cwd=SRC)
        if r.returncode != 0:
            return r.returncode
    _copy_pdf_to_paper()
    return 0


if __name__ == "__main__":
    sys.exit(main())
