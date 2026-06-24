#!/usr/bin/env python3
"""
build_pdf.py — Recompile the manuscript PDF from LaTeX.

Runs ``latexmk`` (pdflatex + bibtex) on sn-article.tex, which handles the
multi-pass build (references, citations, bibliography) automatically.

Usage:
    python scripts/build_pdf.py            # build sn-article.pdf
    python scripts/build_pdf.py --clean    # remove aux files, then build
    python scripts/build_pdf.py -C         # only clean (remove aux + pdf)
    python scripts/build_pdf.py --file foo.tex

Exit code is non-zero if the build fails; on failure the relevant error lines
from the .log are printed.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# paper/Nature_Wildfires/ (parent of scripts/)
PAPER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TEX = "sn-article.tex"


def find_latexmk() -> str:
    exe = shutil.which("latexmk")
    if not exe:
        sys.exit(
            "ERROR: latexmk not found on PATH. Install a TeX distribution "
            "(e.g. MacTeX/TeX Live) or add it to PATH."
        )
    return exe


def print_log_errors(log_path: Path) -> None:
    if not log_path.is_file():
        return
    text = log_path.read_text(errors="replace")
    # LaTeX errors start with "!"; also surface undefined refs/citations.
    interesting = [
        ln for ln in text.splitlines()
        if ln.startswith("!")
        or "Undefined control sequence" in ln
        or "Emergency stop" in ln
        or re.search(r"Reference .* undefined", ln)
        or re.search(r"Citation .* undefined", ln)
    ]
    if interesting:
        print("\n--- LaTeX errors/warnings from log ---", file=sys.stderr)
        for ln in interesting[:40]:
            print(ln, file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_TEX,
                    help=f"Top-level .tex file (default: {DEFAULT_TEX}).")
    ap.add_argument("--clean", action="store_true",
                    help="Remove auxiliary files before building.")
    ap.add_argument("-C", "--clean-only", action="store_true",
                    help="Remove auxiliary files and the PDF, then exit.")
    args = ap.parse_args()

    latexmk = find_latexmk()
    tex = args.file
    tex_path = PAPER_DIR / tex
    if not args.clean_only and not tex_path.is_file():
        sys.exit(f"ERROR: {tex_path} not found.")

    if args.clean_only:
        subprocess.run([latexmk, "-C", tex], cwd=PAPER_DIR)
        print("Cleaned auxiliary files and PDF.")
        return 0

    if args.clean:
        subprocess.run([latexmk, "-c", tex], cwd=PAPER_DIR)

    cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex]
    print(f"Running: {' '.join(cmd)}\n  (cwd: {PAPER_DIR})")
    proc = subprocess.run(cmd, cwd=PAPER_DIR)

    log_path = PAPER_DIR / (Path(tex).stem + ".log")
    pdf_path = PAPER_DIR / (Path(tex).stem + ".pdf")

    if proc.returncode != 0:
        print_log_errors(log_path)
        print(f"\nBUILD FAILED (latexmk exit {proc.returncode}). "
              f"See {log_path} for the full log.", file=sys.stderr)
        return proc.returncode

    # Report page count if available from the log.
    pages = ""
    if log_path.is_file():
        m = re.search(r"Output written on .*\((\d+) page", log_path.read_text(errors="replace"))
        if m:
            pages = f" ({m.group(1)} pages)"
    print(f"\nBuild OK -> {pdf_path}{pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
