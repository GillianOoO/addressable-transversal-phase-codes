"""Command-line entry point for the paper's finite cyclic-ring construction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from exponential.construction import main

if __name__ == "__main__":
    main()
