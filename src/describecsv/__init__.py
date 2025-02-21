"""A tool for analyzing and describing CSV files."""

from .describecsv import main

__version__ = "0.1.1"

def cli():
    """Entry point for the command line interface."""
    import sys
    if len(sys.argv) != 2:
        print("Usage: describecsv <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])

__all__ = ['main', 'cli']
