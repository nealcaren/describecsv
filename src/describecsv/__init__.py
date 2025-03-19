"""A tool for analyzing and describing CSV files."""

from .describecsv import analyze_csv
from .markdown_output import create_markdown_output
import sys
from pathlib import Path
import json

__version__ = "1.0.0"

def cli():
    if len(sys.argv) < 2:
        print("Usage: describecsv <path_to_csv> [--format json|markdown]")
        sys.exit(1)

    file_path = sys.argv[1]
    output_format = "markdown"  # Default to markdown

    if len(sys.argv) > 2:
        if sys.argv[2] == "--format":
            if len(sys.argv) > 3 and sys.argv[3] in ["json", "markdown"]:
                output_format = sys.argv[3]
            else:
                print("Invalid format. Use 'json' or 'markdown'.")
                sys.exit(1)

    try:
        result = analyze_csv(file_path, output_format=output_format)

        # Create output filename based on format
        input_path = Path(file_path)
        if output_format == "json":
            output_path = input_path.with_name(f"{input_path.stem}_details.json")
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        elif output_format == "markdown":
            output_path = input_path.with_name(f"{input_path.stem}_details.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)


        print(f"Analysis saved to: {output_path}")

    except Exception as e:
        print(f"Error analyzing CSV: {e}")
        sys.exit(1)

__all__ = ['cli', 'analyze_csv', 'create_markdown_output']
