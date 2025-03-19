"""A tool for analyzing and describing CSV files."""

from .describecsv import analyze_csv
import sys
from pathlib import Path
import json

__version__ = "0.5.0"

def cli():
    if len(sys.argv) != 2:
        print("Usage: describecsv <path_to_csv>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    try:
        result = analyze_csv(file_path)
        
        # Create output filename
        input_path = Path(file_path)
        output_path = input_path.with_name(f"{input_path.stem}_details.json")
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        print(f"Analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"Error analyzing CSV: {e}")
        sys.exit(1)

__all__ = ['cli', 'analyze_csv']
