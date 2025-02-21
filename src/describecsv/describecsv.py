# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0.0",
#     "chardet>=5.0.0",
#     "tqdm>=4.65.0",
#     "numpy>=1.24.0",
#     "python-dateutil>=2.8.0",
#     "pytz>=2023.3",
#     "tzdata>=2023.3"
# ]
# ///

import pandas as pd
import chardet
import json
from typing import Dict, Any, Optional, Generator
from pathlib import Path
from tqdm import tqdm

def detect_encoding(file_path: Path, sample_size: int = 100000) -> str:
    """
    Detect file encoding from a sample of the file.
    
    Args:
        file_path: Path to the file
        sample_size: Number of bytes to read for detection
        
    Returns:
        str: Detected encoding
    """
    common_encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'ascii']
    
    # First try chardet
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(sample_size)
            result = chardet.detect(raw_data)
            if result['confidence'] > 0.8:
                return result['encoding']
    except Exception:
        pass

    # If chardet fails or has low confidence, try common encodings
    for encoding in common_encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                file.read(sample_size)
                return encoding
        except UnicodeDecodeError:
            continue
    
    # If all else fails, return latin1 which can read any byte stream
    return 'latin1'

def process_csv_chunks(file_path: Path, encoding: str, chunk_size: int = 50000) -> Generator[pd.DataFrame, None, None]:
    """
    Process CSV file in chunks to handle large files efficiently.
    
    Args:
        file_path: Path to the CSV file
        encoding: File encoding
        chunk_size: Number of rows per chunk
        
    Yields:
        pd.DataFrame: Each chunk of the CSV file
    """
    try:
        chunks = pd.read_csv(
            file_path,
            encoding=encoding,
            chunksize=chunk_size,
            low_memory=False,
            on_bad_lines='warn'
        )
        for chunk in chunks:
            yield chunk
    except pd.errors.ParserError:
        raise pd.errors.ParserError(f"Could not parse CSV at {file_path}. Check the file format.")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV: {e}")

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting appropriate columns to categories.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Optimized DataFrame
    """
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:  # Less than 5% unique values
                df[col] = df[col].astype('category')
    return df

def analyze_csv(file_path: str) -> Dict[str, Any]:
    """
    Load and analyze a CSV file, handling different encodings and large files efficiently.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dict[str, Any]: Analysis results in a structured format
    """
    file_path = Path(file_path)
    
    # Validate file
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != '.csv':
        raise ValueError(f"File {file_path} is not a CSV file")
        
    # Detect encoding from file sample
    encoding = detect_encoding(file_path)
    
    # Initialize aggregation variables
    total_rows = 0
    total_missing = 0
    column_stats = {}
    
    # Process file in chunks
    chunks = process_csv_chunks(file_path, encoding)
    first_chunk = True
    
    for chunk in tqdm(chunks, desc="Processing chunks"):
        if first_chunk:
            chunk = optimize_dtypes(chunk)
            columns = chunk.columns
            first_chunk = False
            
        total_rows += len(chunk)
        total_missing += chunk.isna().sum().sum()
        
        # Update column statistics
        for col in columns:
            if col not in column_stats:
                column_stats[col] = {
                    "data_type": str(chunk[col].dtype),
                    "unique_values": set(),
                    "missing_count": 0,
                    "numeric_values": [] if pd.api.types.is_numeric_dtype(chunk[col]) else None,
                    "value_counts": {} if pd.api.types.is_string_dtype(chunk[col]) else None
                }
            
            stats = column_stats[col]
            stats["missing_count"] += chunk[col].isna().sum()
            
            if pd.api.types.is_numeric_dtype(chunk[col]):
                valid_data = chunk[col].dropna()
                if len(valid_data) > 0:
                    stats["numeric_values"].extend(valid_data)
            
            elif pd.api.types.is_string_dtype(chunk[col]):
                value_counts = chunk[col].value_counts()
                for val, count in value_counts.items():
                    stats["value_counts"][val] = stats["value_counts"].get(val, 0) + count
                stats["unique_values"].update(chunk[col].dropna().unique())
    
    # Compile final analysis
    analysis = {
        "basic_info": {
            "num_rows": total_rows,
            "num_columns": len(columns),
            "total_cells": total_rows * len(columns),
            "missing_cells": int(total_missing),
            "missing_percentage": round((total_missing / (total_rows * len(columns))) * 100, 2)
        },
        "column_analysis": {}
    }
    
    # Process column statistics
    for col, stats in column_stats.items():
        col_analysis = {
            "data_type": stats["data_type"],
            "unique_value_count": len(stats["unique_values"]),
            "missing_value_count": stats["missing_count"],
            "missing_percentage": round((stats["missing_count"] / total_rows) * 100, 2)
        }
        
        if stats["numeric_values"] is not None and stats["numeric_values"]:
            numeric_series = pd.Series(stats["numeric_values"])
            col_analysis.update({
                "mean_value": round(numeric_series.mean(), 2),
                "std_dev": round(numeric_series.std(), 2),
                "min_value": round(numeric_series.min(), 2),
                "max_value": round(numeric_series.max(), 2),
                "median": round(numeric_series.median(), 2)
            })
        
        elif stats["value_counts"]:
            sorted_values = sorted(stats["value_counts"].items(), key=lambda x: x[1], reverse=True)
            top_3 = dict(sorted_values[:3])
            top_3_sum = sum(top_3.values())
            
            col_analysis.update({
                "top_3_values": {str(k): int(v) for k, v in top_3.items()},
                "mode_value": str(sorted_values[0][0]),
                "top_3_percentage": round((top_3_sum / total_rows) * 100, 2)
            })
            
            if len(stats["unique_values"]) < total_rows * 0.05:
                col_analysis["optimization_suggestion"] = "Consider using category dtype"
        
        analysis["column_analysis"][col] = col_analysis
    
    return analysis

def main(file_path: str) -> None:
    """Main function to run the analysis and handle errors."""
    try:
        result = analyze_csv(file_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error analyzing CSV: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    
    main(sys.argv[1])
