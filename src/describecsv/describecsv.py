
import pandas as pd
import chardet
import json
import os
from datetime import datetime
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
    # First try the detected encoding
    try:
        chunks = pd.read_csv(
            file_path,
            encoding=encoding,
            chunksize=chunk_size,
            low_memory=False,
            on_bad_lines='warn'
        )
        # Test if we can actually read with this encoding
        next(chunks)
        # If successful, reset and yield all chunks
        chunks = pd.read_csv(
            file_path,
            encoding=encoding,
            chunksize=chunk_size,
            low_memory=False,
            on_bad_lines='warn'
        )
        for chunk in chunks:
            yield chunk
        return
    except (UnicodeDecodeError, StopIteration):
        pass
    
    # If that fails, try common encodings
    for enc in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
        try:
            chunks = pd.read_csv(
                file_path,
                encoding=enc,
                chunksize=chunk_size,
                low_memory=False,
                on_bad_lines='warn'
            )
            # Test if we can actually read with this encoding
            next(chunks)
            # If successful, reset and yield all chunks
            chunks = pd.read_csv(
                file_path,
                encoding=enc,
                chunksize=chunk_size,
                low_memory=False,
                on_bad_lines='warn'
            )
            for chunk in chunks:
                yield chunk
            return
        except (UnicodeDecodeError, StopIteration):
            continue
    
    # If all else fails, try with errors='replace'
    chunks = pd.read_csv(
        file_path,
        encoding='latin1',
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines='warn',
        errors='replace'
    )
    for chunk in chunks:
        yield chunk

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
    duplicate_count = 0
    column_stats = {}
    seen_rows = set()  # For tracking duplicates
    
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
        
        # Check for duplicates
        chunk_tuples = [tuple(row) for _, row in chunk.iterrows()]
        for row in chunk_tuples:
            if row in seen_rows:
                duplicate_count += 1
            else:
                seen_rows.add(row)
        
        # Update column statistics
        for col in columns:
            if col not in column_stats:
                is_string_like = (pd.api.types.is_string_dtype(chunk[col]) or 
                                (pd.api.types.is_object_dtype(chunk[col]) and 
                                 chunk[col].dropna().apply(lambda x: isinstance(x, str)).mean() > 0.9))
                
                column_stats[col] = {
                    "data_type": str(chunk[col].dtype),
                    "unique_values": set(),
                    "missing_count": 0,
                    "numeric_values": [] if pd.api.types.is_numeric_dtype(chunk[col]) else None,
                    "value_counts": {} if is_string_like else None
                }
            
            stats = column_stats[col]
            stats["missing_count"] += chunk[col].isna().sum()
            
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(chunk[col]):
                valid_data = chunk[col].dropna()
                if len(valid_data) > 0:
                    stats["numeric_values"].extend(valid_data)
                    stats["unique_values"].update(valid_data.unique())
            
            # Handle string and object columns
            elif pd.api.types.is_string_dtype(chunk[col]) or pd.api.types.is_object_dtype(chunk[col]):
                # Try to convert to numeric if possible
                try:
                    numeric_data = pd.to_numeric(chunk[col].dropna(), errors='coerce')
                    numeric_ratio = numeric_data.notna().mean()
                    if numeric_ratio > 0.8:  # If more than 80% can be converted to numbers
                        stats["numeric_suggestion"] = "Column contains mostly numeric values"
                        if stats["numeric_values"] is None:
                            stats["numeric_values"] = []
                            stats["non_numeric_values"] = {}
                        stats["numeric_values"].extend(numeric_data.dropna())
                        # Track non-numeric values
                        non_numeric_mask = numeric_data.isna() & chunk[col].notna()
                        non_numeric_values = chunk[col][non_numeric_mask].value_counts()
                        for val, count in non_numeric_values.items():
                            stats["non_numeric_values"][val] = stats["non_numeric_values"].get(val, 0) + count
                except:
                    pass
                
                # Process as string if we have value_counts
                if stats["value_counts"] is not None:  # Only process if value_counts exists
                    value_counts = chunk[col].value_counts()
                    for val, count in value_counts.items():
                        stats["value_counts"][val] = stats["value_counts"].get(val, 0) + count
                    stats["unique_values"].update(chunk[col].dropna().unique())
    
    # Compile final analysis
    # Get file information
    file_stat = file_path.stat()
    file_info = {
        "file_name": file_path.name,
        "directory": str(file_path.parent.absolute()),
        "size_bytes": file_stat.st_size,
        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
        "created_date": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
        "modified_date": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        "encoding": encoding
    }

    analysis = {
        "basic_info": {
            "file_info": file_info,
            "num_rows": int(total_rows),
            "num_columns": int(len(columns)),
            "total_cells": int(total_rows * len(columns)),
            "missing_cells": int(total_missing),
            "missing_percentage": float(round((total_missing / (total_rows * len(columns))) * 100, 2)),
            "duplicate_rows": int(duplicate_count),
            "duplicate_percentage": float(round((duplicate_count / total_rows) * 100, 2))
        },
        "column_analysis": {}
    }
    
    # Process column statistics
    for col, stats in column_stats.items():
        col_analysis = {
            "data_type": stats["data_type"],
            "unique_value_count": int(len(stats["unique_values"])),
            "missing_value_count": int(stats["missing_count"]),
            "missing_percentage": float(round((stats["missing_count"] / total_rows) * 100, 2))
        }
        
        if stats["numeric_values"] is not None and stats["numeric_values"]:
            numeric_series = pd.Series(stats["numeric_values"])
            if pd.api.types.is_bool_dtype(numeric_series):
                col_analysis.update({
                    "mean_value": float(numeric_series.mean()),
                    "min_value": bool(numeric_series.min()),
                    "max_value": bool(numeric_series.max()),
                    "true_count": int(numeric_series.sum()),
                    "false_count": int(len(numeric_series) - numeric_series.sum())
                })
            else:
                col_analysis.update({
                    "mean_value": float(round(numeric_series.mean(), 2)),
                    "std_dev": float(round(numeric_series.std(), 2)),
                    "min_value": float(round(numeric_series.min(), 2)),
                    "max_value": float(round(numeric_series.max(), 2)),
                    "median": float(round(numeric_series.median(), 2))
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
            
            if "numeric_suggestion" in stats:
                col_analysis["data_quality_note"] = "Column contains mostly numeric values stored as strings"
                numeric_series = pd.Series(stats["numeric_values"])
                col_analysis["numeric_summary"] = {
                    "mean_value": float(round(numeric_series.mean(), 2)),
                    "std_dev": float(round(numeric_series.std(), 2)),
                    "min_value": float(round(numeric_series.min(), 2)),
                    "max_value": float(round(numeric_series.max(), 2)),
                    "median": float(round(numeric_series.median(), 2))
                }
                # Add non-numeric value counts
                if stats["non_numeric_values"]:
                    sorted_non_numeric = sorted(stats["non_numeric_values"].items(), key=lambda x: x[1], reverse=True)
                    col_analysis["non_numeric_values"] = {
                        str(k): int(v) for k, v in sorted_non_numeric
                    }
        
        analysis["column_analysis"][col] = col_analysis
    
    return analysis

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python describecsv.py <path_to_csv>")
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
