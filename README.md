# DescribeCSV

[![PyPI version](https://badge.fury.io/py/describecsv.svg)](https://badge.fury.io/py/describecsv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Python tool for analyzing and describing CSV files. It provides detailed information about file structure, data types, missing values, and statistical summaries. Perfect for initial data exploration and quality assessment of large CSV files.

## Features

- Automatic encoding detection and handling
- Memory-efficient processing of large files through chunking
- Comprehensive column analysis including:
  - Data types and structure
  - Missing value detection and statistics
  - Unique value counts and distributions
  - Statistical summaries for numeric columns
  - Most frequent values for categorical columns
- Smart detection of numeric data stored as strings
- Duplicate row detection and counting
- Detailed file metadata information

## Installation

You can install describecsv using pip:

```bash
pip install describecsv
```

Or using uv for faster installation:

```bash
uv tool install describecsv
```

## Usage

From the command line:

```bash
describecsv path/to/your/your_file.csv
```

This will create a JSON file named `your_file.json` in the same directory as your CSV file.

## Output Example

The tool generates a detailed JSON report. Here's a sample of what you'll get:

```json
{
  "basic_info": {
    "file_info": {
      "file_name": "your_file.csv",
      "size_mb": 125.4,
      "created_date": "2024-02-21T10:30:00",
      "encoding": "utf-8"
    },
    "num_rows": 100000,
    "num_columns": 15,
    "missing_cells": 1234,
    "missing_percentage": 0.82,
    "duplicate_rows": 42,
    "duplicate_percentage": 0.042
  },
  "column_analysis": {
    "age": {
      "data_type": "int64",
      "unique_value_count": 75,
      "missing_value_count": 12,
      "mean_value": 34.5,
      "std_dev": 12.8,
      "min_value": 18.0,
      "max_value": 99.0
    },
    "category": {
      "data_type": "object",
      "unique_value_count": 5,
      "missing_value_count": 0,
      "top_3_values": {
        "A": 45000,
        "B": 30000,
        "C": 25000
      },
      "optimization_suggestion": "Consider using category dtype"
    }
  }
}
```

## Features in Detail

### Encoding Detection
- Automatically detects file encoding
- Handles common encodings (UTF-8, Latin-1, etc.)
- Provides fallback options for difficult files

### Memory Efficiency
- Processes files in chunks
- Optimizes data types automatically
- Suitable for large CSV files

### Data Quality Checks
- Identifies potential data type mismatches
- Suggests optimizations for categorical columns
- Reports duplicate rows and missing values

### Statistical Analysis
- Comprehensive numeric column statistics
- Frequency analysis for categorical data
- Missing value patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
