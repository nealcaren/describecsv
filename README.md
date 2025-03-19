# DescribeCSV

[![PyPI version](https://badge.fury.io/py/describecsv.svg)](https://badge.fury.io/py/describecsv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Python tool for analyzing and describing CSV files. It provides detailed information about file structure, data types, missing values, and statistical summaries. It defaults to producing a markdown description, but can also produce JSON. Perfect for initial data exploration and quality assessment of large CSV files.

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

By default, `describecsv` will output a markdown file with a description of the CSV file.

```bash
describecsv path/to/your/your_file.csv
```
This will create a markdown file named `your_file_details.md` in the same directory as your CSV file.

You can also specify the output format:

```bash
describecsv path/to/your/your_file.csv --format json
```

```bash
describecsv path/to/your/your_file.csv --format markdown
```

## Output Example

The tool generates a detailed markdown report. Here's a sample of what you'll get:

```
# CSV File Analysis

## File: your_file.csv

- **Directory:** /path/to/your
- **Size:** 125.4 MB
- **Encoding:** utf-8
- **Created Date:** 2024-02-21T10:30:00
- **Modified Date:** 2024-02-21T10:30:00

## Basic Statistics

- **Number of Rows:** 100000
- **Number of Columns:** 15
- **Total Cells:** 1500000
- **Missing Cells:** 1234 (0.82%)
- **Duplicate Rows:** 42 (0.042%)

## Column Analysis

### Column: age

- **Data Type:** int64
- **Unique Values:** 75
- **Missing Values:** 12 (0.012%)
- **Mean:** 34.5
- **Standard Deviation:** 12.8
- **Minimum Value:** 18.0
- **Maximum Value:** 99.0
- **Median:** 32

### Column: category

- **Data Type:** object
- **Unique Values:** 5
- **Missing Values:** 0 (0.0%)
- **Top 3 Values:**
  - A: 45000
  - B: 30000
  - C: 25000
- **Mode:** A
- **Top 3 Values Percentage of Total:** 95.0%
```

The tool can also generate a detailed JSON report.

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
- Reports duplicate rows and missing values

### Statistical Analysis

-   Comprehensive numeric column statistics
-   Frequency analysis for categorical data
-   Missing value patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
