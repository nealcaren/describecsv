<<<<<<< HEAD
# describecsv
A Python tool for analyzing and describing CSV files
=======
# DescribeCSV

A Python tool for analyzing and describing CSV files. It provides detailed information about file structure, data types, missing values, and statistical summaries.

## Features

- Automatic encoding detection
- Handles large files through chunked processing
- Detailed column analysis including:
  - Data types
  - Missing values
  - Unique value counts
  - Statistical summaries for numeric columns
  - Top values for categorical columns
- Detection of numeric data stored as strings
- Duplicate row detection
- File metadata information

## Installation

```bash
pip install describecsv
```

## Usage

From the command line:

```bash
describecsv path/to/your/file.csv
```

This will create a JSON file with the analysis results in the same directory as your CSV file.

## Output

The tool generates a detailed JSON report including:

- Basic file information (size, encoding, etc.)
- Row and column counts
- Missing value analysis
- Column-by-column analysis including:
  - Data types
  - Unique values
  - Missing values
  - Statistical summaries for numeric columns
  - Most common values for categorical columns
  - Suggestions for data quality improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.
>>>>>>> 549642f (build: Add PyPI packaging configuration and documentation)
