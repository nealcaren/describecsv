from typing import Dict, Any

def create_markdown_output(analysis: Dict[str, Any]) -> str:
    """
    Generates a markdown description from the CSV analysis.

    Args:
        analysis: The dictionary output from analyze_csv.

    Returns:
        A string containing the markdown description.
    """
    markdown = ""

    # Basic File Information
    markdown += "# CSV File Analysis\n\n"
    markdown += f"## File: {analysis['basic_info']['file_info']['file_name']}\n\n"
    markdown += f"- **Directory:** {analysis['basic_info']['file_info']['directory']}\n"
    markdown += f"- **Size:** {analysis['basic_info']['file_info']['size_mb']} MB\n"
    markdown += f"- **Encoding:** {analysis['basic_info']['file_info']['encoding']}\n"
    markdown += f"- **Created Date:** {analysis['basic_info']['file_info']['created_date']}\n"
    markdown += f"- **Modified Date:** {analysis['basic_info']['file_info']['modified_date']}\n\n"

    # Basic Statistics
    markdown += "## Basic Statistics\n\n"
    markdown += f"- **Number of Rows:** {analysis['basic_info']['num_rows']}\n"
    markdown += f"- **Number of Columns:** {analysis['basic_info']['num_columns']}\n"
    markdown += f"- **Total Cells:** {analysis['basic_info']['total_cells']}\n"
    markdown += f"- **Missing Cells:** {analysis['basic_info']['missing_cells']} ({analysis['basic_info']['missing_percentage']}%) \n"
    markdown += f"- **Duplicate Rows:** {analysis['basic_info']['duplicate_rows']} ({analysis['basic_info']['duplicate_percentage']}%) \n\n"

    # Column Analysis
    markdown += "## Column Analysis\n\n"
    for col_name, col_data in analysis['column_analysis'].items():
        markdown += f"### Column: {col_name}\n\n"
        markdown += f"- **Data Type:** {col_data['data_type']}\n"
        markdown += f"- **Unique Values:** {col_data['unique_value_count']}\n"
        markdown += f"- **Missing Values:** {col_data['missing_value_count']} ({col_data['missing_percentage']}%) \n"

        if 'mean_value' in col_data:
            markdown += f"- **Mean:** {col_data['mean_value']}\n"
        if 'std_dev' in col_data:
            markdown += f"- **Standard Deviation:** {col_data['std_dev']}\n"
        if 'min_value' in col_data:
            markdown += f"- **Minimum Value:** {col_data['min_value']}\n"
        if 'max_value' in col_data:
            markdown += f"- **Maximum Value:** {col_data['max_value']}\n"
        if 'median' in col_data:
            markdown += f"- **Median:** {col_data['median']}\n"
        if 'true_count' in col_data:
            markdown += f"- **True Count (Boolean):** {col_data['true_count']}\n"
        if 'false_count' in col_data:
            markdown += f"- **False Count (Boolean):** {col_data['false_count']}\n"

        if 'top_3_values' in col_data:
            markdown += "- **Top 3 Values:**\n"
            for value, count in col_data['top_3_values'].items():
                markdown += f"  - {value}: {count}\n"
        if 'mode_value' in col_data:
            markdown += f"- **Mode:** {col_data['mode_value']}\n"
        if 'top_3_percentage' in col_data:
            markdown += f"- **Top 3 Values Percentage of Total:** {col_data['top_3_percentage']}%\n"
        if 'data_quality_note' in col_data:
            markdown += f"- **Data Quality Note:** {col_data['data_quality_note']}\n"
        if 'numeric_summary' in col_data:
            markdown += "- **Numeric Summary (of numeric values in string column):\n"
            markdown += f"  - **Mean:** {col_data['numeric_summary']['mean_value']}\n"
            markdown += f"  - **Standard Deviation:** {col_data['numeric_summary']['std_dev']}\n"
            markdown += f"  - **Minimum Value:** {col_data['numeric_summary']['min_value']}\n"
            markdown += f"  - **Maximum Value:** {col_data['numeric_summary']['max_value']}\n"
            markdown += f"  - **Median:** {col_data['numeric_summary']['median']}\n"
        if 'non_numeric_values' in col_data:
            markdown += "- **Non-Numeric Values (and counts):\n"
            for value, count in col_data['non_numeric_values'].items():
                markdown += f"  - {value}: {count}\n"

        markdown += "\n"

    return markdown
