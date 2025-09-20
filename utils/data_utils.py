"""
Data utility functions for loading and validating datasets.
"""

import pandas as pd
from typing import Optional, Dict, Any
import os


def load_sample_data(filename: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Load sample data from the data directory.
    
    Args:
        filename: Name of the data file to load
        data_dir: Directory containing the data files
        
    Returns:
        pandas DataFrame containing the loaded data
    """
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Determine file type and load accordingly
    if filename.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filename.endswith('.parquet'):
        return pd.read_parquet(filepath)
    elif filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filename}")


def validate_data(df: pd.DataFrame, 
                  required_columns: Optional[list] = None,
                  min_rows: int = 1) -> Dict[str, Any]:
    """
    Validate a DataFrame meets basic requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict()
        }
    }
    
    # Check minimum rows
    if len(df) < min_rows:
        results['is_valid'] = False
        results['errors'].append(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            results['is_valid'] = False
            results['errors'].append(f"Missing required columns: {list(missing_cols)}")
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        results['warnings'].append(f"Columns with all null values: {empty_cols}")
    
    return results
