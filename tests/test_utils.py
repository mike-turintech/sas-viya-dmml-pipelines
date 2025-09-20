"""
Tests for utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from utils.data_utils import validate_data
from utils.reporting_utils import create_summary_report


def test_validate_data_valid():
    """Test data validation with valid DataFrame."""
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    
    result = validate_data(df, required_columns=['a', 'b'])
    assert result['is_valid'] is True
    assert len(result['errors']) == 0


def test_validate_data_missing_columns():
    """Test data validation with missing required columns."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    
    result = validate_data(df, required_columns=['a', 'b'])
    assert result['is_valid'] is False
    assert 'Missing required columns' in result['errors'][0]


def test_validate_data_insufficient_rows():
    """Test data validation with insufficient rows."""
    df = pd.DataFrame({'a': [1]})
    
    result = validate_data(df, min_rows=5)
    assert result['is_valid'] is False
    assert 'minimum 5 required' in result['errors'][0]


def test_create_summary_report():
    """Test summary report creation."""
    df = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'categorical_col': ['A', 'B', 'A', 'C', 'B'],
        'missing_col': [1, None, 3, None, 5]
    })
    
    report = create_summary_report(df, include_plots=False)
    
    assert 'title' in report
    assert 'dataset_info' in report
    assert 'numeric_summary' in report
    assert 'categorical_summary' in report
    assert 'missing_values' in report
    
    # Check dataset info
    assert report['dataset_info']['shape'] == (5, 3)
    
    # Check missing values summary
    assert len(report['missing_values']) > 0
    assert 'missing_col' in report['missing_values'].index


if __name__ == '__main__':
    pytest.main([__file__])
