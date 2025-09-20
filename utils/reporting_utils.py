"""
Reporting utility functions equivalent to SAS %dmcas_report functionality.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from io import StringIO


def create_summary_report(df: pd.DataFrame, 
                         title: str = "Data Summary Report",
                         include_plots: bool = True) -> Dict[str, Any]:
    """
    Create a comprehensive summary report similar to SAS PROC CONTENTS and PROC MEANS.
    
    Args:
        df: DataFrame to summarize
        title: Title for the report
        include_plots: Whether to include visualizations
        
    Returns:
        Dictionary containing report sections
    """
    report = {
        'title': title,
        'dataset_info': _get_dataset_info(df),
        'numeric_summary': _get_numeric_summary(df),
        'categorical_summary': _get_categorical_summary(df),
        'missing_values': _get_missing_values_summary(df)
    }
    
    if include_plots:
        report['plots'] = _create_summary_plots(df)
    
    return report


def _get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic dataset information."""
    return {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'column_names': list(df.columns)
    }


def _get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for numeric columns."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols].describe()
    return pd.DataFrame()


def _get_categorical_summary(df: pd.DataFrame) -> Dict[str, Dict]:
    """Get summary for categorical columns."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    summary = {}
    
    for col in categorical_cols:
        summary[col] = {
            'unique_count': df[col].nunique(),
            'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'value_counts': df[col].value_counts().head(10).to_dict()
        }
    
    return summary


def _get_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get missing values summary."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    summary = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    
    return summary[summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)


def _create_summary_plots(df: pd.DataFrame) -> Dict[str, str]:
    """Create summary visualizations."""
    plots = {}
    
    # Distribution plots for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(min(len(numeric_cols), 4), 1, figsize=(10, 12))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols[:4]):
            df[col].hist(ax=axes[i], bins=30, alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plots['numeric_distributions'] = "Generated histogram plots for numeric variables"
        plt.close()
    
    return plots
