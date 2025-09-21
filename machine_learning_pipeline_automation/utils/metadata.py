"""
Metadata management utilities for variable tracking and data validation.

This module provides the MetadataHandler class for tracking variable types,
roles, and properties, with automatic detection and SAS-compatible handling.
"""

import logging
from typing import Union, Optional, Dict, List, Any, Set, Tuple
import warnings
import re
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataError(Exception):
    """Custom exception for metadata operations."""
    pass


class VariableInfo:
    """Container for variable metadata information."""
    
    def __init__(self, name: str, dtype: str, role: str = 'INPUT'):
        self.name = name
        self.dtype = dtype  # 'NUMERIC', 'CATEGORICAL', 'DATETIME', 'TEXT'
        self.role = role    # 'TARGET', 'INPUT', 'ID', 'REJECT'
        self.levels = None  # For categorical variables
        self.missing_count = 0
        self.missing_proportion = 0.0
        self.unique_count = 0
        self.min_value = None
        self.max_value = None
        self.mean_value = None
        self.std_value = None
        self.encoding_info = {}
        self.validation_rules = []
        self.original_dtype = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'dtype': self.dtype,
            'role': self.role,
            'levels': self.levels,
            'missing_count': self.missing_count,
            'missing_proportion': self.missing_proportion,
            'unique_count': self.unique_count,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            'encoding_info': self.encoding_info,
            'validation_rules': self.validation_rules,
            'original_dtype': str(self.original_dtype) if self.original_dtype else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VariableInfo':
        """Create from dictionary representation."""
        var_info = cls(data['name'], data['dtype'], data.get('role', 'INPUT'))
        var_info.levels = data.get('levels')
        var_info.missing_count = data.get('missing_count', 0)
        var_info.missing_proportion = data.get('missing_proportion', 0.0)
        var_info.unique_count = data.get('unique_count', 0)
        var_info.min_value = data.get('min_value')
        var_info.max_value = data.get('max_value')
        var_info.mean_value = data.get('mean_value')
        var_info.std_value = data.get('std_value')
        var_info.encoding_info = data.get('encoding_info', {})
        var_info.validation_rules = data.get('validation_rules', [])
        var_info.original_dtype = data.get('original_dtype')
        return var_info


class MetadataHandler:
    """
    Handler for tracking and managing variable metadata.
    
    Provides automatic variable type detection, role assignment,
    and comprehensive metadata management for SAS-compatible operations.
    """
    
    def __init__(self):
        """Initialize MetadataHandler."""
        self.variables = {}  # Dict[str, VariableInfo]
        self._naming_rules = {
            'max_length': 32,
            'valid_chars': r'[A-Za-z0-9_]',
            'start_with_alpha': True,
            'reserved_words': {
                'DATA', 'SET', 'VAR', 'RUN', 'PROC', 'IF', 'THEN', 'ELSE',
                'DO', 'END', 'BY', 'WHERE', 'CLASS', 'MODEL', 'OUTPUT'
            }
        }
        self._type_detection_config = {
            'datetime_formats': [
                '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S', '%Y/%m/%d', '%d-%m-%Y'
            ],
            'categorical_threshold': 0.05,  # If unique values / total < threshold, treat as categorical
            'max_categorical_levels': 50,   # Maximum levels for categorical
            'min_numeric_unique': 10        # Minimum unique values to consider numeric
        }
    
    def analyze_dataframe(self, data: pd.DataFrame, 
                         target_column: Optional[str] = None,
                         id_columns: Optional[List[str]] = None,
                         reject_columns: Optional[List[str]] = None) -> None:
        """
        Analyze DataFrame and extract variable metadata.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            id_columns: List of ID variable names
            reject_columns: List of variables to reject/ignore
        """
        logger.info(f"Analyzing DataFrame with {len(data)} rows and {len(data.columns)} columns")
        
        id_columns = id_columns or []
        reject_columns = reject_columns or []
        
        for column in data.columns:
            # Determine role
            if column == target_column:
                role = 'TARGET'
            elif column in id_columns:
                role = 'ID'
            elif column in reject_columns:
                role = 'REJECT'
            else:
                role = 'INPUT'
            
            # Detect variable type
            var_type = self._detect_variable_type(data[column])
            
            # Create variable info
            var_info = VariableInfo(column, var_type, role)
            var_info.original_dtype = data[column].dtype
            
            # Calculate statistics
            self._calculate_variable_statistics(data[column], var_info)
            
            # Store variable info
            self.variables[column] = var_info
        
        # Validate variable names
        self._validate_variable_names()
        
        logger.info(f"Analysis complete. Found {len(self.get_variables_by_type('NUMERIC'))} numeric, "
                   f"{len(self.get_variables_by_type('CATEGORICAL'))} categorical, "
                   f"{len(self.get_variables_by_type('DATETIME'))} datetime, "
                   f"{len(self.get_variables_by_type('TEXT'))} text variables")
    
    def _detect_variable_type(self, series: pd.Series) -> str:
        """Detect variable type from pandas Series."""
        # Handle completely missing data
        if series.isnull().all():
            return 'TEXT'
        
        # Get non-null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'TEXT'
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'DATETIME'
        
        # Try to detect datetime from strings
        if series.dtype == 'object':
            if self._is_datetime_string(non_null_series):
                return 'DATETIME'
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if it should be treated as categorical
            unique_count = series.nunique()
            total_count = len(series)
            
            if (unique_count / total_count <= self._type_detection_config['categorical_threshold'] and 
                unique_count <= self._type_detection_config['max_categorical_levels']):
                return 'CATEGORICAL'
            else:
                return 'NUMERIC'
        
        # Check for categorical (object/string type)
        if series.dtype == 'object':
            unique_count = series.nunique()
            
            # If too many unique values, treat as text
            if unique_count > self._type_detection_config['max_categorical_levels']:
                return 'TEXT'
            else:
                return 'CATEGORICAL'
        
        # Check for boolean
        if pd.api.types.is_bool_dtype(series):
            return 'CATEGORICAL'
        
        # Default to text
        return 'TEXT'
    
    def _is_datetime_string(self, series: pd.Series) -> bool:
        """Check if string series contains datetime values."""
        sample_size = min(100, len(series))
        sample = series.sample(n=sample_size).astype(str)
        
        datetime_count = 0
        for date_format in self._type_detection_config['datetime_formats']:
            try:
                pd.to_datetime(sample, format=date_format, errors='coerce')
                parsed = pd.to_datetime(sample, format=date_format, errors='coerce')
                datetime_count = parsed.notna().sum()
                if datetime_count / sample_size > 0.8:  # 80% successfully parsed
                    return True
            except:
                continue
        
        # Try without explicit format
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            datetime_count = parsed.notna().sum()
            if datetime_count / sample_size > 0.8:
                return True
        except:
            pass
        
        return False
    
    def _calculate_variable_statistics(self, series: pd.Series, var_info: VariableInfo) -> None:
        """Calculate statistics for a variable."""
        # Basic statistics
        var_info.missing_count = series.isnull().sum()
        var_info.missing_proportion = var_info.missing_count / len(series)
        var_info.unique_count = series.nunique()
        
        # Type-specific statistics
        if var_info.dtype == 'NUMERIC':
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                var_info.min_value = float(non_null_series.min())
                var_info.max_value = float(non_null_series.max())
                var_info.mean_value = float(non_null_series.mean())
                var_info.std_value = float(non_null_series.std())
        
        elif var_info.dtype == 'CATEGORICAL':
            # Get levels (unique values)
            levels = series.dropna().unique().tolist()
            var_info.levels = sorted([str(level) for level in levels])
        
        elif var_info.dtype == 'DATETIME':
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(non_null_series):
                    non_null_series = pd.to_datetime(non_null_series, errors='coerce')
                    non_null_series = non_null_series.dropna()
                
                if len(non_null_series) > 0:
                    var_info.min_value = non_null_series.min()
                    var_info.max_value = non_null_series.max()
    
    def _validate_variable_names(self) -> None:
        """Validate variable names against SAS naming conventions."""
        issues = []
        
        for var_name, var_info in self.variables.items():
            # Check length
            if len(var_name) > self._naming_rules['max_length']:
                issues.append(f"'{var_name}': exceeds maximum length of {self._naming_rules['max_length']}")
            
            # Check valid characters
            if not re.match(f"^{self._naming_rules['valid_chars']}+$", var_name):
                issues.append(f"'{var_name}': contains invalid characters")
            
            # Check starts with alpha
            if self._naming_rules['start_with_alpha'] and not var_name[0].isalpha():
                issues.append(f"'{var_name}': must start with a letter")
            
            # Check reserved words
            if var_name.upper() in self._naming_rules['reserved_words']:
                issues.append(f"'{var_name}': is a reserved word")
        
        if issues:
            warning_msg = "Variable naming issues found:\n" + "\n".join(issues)
            warnings.warn(warning_msg, UserWarning)
    
    def get_variables_by_role(self, role: str) -> List[str]:
        """Get variable names by role."""
        return [name for name, var_info in self.variables.items() if var_info.role == role]
    
    def get_variables_by_type(self, var_type: str) -> List[str]:
        """Get variable names by type."""
        return [name for name, var_info in self.variables.items() if var_info.dtype == var_type]
    
    def set_variable_role(self, variable_name: str, role: str) -> None:
        """Set role for a variable."""
        if variable_name not in self.variables:
            raise MetadataError(f"Variable '{variable_name}' not found")
        
        valid_roles = {'TARGET', 'INPUT', 'ID', 'REJECT'}
        if role not in valid_roles:
            raise MetadataError(f"Invalid role '{role}'. Must be one of: {valid_roles}")
        
        self.variables[variable_name].role = role
        logger.info(f"Set role for '{variable_name}' to '{role}'")
    
    def set_variable_type(self, variable_name: str, var_type: str) -> None:
        """Set type for a variable."""
        if variable_name not in self.variables:
            raise MetadataError(f"Variable '{variable_name}' not found")
        
        valid_types = {'NUMERIC', 'CATEGORICAL', 'DATETIME', 'TEXT'}
        if var_type not in valid_types:
            raise MetadataError(f"Invalid type '{var_type}'. Must be one of: {valid_types}")
        
        old_type = self.variables[variable_name].dtype
        self.variables[variable_name].dtype = var_type
        logger.info(f"Changed type for '{variable_name}' from '{old_type}' to '{var_type}'")
    
    def add_validation_rule(self, variable_name: str, rule: Dict[str, Any]) -> None:
        """Add validation rule for a variable."""
        if variable_name not in self.variables:
            raise MetadataError(f"Variable '{variable_name}' not found")
        
        self.variables[variable_name].validation_rules.append(rule)
        logger.info(f"Added validation rule for '{variable_name}': {rule}")
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate DataFrame against stored metadata.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check for missing variables
        missing_vars = set(self.variables.keys()) - set(data.columns)
        if missing_vars:
            validation_results['errors'].append(f"Missing variables: {missing_vars}")
        
        # Check for extra variables
        extra_vars = set(data.columns) - set(self.variables.keys())
        if extra_vars:
            validation_results['warnings'].append(f"Extra variables found: {extra_vars}")
        
        # Validate each variable
        for var_name, var_info in self.variables.items():
            if var_name not in data.columns:
                continue
            
            series = data[var_name]
            
            # Check missing values
            missing_prop = series.isnull().sum() / len(series)
            if missing_prop > var_info.missing_proportion + 0.1:  # 10% tolerance
                validation_results['warnings'].append(
                    f"'{var_name}': higher missing rate than expected "
                    f"({missing_prop:.2%} vs {var_info.missing_proportion:.2%})"
                )
            
            # Type-specific validation
            if var_info.dtype == 'CATEGORICAL' and var_info.levels:
                unexpected_levels = set(series.dropna().astype(str).unique()) - set(var_info.levels)
                if unexpected_levels:
                    validation_results['warnings'].append(
                        f"'{var_name}': unexpected categorical levels: {unexpected_levels}"
                    )
            
            elif var_info.dtype == 'NUMERIC':
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    if var_info.min_value is not None and non_null_series.min() < var_info.min_value:
                        validation_results['warnings'].append(
                            f"'{var_name}': values below expected minimum ({var_info.min_value})"
                        )
                    if var_info.max_value is not None and non_null_series.max() > var_info.max_value:
                        validation_results['warnings'].append(
                            f"'{var_name}': values above expected maximum ({var_info.max_value})"
                        )
            
            # Custom validation rules
            for rule in var_info.validation_rules:
                try:
                    if not self._apply_validation_rule(series, rule):
                        validation_results['errors'].append(
                            f"'{var_name}': failed validation rule: {rule}"
                        )
                except Exception as e:
                    validation_results['errors'].append(
                        f"'{var_name}': validation rule error: {e}"
                    )
        
        return validation_results
    
    def _apply_validation_rule(self, series: pd.Series, rule: Dict[str, Any]) -> bool:
        """Apply a validation rule to a series."""
        rule_type = rule.get('type')
        
        if rule_type == 'range':
            min_val, max_val = rule.get('min'), rule.get('max')
            non_null = series.dropna()
            if min_val is not None and (non_null < min_val).any():
                return False
            if max_val is not None and (non_null > max_val).any():
                return False
        
        elif rule_type == 'regex':
            pattern = rule.get('pattern')
            if pattern:
                non_null = series.dropna().astype(str)
                if not non_null.str.match(pattern).all():
                    return False
        
        elif rule_type == 'custom':
            func = rule.get('function')
            if func and callable(func):
                return func(series)
        
        return True
    
    def convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame columns to appropriate types based on metadata.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with converted types
        """
        result_data = data.copy()
        
        for var_name, var_info in self.variables.items():
            if var_name not in result_data.columns:
                continue
            
            try:
                if var_info.dtype == 'NUMERIC':
                    result_data[var_name] = pd.to_numeric(result_data[var_name], errors='coerce')
                
                elif var_info.dtype == 'CATEGORICAL':
                    result_data[var_name] = result_data[var_name].astype('category')
                
                elif var_info.dtype == 'DATETIME':
                    result_data[var_name] = pd.to_datetime(result_data[var_name], errors='coerce')
                
                elif var_info.dtype == 'TEXT':
                    result_data[var_name] = result_data[var_name].astype(str)
                
            except Exception as e:
                logger.warning(f"Failed to convert '{var_name}' to {var_info.dtype}: {e}")
        
        return result_data
    
    def handle_missing_values(self, data: pd.DataFrame, 
                             strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values based on variable types and metadata.
        
        Args:
            data: Input DataFrame
            strategy: Missing value strategy ('auto', 'drop', 'fill', 'none')
            
        Returns:
            DataFrame with handled missing values
        """
        if strategy == 'none':
            return data.copy()
        
        result_data = data.copy()
        
        for var_name, var_info in self.variables.items():
            if var_name not in result_data.columns:
                continue
            
            if strategy == 'drop':
                result_data = result_data.dropna(subset=[var_name])
            
            elif strategy in ['fill', 'auto']:
                series = result_data[var_name]
                
                if var_info.dtype == 'NUMERIC':
                    # Fill with mean for numeric variables
                    if var_info.mean_value is not None:
                        fill_value = var_info.mean_value
                    else:
                        fill_value = series.mean()
                    result_data[var_name] = series.fillna(fill_value)
                
                elif var_info.dtype == 'CATEGORICAL':
                    # Fill with mode for categorical variables
                    if not series.dropna().empty:
                        mode_value = series.mode()
                        if not mode_value.empty:
                            result_data[var_name] = series.fillna(mode_value.iloc[0])
                
                elif var_info.dtype == 'TEXT':
                    # Fill with empty string for text variables
                    result_data[var_name] = series.fillna('')
                
                elif var_info.dtype == 'DATETIME':
                    # Fill with median for datetime variables
                    if not series.dropna().empty:
                        median_value = series.dropna().median()
                        result_data[var_name] = series.fillna(median_value)
        
        return result_data
    
    def standardize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names according to SAS conventions.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        result_data = data.copy()
        name_mapping = {}
        
        for column in result_data.columns:
            # Create standardized name
            std_name = self._standardize_name(column)
            
            # Ensure uniqueness
            if std_name in name_mapping.values():
                counter = 1
                while f"{std_name}_{counter}" in name_mapping.values():
                    counter += 1
                std_name = f"{std_name}_{counter}"
            
            name_mapping[column] = std_name
        
        # Rename columns
        result_data.rename(columns=name_mapping, inplace=True)
        
        # Update metadata
        new_variables = {}
        for old_name, new_name in name_mapping.items():
            if old_name in self.variables:
                var_info = self.variables[old_name]
                var_info.name = new_name
                new_variables[new_name] = var_info
        
        self.variables = new_variables
        
        logger.info(f"Standardized {len(name_mapping)} column names")
        return result_data
    
    def _standardize_name(self, name: str) -> str:
        """Standardize a single variable name."""
        # Remove/replace invalid characters
        std_name = re.sub(r'[^A-Za-z0-9_]', '_', str(name))
        
        # Ensure starts with letter
        if not std_name[0].isalpha():
            std_name = 'VAR_' + std_name
        
        # Truncate to maximum length
        if len(std_name) > self._naming_rules['max_length']:
            std_name = std_name[:self._naming_rules['max_length']]
        
        # Check reserved words
        if std_name.upper() in self._naming_rules['reserved_words']:
            std_name = std_name + '_VAR'
        
        return std_name
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all variables."""
        summary_data = []
        
        for var_name, var_info in self.variables.items():
            summary_data.append({
                'Variable': var_name,
                'Type': var_info.dtype,
                'Role': var_info.role,
                'Missing_Count': var_info.missing_count,
                'Missing_Proportion': var_info.missing_proportion,
                'Unique_Count': var_info.unique_count,
                'Min_Value': var_info.min_value,
                'Max_Value': var_info.max_value,
                'Mean_Value': var_info.mean_value,
                'Std_Value': var_info.std_value,
                'Levels_Count': len(var_info.levels) if var_info.levels else None
            })
        
        return pd.DataFrame(summary_data)
    
    def export_metadata(self) -> Dict[str, Any]:
        """Export metadata to dictionary."""
        return {
            'variables': {name: var_info.to_dict() for name, var_info in self.variables.items()},
            'naming_rules': self._naming_rules,
            'type_detection_config': self._type_detection_config
        }
    
    def import_metadata(self, metadata: Dict[str, Any]) -> None:
        """Import metadata from dictionary."""
        self.variables = {}
        for name, var_data in metadata.get('variables', {}).items():
            self.variables[name] = VariableInfo.from_dict(var_data)
        
        if 'naming_rules' in metadata:
            self._naming_rules.update(metadata['naming_rules'])
        
        if 'type_detection_config' in metadata:
            self._type_detection_config.update(metadata['type_detection_config'])
        
        logger.info(f"Imported metadata for {len(self.variables)} variables")


# Convenience functions
def analyze_data(data: pd.DataFrame, 
                target_column: Optional[str] = None,
                id_columns: Optional[List[str]] = None,
                reject_columns: Optional[List[str]] = None) -> MetadataHandler:
    """Quick data analysis function."""
    handler = MetadataHandler()
    handler.analyze_dataframe(data, target_column, id_columns, reject_columns)
    return handler


def detect_variable_types(data: pd.DataFrame) -> Dict[str, str]:
    """Quick variable type detection."""
    handler = MetadataHandler()
    type_mapping = {}
    
    for column in data.columns:
        var_type = handler._detect_variable_type(data[column])
        type_mapping[column] = var_type
    
    return type_mapping
