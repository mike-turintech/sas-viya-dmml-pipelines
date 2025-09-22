"""
Data handling utilities for SAS to Python migration.

This package provides unified interfaces for:
- File I/O operations (CSV and Parquet)
- Data partitioning (train/test/validation splits)
- Metadata management (variable types, roles, properties)
- Configuration management (macro variables, project settings)
- SAS to Python translation utilities

Usage Examples:
    # Data loading
    from utils import DataLoader, load_data
    loader = DataLoader(cache_dir='cache')
    data = loader.load_data('data.csv')
    
    # Quick loading
    data = load_data('data.parquet')
    
    # Data partitioning
    from utils import PartitionManager, create_train_test_split
    manager = PartitionManager(random_state=42)
    train_df, test_df, val_df = manager.split_data(data, stratify_column='target')
    
    # Quick partitioning
    train_df, test_df = create_train_test_split(data, stratify_column='target')
    
    # Metadata analysis
    from utils import MetadataHandler, analyze_data
    handler = analyze_data(data, target_column='target')
    summary = handler.get_summary_statistics()
    
    # Type detection
    types = detect_variable_types(data)

    # Configuration management
    from utils import ConfigManager, MacroVariableConfig
    config_mgr = ConfigManager()
    config_mgr.load_project_config()
    macro_config = config_mgr.setup_macro_variables()

    # SAS variable resolution
    interval_vars = macro_config.resolve_variable('dm_interval_input')
    target_var = macro_config.resolve_variable('dm_dec_target')
"""

# Import main classes
from .data_io import (
    DataLoader,
    DataIOError,
    load_csv,
    load_parquet,
    load_data
)

from .partitioning import (
    PartitionManager,
    PartitionError,
    create_train_test_split,
    create_train_test_val_split
)

from .metadata import (
    MetadataHandler,
    VariableInfo,
    MetadataError,
    analyze_data,
    detect_variable_types
)

from .macro_variables import (
    MacroVariableConfig,
    MacroVariableError
)

from .config import (
    ConfigManager,
    ProjectConfig,
    ConfigurationError
)

# Define public API
__all__ = [
    # Data I/O
    'DataLoader',
    'DataIOError',
    'load_csv',
    'load_parquet', 
    'load_data',
    
    # Partitioning
    'PartitionManager',
    'PartitionError',
    'create_train_test_split',
    'create_train_test_val_split',
    
    # Metadata
    'MetadataHandler',
    'VariableInfo',
    'MetadataError',
    'analyze_data',
    'detect_variable_types',

    # Configuration and Macro Variables
    'MacroVariableConfig',
    'MacroVariableError',
    'ConfigManager',
    'ProjectConfig',
    'ConfigurationError'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'SAS to Python Migration Team'
__description__ = 'Data handling utilities for SAS to Python migration'
