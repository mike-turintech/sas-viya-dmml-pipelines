"""
Configuration management utilities for the ML pipeline automation system.

This module provides utilities for loading, managing, and validating
configuration files and settings for the SAS to Python translation system.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not available. Configuration loading will be limited.")

from .macro_variables import MacroVariableConfig
from .metadata import MetadataHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration operations."""
    pass


@dataclass
class ProjectConfig:
    """Configuration settings for a machine learning project."""
    
    # Project identification
    project_name: str = "ml_pipeline_project"
    project_id: Optional[str] = None
    version: str = "1.0.0"
    
    # Data paths
    data_directory: str = "./data"
    output_directory: str = "./output"
    metadata_file: Optional[str] = None
    
    # Processing settings
    cache_enabled: bool = True
    cache_directory: str = "./cache"
    cache_ttl: int = 3600
    
    # Modeling settings
    target_column: Optional[str] = None
    id_columns: Optional[List[str]] = None
    reject_columns: Optional[List[str]] = None
    
    # Partitioning settings
    train_ratio: float = 0.7
    test_ratio: float = 0.2
    validation_ratio: float = 0.1
    random_seed: int = 42
    
    # Variable naming constraints
    max_variable_name_length: int = 32
    
    def validate(self) -> List[str]:
        """Validate configuration settings."""
        errors = []
        
        # Check ratio sum
        total_ratio = self.train_ratio + self.test_ratio + self.validation_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            errors.append(f"Partition ratios must sum to 1.0, got {total_ratio}")
        
        # Check positive ratios
        if any(ratio <= 0 for ratio in [self.train_ratio, self.test_ratio, self.validation_ratio]):
            errors.append("All partition ratios must be positive")
        
        # Check data directory
        if not Path(self.data_directory).exists():
            errors.append(f"Data directory does not exist: {self.data_directory}")
        
        # Check metadata file if specified
        if self.metadata_file and not Path(self.metadata_file).exists():
            errors.append(f"Metadata file does not exist: {self.metadata_file}")
        
        return errors


class ConfigManager:
    """
    Central configuration manager for the ML pipeline system.
    
    Handles loading and managing various configuration sources including
    project settings, macro variables, and runtime parameters.
    """
    
    def __init__(self, config_directory: str = "./config"):
        """
        Initialize ConfigManager.
        
        Args:
            config_directory: Directory containing configuration files
        """
        self.config_directory = Path(config_directory)
        self.project_config = ProjectConfig()
        self.macro_config = None
        self.metadata_handler = None
        
        # Ensure config directory exists
        self.config_directory.mkdir(parents=True, exist_ok=True)
    
    def load_project_config(self, config_file: str = "project_config.yaml") -> None:
        """Load project configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ConfigurationError("PyYAML not available. Cannot load YAML configuration.")
        
        config_path = self.config_directory / config_file
        
        if not config_path.exists():
            logger.info(f"Project config file not found: {config_path}. Using defaults.")
            return
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update project config with loaded data
            for key, value in config_data.items():
                if hasattr(self.project_config, key):
                    setattr(self.project_config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            logger.info(f"Loaded project configuration from {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error loading project configuration: {e}")
    
    def save_project_config(self, config_file: str = "project_config.yaml") -> None:
        """Save current project configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ConfigurationError("PyYAML not available. Cannot save YAML configuration.")
        
        config_path = self.config_directory / config_file
        
        # Convert dataclass to dict
        config_data = {
            'project_name': self.project_config.project_name,
            'project_id': self.project_config.project_id,
            'version': self.project_config.version,
            'data_directory': self.project_config.data_directory,
            'output_directory': self.project_config.output_directory,
            'metadata_file': self.project_config.metadata_file,
            'cache_enabled': self.project_config.cache_enabled,
            'cache_directory': self.project_config.cache_directory,
            'cache_ttl': self.project_config.cache_ttl,
            'target_column': self.project_config.target_column,
            'id_columns': self.project_config.id_columns,
            'reject_columns': self.project_config.reject_columns,
            'train_ratio': self.project_config.train_ratio,
            'test_ratio': self.project_config.test_ratio,
            'validation_ratio': self.project_config.validation_ratio,
            'random_seed': self.project_config.random_seed,
            'max_variable_name_length': self.project_config.max_variable_name_length
        }
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Project configuration saved to {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error saving project configuration: {e}")
    
    def setup_macro_variables(self, config_file: str = "default_config.yaml") -> MacroVariableConfig:
        """Set up macro variable configuration."""
        # Initialize metadata handler if not exists
        if self.metadata_handler is None:
            self.metadata_handler = MetadataHandler()
        
        # Load macro configuration
        macro_config_path = self.config_directory / config_file
        
        if macro_config_path.exists():
            self.macro_config = MacroVariableConfig.from_yaml(
                macro_config_path, 
                self.metadata_handler
            )
        else:
            logger.info(f"Macro config file not found: {macro_config_path}. Using defaults.")
            self.macro_config = MacroVariableConfig(self.metadata_handler)
        
        # Sync with project config
        self._sync_configurations()
        
        return self.macro_config
    
    def _sync_configurations(self) -> None:
        """Synchronize macro variables with project configuration."""
        if self.macro_config is None:
            return
        
        # Update macro variables based on project config
        self.macro_config.set_variable('dm_lib', self.project_config.output_directory)
        
        if self.project_config.metadata_file:
            self.macro_config.set_variable('dm_metadata', self.project_config.metadata_file)
        
        # Update file paths
        output_dir = self.project_config.output_directory
        self.macro_config.set_variable('dm_file_scorecode', f"{output_dir}/scorecode.py")
        self.macro_config.set_variable('dm_file_deltacode', f"{output_dir}/deltacode.py")
        
        # Update naming constraints
        self.macro_config.set_variable('dm_maxNameLen', self.project_config.max_variable_name_length)
        
        # Set project identifiers
        if self.project_config.project_id:
            self.macro_config.set_variable('dm_projectid', self.project_config.project_id)
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate all configurations.
        
        Returns:
            Dictionary with validation errors for each component
        """
        errors = {
            'project_config': self.project_config.validate(),
            'macro_config': []
        }
        
        if self.macro_config:
            errors['macro_config'] = self.macro_config.validate_configuration()
        
        return errors
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations."""
        summary = {
            'project_config': {
                'project_name': self.project_config.project_name,
                'data_directory': self.project_config.data_directory,
                'output_directory': self.project_config.output_directory,
                'target_column': self.project_config.target_column,
                'cache_enabled': self.project_config.cache_enabled
            }
        }
        
        if self.macro_config:
            summary['macro_config'] = self.macro_config.get_variable_summary()
        
        if self.metadata_handler:
            summary['metadata'] = {
                'variables_count': len(self.metadata_handler.variables),
                'variable_types': {}
            }
            
            # Count variables by type
            type_counts = {}
            for var_info in self.metadata_handler.variables.values():
                type_counts[var_info.dtype] = type_counts.get(var_info.dtype, 0) + 1
            summary['metadata']['variable_types'] = type_counts
        
        return summary
    
    def create_default_configs(self) -> None:
        """Create default configuration files."""
        logger.info(f"Creating default configuration files in {self.config_directory}")
        
        # Create project config
        self.save_project_config()
        
        # Create macro config if it doesn't exist
        macro_config_path = self.config_directory / "default_config.yaml"
        if not macro_config_path.exists():
            if self.macro_config is None:
                self.macro_config = MacroVariableConfig()
            self.macro_config.save_to_yaml(macro_config_path)
        
        logger.info("Default configuration files created")
