"""
Macro variable configuration system for SAS to Python translation.

This module provides the MacroVariableConfig class to handle SAS macro variables
and their Python equivalents, with support for dynamic resolution and variable
list generation from metadata.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import re
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not available. YAML configuration support will be limited.")

from .metadata import MetadataHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacroVariableError(Exception):
    """Custom exception for macro variable operations."""
    pass


class MacroVariableConfig:
    """
    Configuration system for handling SAS macro variables in Python.
    
    Provides mapping and resolution of SAS macro variables like &dm_data, &dm_lib,
    %dm_interval_input, etc. to their Python equivalents with runtime resolution
    support.
    """
    
    def __init__(self, metadata_handler: Optional[MetadataHandler] = None):
        """
        Initialize MacroVariableConfig.
        
        Args:
            metadata_handler: MetadataHandler instance for variable resolution
        """
        self.metadata_handler = metadata_handler
        self.variables = {}  # Core macro variables
        self.resolvers = {}  # Custom resolver functions
        self.templates = {}  # Template patterns
        
        # Initialize default variables and resolvers
        self._setup_default_variables()
        self._setup_default_resolvers()
        self._setup_default_templates()
    
    def _setup_default_variables(self):
        """Set up default SAS macro variable mappings."""
        self.variables.update({
            # Data and library paths
            'dm_data': None,           # Main dataset path/name
            'dm_lib': './output',      # Output library/directory
            'dm_metadata': None,       # Metadata file path
            
            # File paths  
            'dm_file_scorecode': './output/scorecode.py',
            'dm_file_deltacode': './output/deltacode.py',
            
            # CAS/Server settings (not directly applicable but kept for compatibility)
            'dm_cassessref': 'python_session',
            'dm_ds_caslib': 'CASUSER',
            'dm_memname': None,
            'dm_casiocalib': 'CASUSER',
            
            # Variable naming
            'dm_maxNameLen': 32,
            
            # Partition settings
            'dm_partition_statement': "partition role='TRAIN'",
            
            # Project settings
            'dm_projectid': None,
            'dm_nodeid': None,
            'dm_runid': None
        })
    
    def _setup_default_resolvers(self):
        """Set up default resolver functions for variable list macros."""
        self.resolvers.update({
            'dm_interval_input': self._resolve_interval_variables,
            'dm_class_input': self._resolve_class_variables,
            'dm_binary_input': self._resolve_binary_variables,
            'dm_nominal_input': self._resolve_nominal_variables,
            'dm_ordinal_input': self._resolve_ordinal_variables,
            'dm_dec_target': self._resolve_target_variable,
            'dm_all_input': self._resolve_all_input_variables,
            'dm_input': self._resolve_all_input_variables  # Alias
        })
    
    def _setup_default_templates(self):
        """Set up default template patterns."""
        self.templates.update({
            'partition_filter_train': "partition_column == 'TRAIN'",
            'partition_filter_test': "partition_column == 'TEST'",
            'partition_filter_validate': "partition_column == 'VALIDATE'",
            'score_code_header': "# Generated scoring code for model",
            'delta_code_header': "# Generated delta transformations"
        })
    
    def set_variable(self, name: str, value: Any):
        """Set a macro variable value."""
        self.variables[name] = value
        logger.debug(f"Set macro variable {name} = {value}")
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a macro variable value."""
        if name in self.variables:
            return self.variables[name]
        elif default is not None:
            return default
        else:
            raise MacroVariableError(f"Macro variable '{name}' not found and no default provided")
    
    def resolve_variable(self, name: str, **kwargs) -> Any:
        """
        Resolve a macro variable, using custom resolvers if available.
        
        Args:
            name: Variable name to resolve
            **kwargs: Additional parameters for resolvers
            
        Returns:
            Resolved variable value
        """
        # Check if it's a resolver-based variable (like %dm_interval_input)
        if name in self.resolvers:
            if self.metadata_handler is None:
                raise MacroVariableError(f"Cannot resolve '{name}': no metadata handler available")
            return self.resolvers[name](**kwargs)
        
        # Otherwise get regular variable
        return self.get_variable(name)
    
    def _resolve_interval_variables(self, role: str = 'INPUT', **kwargs) -> List[str]:
        """Resolve interval (numeric) input variables."""
        if self.metadata_handler is None:
            return []
        
        variables = []
        for var_name, var_info in self.metadata_handler.variables.items():
            if (var_info.role == role and 
                var_info.dtype == 'NUMERIC'):
                variables.append(var_name)
        
        return variables
    
    def _resolve_class_variables(self, role: str = 'INPUT', **kwargs) -> List[str]:
        """Resolve class (categorical) input variables."""
        if self.metadata_handler is None:
            return []
        
        variables = []
        for var_name, var_info in self.metadata_handler.variables.items():
            if (var_info.role == role and 
                var_info.dtype == 'CATEGORICAL'):
                variables.append(var_name)
        
        return variables
    
    def _resolve_binary_variables(self, role: str = 'INPUT', **kwargs) -> List[str]:
        """Resolve binary categorical variables."""
        if self.metadata_handler is None:
            return []
        
        variables = []
        for var_name, var_info in self.metadata_handler.variables.items():
            if (var_info.role == role and 
                var_info.dtype == 'CATEGORICAL' and
                var_info.unique_count == 2):
                variables.append(var_name)
        
        return variables
    
    def _resolve_nominal_variables(self, role: str = 'INPUT', **kwargs) -> List[str]:
        """Resolve nominal categorical variables (non-binary, non-ordinal)."""
        if self.metadata_handler is None:
            return []
        
        variables = []
        for var_name, var_info in self.metadata_handler.variables.items():
            if (var_info.role == role and 
                var_info.dtype == 'CATEGORICAL' and
                var_info.unique_count > 2):  # Simplified - in practice might need ordinal detection
                variables.append(var_name)
        
        return variables
    
    def _resolve_ordinal_variables(self, role: str = 'INPUT', **kwargs) -> List[str]:
        """Resolve ordinal categorical variables."""
        # This would need additional metadata to distinguish ordinal from nominal
        # For now, return empty list as it requires domain knowledge
        return []
    
    def _resolve_target_variable(self, **kwargs) -> str:
        """Resolve target variable."""
        if self.metadata_handler is None:
            return ""
        
        for var_name, var_info in self.metadata_handler.variables.items():
            if var_info.role == 'TARGET':
                return var_name
        
        return ""
    
    def _resolve_all_input_variables(self, **kwargs) -> List[str]:
        """Resolve all input variables."""
        if self.metadata_handler is None:
            return []
        
        variables = []
        for var_name, var_info in self.metadata_handler.variables.items():
            if var_info.role == 'INPUT':
                variables.append(var_name)
        
        return variables
    
    def substitute_variables(self, text: str, **kwargs) -> str:
        """
        Substitute macro variables in text using SAS-like patterns.
        
        Args:
            text: Text containing macro variables like &dm_data or %dm_interval_input
            **kwargs: Additional parameters for resolvers
            
        Returns:
            Text with variables substituted
        """
        result = text
        
        # Handle &variable patterns (simple substitution)
        var_pattern = r'&(\w+)\.?'
        matches = re.findall(var_pattern, result)
        
        for var_name in matches:
            try:
                value = self.get_variable(var_name)
                if value is not None:
                    # Replace with and without trailing dot
                    result = re.sub(f'&{var_name}\.?', str(value), result)
            except MacroVariableError:
                logger.warning(f"Could not resolve macro variable: {var_name}")
        
        # Handle %variable patterns (list generators)
        list_pattern = r'%(\w+)'
        matches = re.findall(list_pattern, result)
        
        for var_name in matches:
            try:
                value_list = self.resolve_variable(var_name, **kwargs)
                if isinstance(value_list, list):
                    value_str = ' '.join(str(v) for v in value_list)
                    result = re.sub(f'%{var_name}', value_str, result)
                elif value_list:
                    result = re.sub(f'%{var_name}', str(value_list), result)
            except MacroVariableError:
                logger.warning(f"Could not resolve macro variable: {var_name}")
        
        return result
    
    def create_partition_statement(self, partition_column: str = '_partition_', 
                                 partition_value: str = 'TRAIN') -> str:
        """Create a partition filter statement for Python."""
        return f"data[data['{partition_column}'] == '{partition_value}']"
    
    def get_data_path(self, resolve_relative: bool = True) -> str:
        """Get resolved data path."""
        data_path = self.get_variable('dm_data')
        if data_path is None:
            raise MacroVariableError("dm_data variable not set")
        
        if resolve_relative and not Path(data_path).is_absolute():
            # Try to resolve relative to common data directories
            base_paths = ['.', 'data', '../data', '../../data']
            for base in base_paths:
                full_path = Path(base) / data_path
                if full_path.exists():
                    return str(full_path.resolve())
        
        return data_path
    
    def get_output_path(self, filename: str = None) -> str:
        """Get resolved output path."""
        lib_path = self.get_variable('dm_lib', './output')
        
        # Ensure output directory exists
        Path(lib_path).mkdir(parents=True, exist_ok=True)
        
        if filename:
            return str(Path(lib_path) / filename)
        else:
            return lib_path
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required variables
        required_vars = ['dm_data']
        for var in required_vars:
            if self.get_variable(var) is None:
                errors.append(f"Required macro variable '{var}' is not set")
        
        # Check data file exists if dm_data is set
        data_path = self.get_variable('dm_data')
        if data_path and not Path(data_path).exists():
            errors.append(f"Data file does not exist: {data_path}")
        
        # Check metadata handler if variable resolvers will be used
        if self.metadata_handler is None:
            errors.append("No metadata handler available for variable resolution")
        
        return errors
    
    def get_variable_summary(self) -> Dict[str, Any]:
        """Get summary of all configured variables."""
        summary = {
            'core_variables': dict(self.variables),
            'resolvers_available': list(self.resolvers.keys()),
            'templates_available': list(self.templates.keys()),
            'metadata_handler': self.metadata_handler is not None
        }
        
        # Try to resolve some common variable lists if metadata is available
        if self.metadata_handler:
            try:
                summary['resolved_variables'] = {
                    'interval_inputs': self.resolve_variable('dm_interval_input'),
                    'class_inputs': self.resolve_variable('dm_class_input'),
                    'target': self.resolve_variable('dm_dec_target')
                }
            except MacroVariableError:
                summary['resolved_variables'] = "Error resolving variables"
        
        return summary
    
    def load_from_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if not YAML_AVAILABLE:
            raise MacroVariableError("PyYAML not available. Cannot load YAML configuration.")
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise MacroVariableError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update variables
            if 'variables' in config_data:
                self.variables.update(config_data['variables'])
                logger.info(f"Loaded {len(config_data['variables'])} variables from {config_path}")
            
            # Update templates
            if 'templates' in config_data:
                self.templates.update(config_data['templates'])
                logger.info(f"Loaded {len(config_data['templates'])} templates from {config_path}")
            
        except yaml.YAMLError as e:
            raise MacroVariableError(f"Error parsing YAML file {config_path}: {e}")
        except Exception as e:
            raise MacroVariableError(f"Error loading configuration from {config_path}: {e}")
    
    def save_to_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path where to save YAML configuration
        """
        if not YAML_AVAILABLE:
            raise MacroVariableError("PyYAML not available. Cannot save YAML configuration.")
        
        config_data = {
            'variables': dict(self.variables),
            'templates': dict(self.templates),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            raise MacroVariableError(f"Error saving configuration to {config_path}: {e}")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path], 
                  metadata_handler: Optional[MetadataHandler] = None) -> 'MacroVariableConfig':
        """
        Create MacroVariableConfig instance from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            metadata_handler: Optional MetadataHandler instance
            
        Returns:
            Configured MacroVariableConfig instance
        """
        config = cls(metadata_handler=metadata_handler)
        config.load_from_yaml(config_path)
        return config
