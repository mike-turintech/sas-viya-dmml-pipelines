#!/usr/bin/env python3
"""
Example usage of the Macro Variable Configuration System.

This script demonstrates how to use the new configuration system
for SAS to Python translation, including macro variable resolution,
configuration management, and variable list generation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    ConfigManager, MacroVariableConfig, MetadataHandler,
    DataLoader, analyze_data
)
import pandas as pd


def example_basic_configuration():
    """Example of basic configuration setup."""
    print("=== Basic Configuration Example ===")
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Load or create default project configuration
    try:
        config_manager.load_project_config()
        print("✓ Loaded existing project configuration")
    except Exception:
        print("✓ Using default project configuration")
    
    # Setup macro variables
    macro_config = config_manager.setup_macro_variables()
    
    # Display current variables
    summary = macro_config.get_variable_summary()
    print(f"✓ Loaded {len(summary['core_variables'])} core variables")
    print(f"✓ Available resolvers: {summary['resolvers_available']}")
    
    return config_manager, macro_config


def example_variable_resolution():
    """Example of variable resolution with sample data."""
    print("\n=== Variable Resolution Example ===")
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'customer_id': range(100),
        'age': [25 + i % 50 for i in range(100)],
        'income': [30000 + i * 500 for i in range(100)],
        'category': ['A', 'B', 'C'] * 33 + ['A'],
        'is_premium': [True, False] * 50,
        'target': [0, 1] * 50
    })
    
    # Analyze data to generate metadata
    metadata_handler = analyze_data(
        sample_data, 
        target_column='target',
        id_columns=['customer_id']
    )
    
    # Create macro config with metadata
    macro_config = MacroVariableConfig(metadata_handler)
    
    # Set data path
    macro_config.set_variable('dm_data', 'sample_data.csv')
    
    # Resolve variable lists
    try:
        interval_vars = macro_config.resolve_variable('dm_interval_input')
        class_vars = macro_config.resolve_variable('dm_class_input')
        target_var = macro_config.resolve_variable('dm_dec_target')
        
        print(f"✓ Interval variables: {interval_vars}")
        print(f"✓ Class variables: {class_vars}")  
        print(f"✓ Target variable: {target_var}")
        
    except Exception as e:
        print(f"✗ Error resolving variables: {e}")
    
    return macro_config


def example_sas_substitution():
    """Example of SAS code substitution."""
    print("\n=== SAS Code Substitution Example ===")
    
    # Setup configuration
    _, macro_config = example_basic_configuration()
    
    # Sample SAS code with macro variables
    sas_code = """
    proc logselect data=&dm_data;
        class %dm_class_input;
        model %dm_dec_target = %dm_interval_input %dm_class_input / link=logit;
        &dm_partition_statement;
        output out=&dm_lib..scored_data;
    run;
    """
    
    # Set some sample variables for substitution
    macro_config.set_variable('dm_data', 'training_data')
    
    # Perform substitution
    try:
        python_code = macro_config.substitute_variables(sas_code)
        print("✓ SAS code substitution successful:")
        print("Original SAS code:")
        print(sas_code)
        print("\nAfter variable substitution:")
        print(python_code)
        
    except Exception as e:
        print(f"✗ Error in substitution: {e}")


def example_configuration_validation():
    """Example of configuration validation."""
    print("\n=== Configuration Validation Example ===")
    
    config_manager = ConfigManager()
    macro_config = config_manager.setup_macro_variables()
    
    # Validate configuration
    errors = config_manager.validate_configuration()
    
    print("Validation results:")
    for component, error_list in errors.items():
        if error_list:
            print(f"✗ {component} errors:")
            for error in error_list:
                print(f"  - {error}")
        else:
            print(f"✓ {component}: OK")
    
    # Set required variable and re-validate
    print("\nSetting required dm_data variable...")
    macro_config.set_variable('dm_data', 'sample_data.csv')
    
    errors = config_manager.validate_configuration()
    remaining_errors = sum(len(err_list) for err_list in errors.values())
    print(f"Remaining validation errors: {remaining_errors}")


def example_yaml_configuration():
    """Example of YAML configuration loading/saving."""
    print("\n=== YAML Configuration Example ===")
    
    try:
        # Create macro config
        macro_config = MacroVariableConfig()
        
        # Load from default YAML
        default_config_path = Path(__file__).parent / "default_config.yaml"
        if default_config_path.exists():
            macro_config.load_from_yaml(default_config_path)
            print("✓ Loaded configuration from YAML")
            
            # Display loaded variables
            summary = macro_config.get_variable_summary()
            print(f"✓ Loaded {len(summary['core_variables'])} variables")
        else:
            print("⚠ Default YAML config not found, using defaults")
        
        # Save current config
        output_path = Path(__file__).parent / "current_config.yaml"
        macro_config.save_to_yaml(output_path)
        print(f"✓ Saved current configuration to {output_path}")
        
    except Exception as e:
        print(f"✗ YAML configuration error: {e}")


def main():
    """Run all examples."""
    print("Macro Variable Configuration System Examples")
    print("=" * 50)
    
    try:
        example_basic_configuration()
        example_variable_resolution()
        example_sas_substitution()
        example_configuration_validation()
        example_yaml_configuration()
        
        print("\n" + "=" * 50)
        print("✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
