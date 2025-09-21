# Configuration System for SAS to Python Translation

This directory contains the configuration system for translating SAS macro variables and settings to their Python equivalents. The system provides dynamic resolution of data paths, variable lists, and processing settings.

## Files

- **`default_config.yaml`** - Default macro variable mappings and templates
- **`project_config.yaml`** - Project-level configuration template  
- **`example_usage.py`** - Example usage and demonstration script
- **`README.md`** - This documentation file

## Key Components

### 1. MacroVariableConfig Class

Handles SAS macro variables and their Python equivalents:

```python
from utils import MacroVariableConfig, MetadataHandler

# Initialize with metadata handler for variable resolution
metadata_handler = MetadataHandler()
macro_config = MacroVariableConfig(metadata_handler)

# Set core variables
macro_config.set_variable('dm_data', 'data/training_data.csv')
macro_config.set_variable('dm_lib', './output')

# Resolve variable lists from metadata
interval_vars = macro_config.resolve_variable('dm_interval_input')
class_vars = macro_config.resolve_variable('dm_class_input')
target_var = macro_config.resolve_variable('dm_dec_target')
```

### 2. ConfigManager Class

Central configuration management:

```python
from utils import ConfigManager

# Initialize and load configurations
config_manager = ConfigManager()
config_manager.load_project_config()
macro_config = config_manager.setup_macro_variables()

# Validate all configurations
errors = config_manager.validate_configuration()
```

## Supported SAS Macro Variables

### Core Variables (&variable_name)

| SAS Variable | Python Equivalent | Description |
|--------------|------------------|-------------|
| `&dm_data` | Data file path | Main dataset path/name |
| `&dm_lib` | Output directory | Directory for results and generated files |
| `&dm_metadata` | Metadata file path | Variable metadata and properties |
| `&dm_file_scorecode` | Scoring code path | Generated Python scoring code |
| `&dm_file_deltacode` | Transform code path | Generated Python transformations |
| `&dm_partition_statement` | Partition filter | Data partitioning logic |
| `&dm_maxNameLen` | Name length limit | Maximum variable name length |

### Variable List Macros (%macro_name)

| SAS Macro | Python Resolution | Description |
|-----------|------------------|-------------|
| `%dm_interval_input` | List of numeric variables | Continuous input variables |
| `%dm_class_input` | List of categorical variables | Class/categorical input variables |
| `%dm_binary_input` | List of binary variables | Binary categorical variables |
| `%dm_nominal_input` | List of nominal variables | Nominal categorical variables |
| `%dm_dec_target` | Target variable name | Decision/target variable |
| `%dm_all_input` | List of all input variables | All input variables |

## Variable Resolution

Variables are resolved automatically from metadata when available:

```python
# Requires MetadataHandler with analyzed data
metadata_handler = analyze_data(data, target_column='target')
macro_config = MacroVariableConfig(metadata_handler)

# Automatic resolution based on variable types and roles
interval_vars = macro_config.resolve_variable('dm_interval_input')
# Returns: ['age', 'income', 'score'] (numeric variables with role='INPUT')

class_vars = macro_config.resolve_variable('dm_class_input')  
# Returns: ['category', 'region'] (categorical variables with role='INPUT')

target_var = macro_config.resolve_variable('dm_dec_target')
# Returns: 'target' (variable with role='TARGET')
```

## SAS Code Substitution

Transform SAS code by substituting macro variables:

```python
sas_code = """
proc logselect data=&dm_data;
  class %dm_class_input;
  model %dm_dec_target = %dm_interval_input %dm_class_input / link=logit;
  &dm_partition_statement;
run;
"""

# Substitute variables
python_equivalent = macro_config.substitute_variables(sas_code)
```

## Configuration Files

### YAML Configuration

Load and save configurations from YAML files:

```python
# Load from YAML
macro_config = MacroVariableConfig.from_yaml('config/default_config.yaml')

# Save current config
macro_config.save_to_yaml('config/my_config.yaml')
```

### Project Configuration

Set project-wide settings in `project_config.yaml`:

```yaml
project_name: "customer_churn_prediction"
target_column: "churn_flag"
data_directory: "./data/customer_data"
train_ratio: 0.7
test_ratio: 0.2
validation_ratio: 0.1
```

## Templates and Patterns

The system includes templates for common SAS patterns:

```python
# Partition filters
train_filter = macro_config.templates['partition_filter_train']
# Returns: "data[data['_partition_'] == 'TRAIN']"

# Code generation headers
score_header = macro_config.templates['score_code_header']
# Returns: "# Generated Python Scoring Code\n..."
```

## Validation

Validate configurations before use:

```python
# Validate macro variables
errors = macro_config.validate_configuration()

# Validate project configuration
config_manager = ConfigManager()
all_errors = config_manager.validate_configuration()

for component, error_list in all_errors.items():
    if error_list:
        print(f"{component} errors: {error_list}")
```

## Usage Examples

### Basic Setup

```python
from utils import ConfigManager

# Initialize system
config_manager = ConfigManager()
config_manager.load_project_config()
macro_config = config_manager.setup_macro_variables()

# Set data file
macro_config.set_variable('dm_data', 'data/my_data.csv')
```

### With Data Analysis

```python
from utils import ConfigManager, analyze_data
import pandas as pd

# Load and analyze data
data = pd.read_csv('data/training_data.csv')
metadata_handler = analyze_data(data, target_column='churn')

# Setup configuration with metadata
config_manager = ConfigManager()
macro_config = config_manager.setup_macro_variables()
macro_config.metadata_handler = metadata_handler

# Now variable resolution works automatically
features = macro_config.resolve_variable('dm_interval_input')
target = macro_config.resolve_variable('dm_dec_target')
```

### Complete Example

See `example_usage.py` for a complete working example that demonstrates all features.

## Error Handling

The system provides comprehensive error handling:

```python
from utils import MacroVariableError, ConfigurationError

try:
    value = macro_config.get_variable('nonexistent_var')
except MacroVariableError as e:
    print(f"Variable not found: {e}")

try:
    macro_config.load_from_yaml('missing_file.yaml')
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Integration with Other Utilities

The configuration system integrates seamlessly with other utilities:

```python
from utils import DataLoader, PartitionManager, MetadataHandler

# Use resolved paths with DataLoader
data_path = macro_config.get_data_path()
loader = DataLoader()
data = loader.load_data(data_path)

# Use with PartitionManager
manager = PartitionManager(random_state=macro_config.get_variable('random_seed'))
train_df, test_df, val_df = manager.split_data(data)

# Analyze with MetadataHandler
handler = MetadataHandler()
handler.analyze_dataframe(data, target_column=macro_config.resolve_variable('dm_dec_target'))
```

## Next Steps

This configuration system provides the foundation for SAS to Python translation. Next steps include:

1. **Code Generation**: Use templates to generate Python code from SAS procedures
2. **Procedure Translation**: Implement specific SAS procedure equivalents
3. **Workflow Automation**: Create end-to-end pipeline automation
4. **Integration Testing**: Test with real SAS projects and data
