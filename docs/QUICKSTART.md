# Quick Start Guide

## Setup

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install project dependencies**:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

4. **Verify setup**:
   ```bash
   python verify_setup.py
   ```

## Project Structure Overview

```
sas-to-python-examples/
├── sas_to_python/          # Main package
├── examples/               # Translated examples by category
│   ├── data_manipulation/
│   ├── visualization/
│   ├── modeling/
│   ├── preprocessing/
│   ├── clustering/
│   └── metadata_operations/
├── utils/                  # Shared utilities
│   ├── data_utils.py      # Data loading and validation
│   └── reporting_utils.py  # Reporting functions
└── tests/                  # Unit tests
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=sas_to_python

# Run specific test file
poetry run pytest tests/test_utils.py
```

## Code Quality

```bash
# Format code
poetry run black sas_to_python/ examples/ tests/

# Lint code
poetry run flake8 sas_to_python/ examples/ tests/

# Type checking
poetry run mypy sas_to_python/
```

## Adding New Examples

1. Choose the appropriate category directory under `examples/`
2. Create a new Python module with the example implementation
3. Add corresponding tests in `tests/`
4. Update the category README.md with the new example
5. Ensure all dependencies are listed in `pyproject.toml`

## Example Template

```python
"""
Example: [SAS Example Name]

Python translation of SAS [procedure/technique].
Equivalent to SAS code in: sas_code_node/[example_name]/
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.data_utils import validate_data
from utils.reporting_utils import create_summary_report


def main_function(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Main function implementing the SAS equivalent.
    
    Args:
        data: Input DataFrame
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with results and metadata
    """
    # Validate input data
    validation = validate_data(data, required_columns=['col1', 'col2'])
    if not validation['is_valid']:
        raise ValueError(f"Invalid data: {validation['errors']}")
    
    # Implementation logic here
    result_data = data.copy()
    
    # Return results with metadata
    return {
        'data': result_data,
        'metadata': validation['info'],
        'summary': create_summary_report(result_data, include_plots=False)
    }


if __name__ == '__main__':
    # Example usage
    from utils.data_utils import load_sample_data
    
    # Load sample data
    sample_data = load_sample_data('sample.csv')
    
    # Run the function
    results = main_function(sample_data)
    
    print(f"Processed {results['data'].shape[0]} rows")
    print(f"Output columns: {list(results['data'].columns)}")
```
