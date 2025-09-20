# SAS to Python Examples

Python translations of SAS Code node examples for machine learning pipelines. This project provides equivalent Python implementations of various SAS data manipulation, visualization, and modeling techniques using modern Python libraries.

## Overview

SAS Viya provides a comprehensive, collaborative visual interface for accomplishing all steps related to the analytical life cycle. This project translates **SAS Code node examples from Model Studio** into Python equivalents, enabling data scientists to leverage the same analytical approaches using open-source Python libraries.

## Project Structure

```
sas-to-python-examples/
├── sas_to_python/          # Main package
├── examples/               # Python translations organized by category
│   ├── data_manipulation/  # Data filtering, subsetting, transformations
│   ├── visualization/      # Plotting and reporting examples
│   ├── modeling/          # Predictive modeling examples
│   ├── preprocessing/     # Data preprocessing and feature engineering
│   ├── clustering/        # Clustering and segmentation
│   └── metadata_operations/ # Metadata handling utilities
├── utils/                 # Shared utility functions
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── data/                  # Sample datasets (existing)
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To get started:

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository and install dependencies**:
   ```bash
   git clone <repository-url>
   cd sas-to-python-examples
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Core Dependencies

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and utilities
- **matplotlib**: Basic plotting and visualization
- **seaborn**: Statistical data visualization
- **scipy**: Scientific computing
- **statsmodels**: Statistical modeling
- **numpy**: Numerical computing

## Development Dependencies

- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Type checking
- **jupyter**: Notebook environment

## SAS Examples Translated

This project covers 15+ SAS Code node examples organized by complexity and functionality:

### Data Manipulation
- **subset_data**: Data filtering and subsetting
- **reverse_filter**: Inverting outlier filters
- **access_data_from_predecessor**: Data pipeline integration

### Visualization & Reporting
- **plot_samples**: Creating plots and reports
- **plot_pdf_interval_var**: PDF and interval variable plotting

### Preprocessing & Feature Engineering
- **log_transform_for_skewed_inputs**: Log transformations for skewed data
- **class_level_indicators**: One-hot encoding for categorical variables
- **high_proportion_levels**: Handling high-cardinality categorical variables

### Modeling & Assessment
- **proc_samples**: Statistical procedures and model building
- **model_assessment_report**: Model evaluation and reporting
- **bin_model_plot_assessments**: Model assessment visualizations
- **lin_reg_by_segment**: Segmented regression modeling

### Clustering & Segmentation
- **cluster_profiling**: Cluster analysis and profiling

### Advanced Examples
- **economic_capital_modeling**: Econometric modeling workflows
- **update_metadata**: Metadata manipulation

## Usage

Each example is self-contained and includes:
- Python implementation equivalent to the SAS code
- Sample data or data generation
- Comprehensive documentation
- Unit tests

Example usage:
```python
from sas_to_python.examples.data_manipulation import subset_data
from sas_to_python.utils import load_sample_data

# Load sample data
data = load_sample_data('sample_dataset.csv')

# Apply data subsetting (equivalent to SAS filtering)
filtered_data = subset_data.filter_data(data, conditions={'age': '>= 18'})
```

## Testing

Run the test suite:
```bash
poetry run pytest
```

Run tests with coverage:
```bash
poetry run pytest --cov=sas_to_python
```

## Code Quality

Format code with Black:
```bash
poetry run black sas_to_python/ examples/ tests/
```

Lint with flake8:
```bash
poetry run flake8 sas_to_python/ examples/ tests/
```

Type checking with mypy:
```bash
poetry run mypy sas_to_python/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your Python translation with tests
4. Ensure code quality standards are met
5. Submit a pull request

## Additional Resources

- [SAS Viya Overview](https://www.sas.com/en_us/software/viya.html)
- [SAS Viya: Machine Learning Documentation](http://support.sas.com/documentation/prod-p/vdmml/index.html)
- [Original SAS Code Examples](./sas_code_node/README.md)

## Contributors

**Original SAS Contributors:** Wendy Czika, Christian Medins, Radhikha Myneni, Ray Wright and Brett Wujek

**Python Translation Team:** [To be updated]

## License

[License information - see LICENSE file]
