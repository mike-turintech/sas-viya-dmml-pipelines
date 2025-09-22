"""
Data partitioning utilities for train/test/validation splits.

This module provides the PartitionManager class for creating reproducible
data partitions with support for stratified sampling and SAS-compatible
partition handling.
"""

import logging
from typing import Union, Optional, Dict, List, Tuple, Any
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PartitionError(Exception):
    """Custom exception for partitioning operations."""
    pass


class PartitionManager:
    """
    Manager for creating reproducible train/test/validation data partitions.
    
    Provides SAS-compatible partitioning with support for stratified sampling,
    reproducible splits, and flexible partition configurations.
    """
    
    def __init__(self, random_state: Optional[int] = 42):
        """
        Initialize PartitionManager.
        
        Args:
            random_state: Random seed for reproducible splits. If None, splits won't be reproducible.
        """
        self.random_state = random_state
        self._partition_history = []
    
    def create_partition_column(self, data: pd.DataFrame, 
                              train_ratio: float = 0.7,
                              test_ratio: float = 0.2,
                              validation_ratio: float = 0.1,
                              stratify_column: Optional[str] = None,
                              partition_column: str = '_partition_') -> pd.DataFrame:
        """
        Create a partition column in the DataFrame (SAS-style).
        
        Args:
            data: Input DataFrame
            train_ratio: Proportion for training set
            test_ratio: Proportion for test set  
            validation_ratio: Proportion for validation set
            stratify_column: Column name to use for stratified sampling
            partition_column: Name for the partition column to create
            
        Returns:
            DataFrame with added partition column
        """
        # Validate ratios
        total_ratio = train_ratio + test_ratio + validation_ratio
        if not np.isclose(total_ratio, 1.0, rtol=1e-5):
            raise PartitionError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        if any(ratio <= 0 for ratio in [train_ratio, test_ratio, validation_ratio]):
            raise PartitionError("All ratios must be positive")
        
        # Check if stratify column exists
        if stratify_column and stratify_column not in data.columns:
            raise PartitionError(f"Stratify column '{stratify_column}' not found in data")
        
        result_data = data.copy()
        n_samples = len(data)
        
        # Calculate sample sizes
        n_train = int(n_samples * train_ratio)
        n_test = int(n_samples * test_ratio)
        n_validation = n_samples - n_train - n_test  # Remainder goes to validation
        
        logger.info(f"Creating partitions: train={n_train}, test={n_test}, validation={n_validation}")
        
        if stratify_column:
            # Stratified partitioning
            partition_labels = self._create_stratified_partitions(
                data, stratify_column, train_ratio, test_ratio, validation_ratio
            )
        else:
            # Random partitioning
            partition_labels = self._create_random_partitions(
                n_samples, train_ratio, test_ratio, validation_ratio
            )
        
        result_data[partition_column] = partition_labels
        
        # Store partition history
        partition_info = {
            'train_ratio': train_ratio,
            'test_ratio': test_ratio, 
            'validation_ratio': validation_ratio,
            'stratify_column': stratify_column,
            'partition_column': partition_column,
            'n_train': (partition_labels == 'TRAIN').sum(),
            'n_test': (partition_labels == 'TEST').sum(),
            'n_validation': (partition_labels == 'VALIDATE').sum()
        }
        self._partition_history.append(partition_info)
        
        return result_data
    
    def _create_random_partitions(self, n_samples: int, 
                                train_ratio: float,
                                test_ratio: float, 
                                validation_ratio: float) -> np.ndarray:
        """Create random partitions."""
        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Create indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Calculate split points
        n_train = int(n_samples * train_ratio)
        n_test = int(n_samples * test_ratio)
        
        # Create partition labels
        labels = np.empty(n_samples, dtype='<U8')
        labels[indices[:n_train]] = 'TRAIN'
        labels[indices[n_train:n_train + n_test]] = 'TEST'
        labels[indices[n_train + n_test:]] = 'VALIDATE'
        
        return labels
    
    def _create_stratified_partitions(self, data: pd.DataFrame,
                                    stratify_column: str,
                                    train_ratio: float,
                                    test_ratio: float,
                                    validation_ratio: float) -> np.ndarray:
        """Create stratified partitions."""
        y = data[stratify_column].values
        n_samples = len(data)
        
        # Check if stratification is meaningful
        unique_values, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        
        if min_count < 3:
            warnings.warn(
                f"Stratify column '{stratify_column}' has classes with fewer than 3 samples. "
                "Falling back to random partitioning.",
                UserWarning
            )
            return self._create_random_partitions(n_samples, train_ratio, test_ratio, validation_ratio)
        
        # First split: train vs (test + validation)
        temp_ratio = test_ratio + validation_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            np.arange(n_samples), y,
            test_size=temp_ratio,
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: test vs validation from temp
        test_ratio_adjusted = test_ratio / temp_ratio
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp,
            test_size=(1 - test_ratio_adjusted),
            stratify=y_temp,
            random_state=self.random_state
        )
        
        # Create partition labels
        labels = np.empty(n_samples, dtype='<U8')
        labels[X_train] = 'TRAIN'
        labels[X_test] = 'TEST'
        labels[X_val] = 'VALIDATE'
        
        return labels
    
    def split_data(self, data: pd.DataFrame,
                  train_ratio: float = 0.7,
                  test_ratio: float = 0.2,
                  validation_ratio: float = 0.1,
                  stratify_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, test, and validation DataFrames.
        
        Args:
            data: Input DataFrame
            train_ratio: Proportion for training set
            test_ratio: Proportion for test set
            validation_ratio: Proportion for validation set
            stratify_column: Column name to use for stratified sampling
            
        Returns:
            Tuple of (train_df, test_df, validation_df)
        """
        # Create temporary partition column
        temp_partition_col = '_temp_partition_'
        partitioned_data = self.create_partition_column(
            data, train_ratio, test_ratio, validation_ratio, 
            stratify_column, temp_partition_col
        )
        
        # Split into separate DataFrames
        train_df = partitioned_data[partitioned_data[temp_partition_col] == 'TRAIN'].copy()
        test_df = partitioned_data[partitioned_data[temp_partition_col] == 'TEST'].copy()
        validation_df = partitioned_data[partitioned_data[temp_partition_col] == 'VALIDATE'].copy()
        
        # Remove temporary partition column
        train_df.drop(columns=[temp_partition_col], inplace=True)
        test_df.drop(columns=[temp_partition_col], inplace=True)
        validation_df.drop(columns=[temp_partition_col], inplace=True)
        
        return train_df, test_df, validation_df
    
    def create_time_series_partition(self, data: pd.DataFrame,
                                   time_column: str,
                                   train_end: Union[str, pd.Timestamp],
                                   test_end: Optional[Union[str, pd.Timestamp]] = None,
                                   partition_column: str = '_partition_') -> pd.DataFrame:
        """
        Create time-series based partitions.
        
        Args:
            data: Input DataFrame
            time_column: Name of the datetime column
            train_end: End date/time for training set
            test_end: End date/time for test set (remaining becomes validation)
            partition_column: Name for the partition column to create
            
        Returns:
            DataFrame with added partition column
        """
        if time_column not in data.columns:
            raise PartitionError(f"Time column '{time_column}' not found in data")
        
        result_data = data.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(result_data[time_column]):
            result_data[time_column] = pd.to_datetime(result_data[time_column])
        
        # Convert partition points to datetime
        train_end = pd.to_datetime(train_end)
        if test_end is not None:
            test_end = pd.to_datetime(test_end)
        
        # Create partition labels
        conditions = [
            result_data[time_column] <= train_end,
        ]
        choices = ['TRAIN']
        
        if test_end is not None:
            conditions.append(result_data[time_column] <= test_end)
            choices.append('TEST')
            default = 'VALIDATE'
        else:
            default = 'TEST'
        
        partition_labels = np.select(conditions, choices, default=default)
        result_data[partition_column] = partition_labels
        
        # Log partition info
        partition_counts = pd.Series(partition_labels).value_counts()
        logger.info(f"Time-series partitions created: {dict(partition_counts)}")
        
        return result_data
    
    def validate_partitions(self, data: pd.DataFrame, 
                          partition_column: str = '_partition_',
                          target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate partition quality and balance.
        
        Args:
            data: DataFrame with partition column
            partition_column: Name of the partition column
            target_column: Target variable for additional validation
            
        Returns:
            Dictionary with validation results
        """
        if partition_column not in data.columns:
            raise PartitionError(f"Partition column '{partition_column}' not found in data")
        
        results = {}
        
        # Basic partition statistics
        partition_counts = data[partition_column].value_counts()
        total_count = len(data)
        
        results['partition_counts'] = dict(partition_counts)
        results['partition_proportions'] = {
            k: v / total_count for k, v in partition_counts.items()
        }
        
        # Check for empty partitions
        results['has_empty_partitions'] = any(count == 0 for count in partition_counts.values())
        
        # Target variable balance if provided
        if target_column and target_column in data.columns:
            target_balance = {}
            for partition in partition_counts.index:
                partition_data = data[data[partition_column] == partition]
                if len(partition_data) > 0:
                    target_dist = partition_data[target_column].value_counts(normalize=True)
                    target_balance[partition] = dict(target_dist)
            
            results['target_balance'] = target_balance
            
            # Check for significant imbalances
            results['balanced_target'] = self._check_target_balance(target_balance)
        
        return results
    
    def _check_target_balance(self, target_balance: Dict[str, Dict]) -> bool:
        """Check if target variable is reasonably balanced across partitions."""
        if len(target_balance) < 2:
            return True
        
        # Get all target classes
        all_classes = set()
        for partition_dist in target_balance.values():
            all_classes.update(partition_dist.keys())
        
        # Check balance for each class
        for target_class in all_classes:
            proportions = []
            for partition_dist in target_balance.values():
                proportions.append(partition_dist.get(target_class, 0))
            
            # Check if proportions are reasonably similar (within 20%)
            if len(proportions) > 1:
                prop_range = max(proportions) - min(proportions)
                if prop_range > 0.2:  # More than 20% difference
                    return False
        
        return True
    
    def get_partition_info(self, data: pd.DataFrame, 
                          partition_column: str = '_partition_') -> pd.DataFrame:
        """
        Get detailed information about data partitions.
        
        Args:
            data: DataFrame with partition column
            partition_column: Name of the partition column
            
        Returns:
            DataFrame with partition statistics
        """
        if partition_column not in data.columns:
            raise PartitionError(f"Partition column '{partition_column}' not found in data")
        
        # Calculate basic statistics by partition
        info_list = []
        for partition in data[partition_column].unique():
            partition_data = data[data[partition_column] == partition]
            
            info = {
                'partition': partition,
                'count': len(partition_data),
                'proportion': len(partition_data) / len(data)
            }
            
            # Add numeric column statistics
            numeric_cols = partition_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                info['mean_numeric'] = partition_data[numeric_cols].mean().mean()
                info['std_numeric'] = partition_data[numeric_cols].std().mean()
            
            # Add missing value statistics
            info['missing_values'] = partition_data.isnull().sum().sum()
            info['missing_proportion'] = info['missing_values'] / (len(partition_data) * len(partition_data.columns))
            
            info_list.append(info)
        
        return pd.DataFrame(info_list)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get partition history."""
        return self._partition_history.copy()
    
    def clear_history(self) -> None:
        """Clear partition history."""
        self._partition_history.clear()


# Convenience functions for quick partitioning
def create_train_test_split(data: pd.DataFrame, 
                           test_size: float = 0.2,
                           stratify_column: Optional[str] = None,
                           random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick train/test split function.
    
    Args:
        data: Input DataFrame
        test_size: Proportion for test set
        stratify_column: Column for stratified sampling
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    manager = PartitionManager(random_state=random_state)
    train_ratio = 1.0 - test_size
    train_df, test_df, _ = manager.split_data(
        data, train_ratio=train_ratio, test_ratio=test_size, 
        validation_ratio=0.0, stratify_column=stratify_column
    )
    return train_df, test_df


def create_train_test_val_split(data: pd.DataFrame,
                               train_ratio: float = 0.7,
                               test_ratio: float = 0.2, 
                               validation_ratio: float = 0.1,
                               stratify_column: Optional[str] = None,
                               random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Quick train/test/validation split function.
    
    Args:
        data: Input DataFrame  
        train_ratio: Proportion for training set
        test_ratio: Proportion for test set
        validation_ratio: Proportion for validation set
        stratify_column: Column for stratified sampling
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df, validation_df)
    """
    manager = PartitionManager(random_state=random_state)
    return manager.split_data(
        data, train_ratio, test_ratio, validation_ratio, stratify_column
    )
