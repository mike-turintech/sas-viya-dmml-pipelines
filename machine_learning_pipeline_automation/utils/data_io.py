"""
Data I/O utilities for handling CSV and Parquet files with unified interface.

This module provides the DataLoader class for consistent data loading across
different file formats, with caching and path resolution capabilities.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import hashlib
from datetime import datetime, timedelta

import pandas as pd

# Optional imports for enhanced functionality
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logging.warning("PyArrow not available. Parquet support will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIOError(Exception):
    """Custom exception for data I/O operations."""
    pass


class DataLoader:
    """
    Unified interface for loading CSV and Parquet files with caching capabilities.
    
    Provides consistent data loading methods, automatic type detection,
    and performance optimization through caching mechanisms.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, cache_ttl: int = 3600):
        """
        Initialize DataLoader with optional caching.
        
        Args:
            cache_dir: Directory for caching loaded data. If None, caching is disabled.
            cache_ttl: Cache time-to-live in seconds (default: 1 hour).
        """
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self._cache_enabled = cache_dir is not None
        
        if self._cache_enabled:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, file_path: str, **kwargs) -> str:
        """Generate cache key based on file path and parameters."""
        content = f"{file_path}_{str(kwargs)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cached file is still valid based on TTL."""
        if not os.path.exists(cache_file):
            return False
        
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return datetime.now() - cache_time < timedelta(seconds=self.cache_ttl)
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid."""
        if not self._cache_enabled:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    logger.info(f"Loading data from cache: {cache_key}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save data to cache."""
        if not self._cache_enabled:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data cached: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def resolve_file_path(self, file_path: Union[str, Path], 
                         base_paths: Optional[List[str]] = None) -> str:
        """
        Resolve file path with support for multiple base directories.
        
        Args:
            file_path: File path to resolve
            base_paths: List of base directories to search in
            
        Returns:
            Resolved absolute file path
            
        Raises:
            DataIOError: If file cannot be found
        """
        file_path = Path(file_path)
        
        # If already absolute and exists, return as-is
        if file_path.is_absolute() and file_path.exists():
            return str(file_path)
        
        # Define default base paths if none provided
        if base_paths is None:
            base_paths = [
                '.',
                'data',
                '../data',
                '../../data'
            ]
        
        # Try each base path
        for base_path in base_paths:
            full_path = Path(base_path) / file_path
            if full_path.exists():
                return str(full_path.resolve())
        
        raise DataIOError(f"File not found: {file_path}. Searched in: {base_paths}")
    
    def load_csv(self, file_path: Union[str, Path], 
                 use_cache: bool = True,
                 **kwargs) -> pd.DataFrame:
        """
        Load CSV file with pandas.
        
        Args:
            file_path: Path to CSV file
            use_cache: Whether to use caching
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        resolved_path = self.resolve_file_path(file_path)
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(resolved_path, **kwargs)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Set default CSV parameters for better SAS compatibility
        csv_defaults = {
            'encoding': 'utf-8',
            'na_values': ['', 'NA', 'NULL', '.', 'null'],
            'keep_default_na': True,
            'low_memory': False
        }
        csv_defaults.update(kwargs)
        
        try:
            logger.info(f"Loading CSV file: {resolved_path}")
            data = pd.read_csv(resolved_path, **csv_defaults)
            
            # Cache the data
            if use_cache:
                self._save_to_cache(data, cache_key)
            
            return data
            
        except Exception as e:
            raise DataIOError(f"Failed to load CSV file {resolved_path}: {e}")
    
    def load_parquet(self, file_path: Union[str, Path], 
                    use_cache: bool = True,
                    engine: str = 'auto',
                    **kwargs) -> pd.DataFrame:
        """
        Load Parquet file with pandas/pyarrow.
        
        Args:
            file_path: Path to Parquet file
            use_cache: Whether to use caching
            engine: Parquet engine to use ('auto', 'pyarrow', 'fastparquet')
            **kwargs: Additional arguments passed to pd.read_parquet
            
        Returns:
            DataFrame with loaded data
        """
        resolved_path = self.resolve_file_path(file_path)
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(resolved_path, engine=engine, **kwargs)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Determine engine
        if engine == 'auto':
            engine = 'pyarrow' if PARQUET_AVAILABLE else 'fastparquet'
        
        try:
            logger.info(f"Loading Parquet file: {resolved_path}")
            data = pd.read_parquet(resolved_path, engine=engine, **kwargs)
            
            # Cache the data
            if use_cache:
                self._save_to_cache(data, cache_key)
            
            return data
            
        except Exception as e:
            raise DataIOError(f"Failed to load Parquet file {resolved_path}: {e}")
    
    def load_data(self, file_path: Union[str, Path], 
                  file_format: Optional[str] = None,
                  use_cache: bool = True,
                  **kwargs) -> pd.DataFrame:
        """
        Unified interface for loading data files (CSV or Parquet).
        
        Args:
            file_path: Path to data file
            file_format: File format ('csv' or 'parquet'). Auto-detected if None.
            use_cache: Whether to use caching
            **kwargs: Additional arguments passed to specific loader
            
        Returns:
            DataFrame with loaded data
        """
        resolved_path = self.resolve_file_path(file_path)
        
        # Auto-detect format if not specified
        if file_format is None:
            file_ext = Path(resolved_path).suffix.lower()
            if file_ext == '.csv':
                file_format = 'csv'
            elif file_ext in ['.parquet', '.pq']:
                file_format = 'parquet'
            else:
                raise DataIOError(f"Cannot auto-detect format for file: {resolved_path}")
        
        # Load data using appropriate method
        if file_format.lower() == 'csv':
            return self.load_csv(resolved_path, use_cache=use_cache, **kwargs)
        elif file_format.lower() == 'parquet':
            return self.load_parquet(resolved_path, use_cache=use_cache, **kwargs)
        else:
            raise DataIOError(f"Unsupported file format: {file_format}")
    
    def save_csv(self, data: pd.DataFrame, file_path: Union[str, Path], 
                 **kwargs) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            file_path: Output file path
            **kwargs: Additional arguments passed to pd.DataFrame.to_csv
        """
        # Set default CSV parameters
        csv_defaults = {
            'index': False,
            'encoding': 'utf-8',
            'na_rep': ''
        }
        csv_defaults.update(kwargs)
        
        # Ensure output directory exists
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Saving CSV file: {output_path}")
            data.to_csv(output_path, **csv_defaults)
        except Exception as e:
            raise DataIOError(f"Failed to save CSV file {output_path}: {e}")
    
    def save_parquet(self, data: pd.DataFrame, file_path: Union[str, Path],
                    engine: str = 'auto', **kwargs) -> None:
        """
        Save DataFrame to Parquet file.
        
        Args:
            data: DataFrame to save
            file_path: Output file path
            engine: Parquet engine to use
            **kwargs: Additional arguments passed to pd.DataFrame.to_parquet
        """
        # Determine engine
        if engine == 'auto':
            engine = 'pyarrow' if PARQUET_AVAILABLE else 'fastparquet'
        
        # Ensure output directory exists
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Saving Parquet file: {output_path}")
            data.to_parquet(output_path, engine=engine, **kwargs)
        except Exception as e:
            raise DataIOError(f"Failed to save Parquet file {output_path}: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached files."""
        if not self._cache_enabled:
            return
        
        cache_dir = Path(self.cache_dir)
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                    logger.info(f"Removed cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a data file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Dictionary with file information
        """
        resolved_path = Path(self.resolve_file_path(file_path))
        
        info = {
            'path': str(resolved_path),
            'size_bytes': resolved_path.stat().st_size,
            'modified': datetime.fromtimestamp(resolved_path.stat().st_mtime),
            'format': resolved_path.suffix.lower()
        }
        
        # Add size in human-readable format
        size_mb = info['size_bytes'] / (1024 * 1024)
        if size_mb < 1:
            info['size_human'] = f"{info['size_bytes'] / 1024:.1f} KB"
        else:
            info['size_human'] = f"{size_mb:.1f} MB"
        
        return info


# Convenience functions for quick data loading
def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Quick CSV loading function."""
    loader = DataLoader()
    return loader.load_csv(file_path, **kwargs)


def load_parquet(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Quick Parquet loading function."""
    loader = DataLoader()
    return loader.load_parquet(file_path, **kwargs)


def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Quick data loading function with auto-format detection."""
    loader = DataLoader()
    return loader.load_data(file_path, **kwargs)
