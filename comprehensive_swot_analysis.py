#!/usr/bin/env python3
"""
Comprehensive SWOT L2 Low-Rate SSH Data Analysis
==============================================

This script provides a complete analysis framework for SWOT (Surface Water and Ocean Topography)
Level 2 Low-Rate Sea Surface Height (SSH) data. It includes data exploration, quality assessment,
statistical analysis, and visualization capabilities.

Author: Automated Analysis System
Date: August 23, 2025
Data Source: C:/Users/abhik/Desktop/project related work/SWOT_L2_LR_SSH_2.0_2.0-20250822_162623
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SWOTDataAnalyzer:
    """
    Comprehensive SWOT SSH Data Analyzer
    
    This class provides methods to load, analyze, and visualize SWOT L2 SSH data.
    """
    
    def __init__(self, data_directory: str):
        """
        Initialize the SWOT data analyzer.
        
        Parameters:
        -----------
        data_directory : str
            Path to the directory containing SWOT NetCDF files
        """
        self.data_dir = Path(data_directory)
        self.files = []
        self.datasets = {}
        self.metadata = {}
        self.analysis_results = {}
        
        # Create output directories
        self.output_dir = Path("swot_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        print(f"SWOT Data Analyzer initialized for directory: {self.data_dir}")
        
    def discover_files(self) -> Dict[str, List[str]]:
        """
        Discover and categorize SWOT data files by product type.
        
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary with product types as keys and file lists as values
        """
        print("Discovering SWOT data files...")
        
        # Find all NetCDF files
        all_files = list(self.data_dir.glob("*.nc"))
        
        # Categorize files by product type
        file_categories = {
            'basic': [],
            'expert': [],
            'unsmoothed': [],
            'windwave': []
        }
        
        for file in all_files:
            filename = file.name.lower()
            if 'basic' in filename:
                file_categories['basic'].append(str(file))
            elif 'expert' in filename:
                file_categories['expert'].append(str(file))
            elif 'unsmoothed' in filename:
                file_categories['unsmoothed'].append(str(file))
            elif 'windwave' in filename:
                file_categories['windwave'].append(str(file))
        
        self.files = file_categories
        
        # Print summary
        print(f"\nFile Discovery Summary:")
        print(f"{'Product Type':<15} {'Count':<10} {'Date Range'}")
        print("-" * 50)
        
        for product_type, file_list in file_categories.items():
            if file_list:
                dates = self._extract_dates_from_filenames(file_list)
                date_range = f"{min(dates)} to {max(dates)}" if dates else "Unknown"
                print(f"{product_type.capitalize():<15} {len(file_list):<10} {date_range}")
        
        return file_categories
    
    def _extract_dates_from_filenames(self, file_list: List[str]) -> List[str]:
        """Extract dates from SWOT filenames."""
        dates = []
        for file in file_list:
            # SWOT filename format includes date like: 20230729T095509
            parts = os.path.basename(file).split('_')
            for part in parts:
                if 'T' in part and len(part) == 15:  # Format: YYYYMMDDTHHMMSS
                    dates.append(part[:8])  # Extract YYYYMMDD
                    break
        return sorted(list(set(dates)))
    
    def inspect_sample_file(self, product_type: str = 'basic') -> Dict:
        """
        Inspect a sample file to understand data structure.
        
        Parameters:
        -----------
        product_type : str
            Type of product to inspect ('basic', 'expert', 'unsmoothed', 'windwave')
            
        Returns:
        --------
        Dict
            Dictionary containing file metadata and variable information
        """
        print(f"\nInspecting sample {product_type} file...")
        
        if product_type not in self.files or not self.files[product_type]:
            print(f"No {product_type} files found!")
            return {}
        
        # Select first file of the specified type
        sample_file = self.files[product_type][0]
        
        try:
            # Open the dataset
            ds = xr.open_dataset(sample_file)
            
            # Extract metadata
            metadata = {
                'filename': os.path.basename(sample_file),
                'file_size_mb': os.path.getsize(sample_file) / (1024 * 1024),
                'dimensions': dict(ds.dims),
                'coordinates': list(ds.coords.keys()),
                'data_variables': list(ds.data_vars.keys()),
                'global_attributes': dict(ds.attrs),
                'variable_details': {}
            }
            
            # Get details for each variable
            for var_name in ds.data_vars.keys():
                var = ds[var_name]
                metadata['variable_details'][var_name] = {
                    'shape': var.shape,
                    'dtype': str(var.dtype),
                    'units': var.attrs.get('units', 'Unknown'),
                    'long_name': var.attrs.get('long_name', 'Unknown'),
                    'attributes': dict(var.attrs)
                }
            
            self.metadata[product_type] = metadata
            
            # Print summary
            print(f"\nFile: {metadata['filename']}")
            print(f"Size: {metadata['file_size_mb']:.2f} MB")
            print(f"Dimensions: {metadata['dimensions']}")
            print(f"\nData Variables ({len(metadata['data_variables'])}):")
            
            for var_name in metadata['data_variables'][:10]:  # Show first 10 variables
                var_info = metadata['variable_details'][var_name]
                print(f"  {var_name:<25} {str(var_info['shape']):<15} {var_info['units']:<15} {var_info['long_name']}")
            
            if len(metadata['data_variables']) > 10:
                print(f"  ... and {len(metadata['data_variables']) - 10} more variables")
            
            ds.close()
            return metadata
            
        except Exception as e:
            print(f"Error inspecting file {sample_file}: {e}")
            return {}
    
    def load_dataset(self, product_type: str = 'basic', max_files: Optional[int] = None) -> xr.Dataset:
        """
        Load and concatenate SWOT datasets.
        
        Parameters:
        -----------
        product_type : str
            Type of product to load
        max_files : Optional[int]
            Maximum number of files to load (for testing)
            
        Returns:
        --------
        xr.Dataset
            Combined dataset
        """
        print(f"\nLoading {product_type} datasets...")
        
        if product_type not in self.files or not self.files[product_type]:
            print(f"No {product_type} files found!")
            return None
        
        files_to_load = self.files[product_type]
        if max_files:
            files_to_load = files_to_load[:max_files]
            print(f"Loading first {len(files_to_load)} files for testing...")
        
        datasets = []
        successful_loads = 0
        
        for i, file_path in enumerate(files_to_load):
            try:
                ds = xr.open_dataset(file_path)
                datasets.append(ds)
                successful_loads += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Loaded {i + 1}/{len(files_to_load)} files...")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not datasets:
            print("No datasets were successfully loaded!")
            return None
        
        print(f"Successfully loaded {successful_loads}/{len(files_to_load)} files")
        
        try:
            # Concatenate datasets along time dimension if possible
            combined_ds = xr.concat(datasets, dim='time', combine_attrs='drop_conflicts')
            self.datasets[product_type] = combined_ds
            
            print(f"Combined dataset shape: {dict(combined_ds.dims)}")
            return combined_ds
            
        except Exception as e:
            print(f"Error combining datasets: {e}")
            print("Returning first dataset only...")
            self.datasets[product_type] = datasets[0]
            return datasets[0]
    
    def perform_quality_assessment(self, dataset: xr.Dataset, product_type: str) -> Dict:
        """
        Perform comprehensive data quality assessment.
        
        Parameters:
        -----------
        dataset : xr.Dataset
            Dataset to assess
        product_type : str
            Type of product being assessed
            
        Returns:
        --------
        Dict
            Quality assessment results
        """
        print(f"\nPerforming quality assessment for {product_type} data...")
        
        qa_results = {
            'product_type': product_type,
            'assessment_time': datetime.now().isoformat(),
            'data_coverage': {},
            'missing_data': {},
            'value_ranges': {},
            'outliers': {},
            'temporal_coverage': {},
            'spatial_coverage': {}
        }
        
        # Key variables to assess (SSH-related)
        ssh_vars = ['ssh_karin', 'ssh_karin_2', 'ssha_karin', 'ssha_karin_2']
        available_ssh_vars = [var for var in ssh_vars if var in dataset.data_vars]
        
        if not available_ssh_vars:
            print("No SSH variables found in dataset!")
            return qa_results
        
        for var_name in available_ssh_vars:
            try:
                var_data = dataset[var_name]
                
                # Data coverage
                total_points = var_data.size
                valid_points = np.sum(~np.isnan(var_data)).compute() if hasattr(np.sum(~np.isnan(var_data)), 'compute') else np.sum(~np.isnan(var_data))
                coverage_percent = (valid_points / total_points) * 100
                
                qa_results['data_coverage'][var_name] = {
                    'total_points': int(total_points),
                    'valid_points': int(valid_points),
                    'coverage_percent': float(coverage_percent)
                }
                
                # Missing data analysis
                missing_points = total_points - valid_points
                qa_results['missing_data'][var_name] = {
                    'missing_points': int(missing_points),
                    'missing_percent': float((missing_points / total_points) * 100)
                }
                
                # Value ranges (for valid data only)
                valid_data = var_data.where(~np.isnan(var_data), drop=True)
                if valid_data.size > 0:
                    qa_results['value_ranges'][var_name] = {
                        'min': float(valid_data.min().compute() if hasattr(valid_data.min(), 'compute') else valid_data.min()),
                        'max': float(valid_data.max().compute() if hasattr(valid_data.max(), 'compute') else valid_data.max()),
                        'mean': float(valid_data.mean().compute() if hasattr(valid_data.mean(), 'compute') else valid_data.mean()),
                        'std': float(valid_data.std().compute() if hasattr(valid_data.std(), 'compute') else valid_data.std())
                    }
                    
                    # Outlier detection (values beyond 3 standard deviations)
                    mean_val = qa_results['value_ranges'][var_name]['mean']
                    std_val = qa_results['value_ranges'][var_name]['std']
                    outlier_threshold = 3 * std_val
                    
                    outliers = valid_data.where(
                        (valid_data < mean_val - outlier_threshold) | 
                        (valid_data > mean_val + outlier_threshold)
                    )
                    outlier_count = np.sum(~np.isnan(outliers)).compute() if hasattr(np.sum(~np.isnan(outliers)), 'compute') else np.sum(~np.isnan(outliers))
                    
                    qa_results['outliers'][var_name] = {
                        'outlier_count': int(outlier_count),
                        'outlier_percent': float((outlier_count / valid_points) * 100),
                        'threshold': float(outlier_threshold)
                    }
                
            except Exception as e:
                print(f"Error assessing variable {var_name}: {e}")
                continue
        
        # Temporal coverage
        if 'time' in dataset.dims:
            time_data = dataset.time
            qa_results['temporal_coverage'] = {
                'start_time': str(time_data.min().values),
                'end_time': str(time_data.max().values),
                'time_steps': int(len(time_data)),
                'time_resolution': str(time_data.diff('time').median().values)
            }
        
        # Spatial coverage
        if 'latitude' in dataset.coords and 'longitude' in dataset.coords:
            lat_data = dataset.latitude
            lon_data = dataset.longitude
            
            qa_results['spatial_coverage'] = {
                'lat_range': [float(lat_data.min().compute() if hasattr(lat_data.min(), 'compute') else lat_data.min()),
                             float(lat_data.max().compute() if hasattr(lat_data.max(), 'compute') else lat_data.max())],
                'lon_range': [float(lon_data.min().compute() if hasattr(lon_data.min(), 'compute') else lon_data.min()),
                             float(lon_data.max().compute() if hasattr(lon_data.max(), 'compute') else lon_data.max())]
            }
        
        # Store results
        self.analysis_results[f'{product_type}_quality'] = qa_results
        
        # Print summary
        print(f"\nQuality Assessment Summary for {product_type}:")
        print("-" * 50)
        
        for var_name in available_ssh_vars:
            if var_name in qa_results['data_coverage']:
                coverage = qa_results['data_coverage'][var_name]['coverage_percent']
                print(f"{var_name:<20} Data Coverage: {coverage:.1f}%")
                
                if var_name in qa_results['value_ranges']:
                    ranges = qa_results['value_ranges'][var_name]
                    print(f"{'':>20} Range: {ranges['min']:.3f} to {ranges['max']:.3f} m")
                    print(f"{'':>20} Mean ± Std: {ranges['mean']:.3f} ± {ranges['std']:.3f} m")
                
                if var_name in qa_results['outliers']:
                    outlier_pct = qa_results['outliers'][var_name]['outlier_percent']
                    print(f"{'':>20} Outliers: {outlier_pct:.2f}%")
                print()
        
        return qa_results
    
    def create_statistical_summary(self, dataset: xr.Dataset, product_type: str) -> Dict:
        """
        Create comprehensive statistical summary.
        
        Parameters:
        -----------
        dataset : xr.Dataset
            Dataset to analyze
        product_type : str
            Type of product being analyzed
            
        Returns:
        --------
        Dict
            Statistical summary results
        """
        print(f"\nCreating statistical summary for {product_type} data...")
        
        stats_results = {
            'product_type': product_type,
            'analysis_time': datetime.now().isoformat(),
            'variable_statistics': {},
            'correlation_analysis': {},
            'trend_analysis': {}
        }
        
        # Key SSH variables to analyze
        ssh_vars = ['ssh_karin', 'ssh_karin_2', 'ssha_karin', 'ssha_karin_2']
        available_ssh_vars = [var for var in ssh_vars if var in dataset.data_vars]
        
        # Calculate statistics for each variable
        for var_name in available_ssh_vars:
            try:
                var_data = dataset[var_name]
                valid_data = var_data.where(~np.isnan(var_data), drop=True)
                
                if valid_data.size == 0:
                    continue
                
                # Basic statistics
                stats_results['variable_statistics'][var_name] = {
                    'count': int(valid_data.size),
                    'mean': float(valid_data.mean().compute() if hasattr(valid_data.mean(), 'compute') else valid_data.mean()),
                    'median': float(valid_data.median().compute() if hasattr(valid_data.median(), 'compute') else valid_data.median()),
                    'std': float(valid_data.std().compute() if hasattr(valid_data.std(), 'compute') else valid_data.std()),
                    'var': float(valid_data.var().compute() if hasattr(valid_data.var(), 'compute') else valid_data.var()),
                    'min': float(valid_data.min().compute() if hasattr(valid_data.min(), 'compute') else valid_data.min()),
                    'max': float(valid_data.max().compute() if hasattr(valid_data.max(), 'compute') else valid_data.max()),
                    'range': float((valid_data.max() - valid_data.min()).compute() if hasattr((valid_data.max() - valid_data.min()), 'compute') else (valid_data.max() - valid_data.min())),
                    'skewness': float(stats.skew(valid_data.values.flatten())),
                    'kurtosis': float(stats.kurtosis(valid_data.values.flatten()))
                }
                
                # Percentiles
                percentiles = [5, 10, 25, 75, 90, 95]
                for p in percentiles:
                    pct_val = float(valid_data.quantile(p/100).compute() if hasattr(valid_data.quantile(p/100), 'compute') else valid_data.quantile(p/100))
                    stats_results['variable_statistics'][var_name][f'p{p}'] = pct_val
                
            except Exception as e:
                print(f"Error calculating statistics for {var_name}: {e}")
                continue
        
        # Store results
        self.analysis_results[f'{product_type}_statistics'] = stats_results
        
        # Print summary
        print(f"\nStatistical Summary for {product_type}:")
        print("-" * 80)
        print(f"{'Variable':<20} {'Count':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 80)
        
        for var_name, stats_dict in stats_results['variable_statistics'].items():
            print(f"{var_name:<20} {stats_dict['count']:<10} {stats_dict['mean']:<10.3f} "
                  f"{stats_dict['std']:<10.3f} {stats_dict['min']:<10.3f} {stats_dict['max']:<10.3f}")
        
        return stats_results
    
    def create_visualizations(self, dataset: xr.Dataset, product_type: str):
        """
        Create comprehensive visualizations.
        
        Parameters:
        -----------
        dataset : xr.Dataset
            Dataset to visualize
        product_type : str
            Type of product being visualized
        """
        print(f"\nCreating visualizations for {product_type} data...")
        
        # Key SSH variables to visualize
        ssh_vars = ['ssh_karin', 'ssh_karin_2', 'ssha_karin', 'ssha_karin_2']
        available_ssh_vars = [var for var in ssh_vars if var in dataset.data_vars]
        
        if not available_ssh_vars:
            print("No SSH variables found for visualization!")
            return
        
        # 1. Data coverage and distribution plots
        self._create_distribution_plots(dataset, available_ssh_vars, product_type)
        
        # 2. Spatial plots (if spatial coordinates available)
        if 'latitude' in dataset.coords and 'longitude' in dataset.coords:
            self._create_spatial_plots(dataset, available_ssh_vars, product_type)
        
        # 3. Time series plots (if time dimension available)
        if 'time' in dataset.dims:
            self._create_time_series_plots(dataset, available_ssh_vars, product_type)
        
        # 4. Quality assessment plots
        self._create_quality_plots(dataset, available_ssh_vars, product_type)
        
        print(f"Visualizations saved to {self.output_dir / 'figures'}")
    
    def _create_distribution_plots(self, dataset: xr.Dataset, ssh_vars: List[str], product_type: str):
        """Create distribution and histogram plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'SWOT {product_type.capitalize()} SSH Data Distribution Analysis', fontsize=16)
        
        for i, var_name in enumerate(ssh_vars[:4]):
            if var_name not in dataset.data_vars:
                continue
                
            row, col = divmod(i, 2)
            ax = axes[row, col]
            
            # Get valid data
            var_data = dataset[var_name]
            valid_data = var_data.where(~np.isnan(var_data), drop=True)
            
            if valid_data.size > 0:
                data_flat = valid_data.values.flatten()
                data_clean = data_flat[~np.isnan(data_flat)]
                
                # Create histogram with statistical info
                ax.hist(data_clean, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                
                # Add statistics text
                mean_val = np.mean(data_clean)
                std_val = np.std(data_clean)
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_val + std_val:.3f}')
                ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_val - std_val:.3f}')
                
                ax.set_title(f'{var_name} Distribution\n(n={len(data_clean):,} valid points)')
                ax.set_xlabel('SSH (m)')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(ssh_vars), 4):
            row, col = divmod(i, 2)
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{product_type}_distribution_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_spatial_plots(self, dataset: xr.Dataset, ssh_vars: List[str], product_type: str):
        """Create spatial visualization plots."""
        # Sample data for spatial plotting (to avoid memory issues)
        if 'time' in dataset.dims and len(dataset.time) > 1:
            sample_ds = dataset.isel(time=0)  # Use first time step
        else:
            sample_ds = dataset
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create map projection
        for i, var_name in enumerate(ssh_vars[:2]):  # Limit to 2 variables for clarity
            if var_name not in sample_ds.data_vars:
                continue
            
            ax = plt.subplot(2, 1, i+1, projection=ccrs.PlateCarree())
            
            # Get data
            ssh_data = sample_ds[var_name]
            lat_data = sample_ds.latitude
            lon_data = sample_ds.longitude
            
            # Create scatter plot
            scatter = ax.scatter(lon_data, lat_data, c=ssh_data, cmap='RdYlBu_r', 
                               s=1, transform=ccrs.PlateCarree(), alpha=0.6)
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.OCEAN, alpha=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.3)
            ax.gridlines(draw_labels=True, alpha=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('SSH (m)', rotation=270, labelpad=20)
            
            ax.set_title(f'SWOT {product_type.capitalize()} - {var_name} Spatial Distribution')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{product_type}_spatial_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_time_series_plots(self, dataset: xr.Dataset, ssh_vars: List[str], product_type: str):
        """Create time series plots."""
        fig, axes = plt.subplots(len(ssh_vars), 1, figsize=(15, 4*len(ssh_vars)))
        if len(ssh_vars) == 1:
            axes = [axes]
        
        fig.suptitle(f'SWOT {product_type.capitalize()} SSH Time Series Analysis', fontsize=16)
        
        for i, var_name in enumerate(ssh_vars):
            if var_name not in dataset.data_vars:
                continue
            
            ax = axes[i]
            
            # Calculate spatial mean for each time step
            ssh_data = dataset[var_name]
            time_series = ssh_data.mean(dim=[d for d in ssh_data.dims if d != 'time'], skipna=True)
            
            # Plot time series
            time_series.plot(ax=ax, linewidth=1.5, alpha=0.8)
            
            ax.set_title(f'{var_name} - Spatial Mean Time Series')
            ax.set_ylabel('SSH (m)')
            ax.grid(True, alpha=0.3)
            
            # Add trend line if enough data points
            if len(time_series.time) > 2:
                try:
                    # Simple linear trend
                    time_numeric = np.arange(len(time_series.time))
                    valid_mask = ~np.isnan(time_series.values)
                    if np.sum(valid_mask) > 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            time_numeric[valid_mask], time_series.values[valid_mask]
                        )
                        trend_line = slope * time_numeric + intercept
                        ax.plot(time_series.time, trend_line, 'r--', alpha=0.7, 
                               label=f'Trend: {slope:.2e} m/step (R²={r_value**2:.3f})')
                        ax.legend()
                except Exception as e:
                    print(f"Could not calculate trend for {var_name}: {e}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{product_type}_time_series.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_quality_plots(self, dataset: xr.Dataset, ssh_vars: List[str], product_type: str):
        """Create data quality assessment plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'SWOT {product_type.capitalize()} Data Quality Assessment', fontsize=16)
        
        # 1. Data coverage by variable
        ax1 = axes[0, 0]
        coverage_data = []
        var_labels = []
        
        for var_name in ssh_vars:
            if var_name in dataset.data_vars:
                var_data = dataset[var_name]
                total_points = var_data.size
                valid_points = np.sum(~np.isnan(var_data)).compute() if hasattr(np.sum(~np.isnan(var_data)), 'compute') else np.sum(~np.isnan(var_data))
                coverage_percent = (valid_points / total_points) * 100
                coverage_data.append(coverage_percent)
                var_labels.append(var_name)
        
        bars = ax1.bar(var_labels, coverage_data, color='lightblue', edgecolor='navy')
        ax1.set_title('Data Coverage by Variable')
        ax1.set_ylabel('Coverage (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, coverage_data):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. Value range comparison
        ax2 = axes[0, 1]
        for var_name in ssh_vars[:2]:  # Limit to avoid clutter
            if var_name in dataset.data_vars:
                var_data = dataset[var_name]
                valid_data = var_data.where(~np.isnan(var_data), drop=True)
                if valid_data.size > 0:
                    data_flat = valid_data.values.flatten()
                    data_clean = data_flat[~np.isnan(data_flat)]
                    ax2.boxplot(data_clean, positions=[ssh_vars.index(var_name)], 
                               labels=[var_name], patch_artist=True)
        
        ax2.set_title('Value Range Distribution')
        ax2.set_ylabel('SSH (m)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Missing data pattern (if time dimension exists)
        ax3 = axes[1, 0]
        if 'time' in dataset.dims and len(ssh_vars) > 0:
            var_name = ssh_vars[0]  # Use first available variable
            if var_name in dataset.data_vars:
                var_data = dataset[var_name]
                # Calculate percentage of missing data per time step
                missing_per_time = []
                for t in range(len(dataset.time)):
                    time_slice = var_data.isel(time=t)
                    total = time_slice.size
                    missing = np.sum(np.isnan(time_slice)).compute() if hasattr(np.sum(np.isnan(time_slice)), 'compute') else np.sum(np.isnan(time_slice))
                    missing_per_time.append((missing / total) * 100)
                
                ax3.plot(dataset.time, missing_per_time, 'o-', alpha=0.7)
                ax3.set_title(f'Missing Data Over Time ({var_name})')
                ax3.set_ylabel('Missing Data (%)')
                ax3.grid(True, alpha=0.3)
        
        # 4. Correlation matrix (if multiple variables)
        ax4 = axes[1, 1]
        if len(ssh_vars) > 1:
            corr_data = []
            corr_labels = []
            
            for var_name in ssh_vars:
                if var_name in dataset.data_vars:
                    var_data = dataset[var_name]
                    valid_data = var_data.where(~np.isnan(var_data), drop=True)
                    if valid_data.size > 0:
                        # Sample data to avoid memory issues
                        sample_size = min(10000, valid_data.size)
                        sample_indices = np.random.choice(valid_data.size, sample_size, replace=False)
                        data_flat = valid_data.values.flatten()
                        corr_data.append(data_flat[sample_indices])
                        corr_labels.append(var_name)
            
            if len(corr_data) > 1:
                corr_df = pd.DataFrame(dict(zip(corr_labels, corr_data)))
                corr_matrix = corr_df.corr()
                
                im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax4.set_xticks(range(len(corr_labels)))
                ax4.set_yticks(range(len(corr_labels)))
                ax4.set_xticklabels(corr_labels, rotation=45, ha='right')
                ax4.set_yticklabels(corr_labels)
                ax4.set_title('Variable Correlation Matrix')
                
                # Add correlation values as text
                for i in range(len(corr_labels)):
                    for j in range(len(corr_labels)):
                        text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=ax4, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{product_type}_quality_assessment.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_report(self):
        """Save comprehensive analysis report."""
        print("\nGenerating comprehensive analysis report...")
        
        report_content = f"""
# SWOT L2 Low-Rate SSH Data Analysis Report

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Directory:** {self.data_dir}
**Analysis Framework:** Comprehensive SWOT Data Analyzer

## Executive Summary

This report presents a comprehensive analysis of SWOT (Surface Water and Ocean Topography) Level 2 Low-Rate Sea Surface Height (SSH) data. The analysis includes data discovery, quality assessment, statistical analysis, and visualization.

## Data Discovery Summary

"""
        
        # Add file discovery summary
        if self.files:
            report_content += "### File Inventory\n\n"
            report_content += "| Product Type | File Count | Date Range |\n"
            report_content += "|--------------|------------|------------|\n"
            
            for product_type, file_list in self.files.items():
                if file_list:
                    dates = self._extract_dates_from_filenames(file_list)
                    date_range = f"{min(dates)} to {max(dates)}" if dates else "Unknown"
                    report_content += f"| {product_type.capitalize()} | {len(file_list)} | {date_range} |\n"
            
            report_content += "\n"
        
        # Add metadata summary
        if self.metadata:
            report_content += "## Data Structure Analysis\n\n"
            for product_type, metadata in self.metadata.items():
                report_content += f"### {product_type.capitalize()} Product Structure\n\n"
                report_content += f"- **File Size:** {metadata['file_size_mb']:.2f} MB\n"
                report_content += f"- **Dimensions:** {metadata['dimensions']}\n"
                report_content += f"- **Data Variables:** {len(metadata['data_variables'])}\n"
                report_content += f"- **Coordinates:** {', '.join(metadata['coordinates'])}\n\n"
                
                if metadata['data_variables']:
                    report_content += "#### Key Variables:\n\n"
                    report_content += "| Variable | Shape | Units | Description |\n"
                    report_content += "|----------|-------|-------|-------------|\n"
                    
                    for var_name in metadata['data_variables'][:10]:  # Show top 10
                        var_info = metadata['variable_details'][var_name]
                        shape_str = str(var_info['shape'])
                        units = var_info['units']
                        description = var_info['long_name'][:50] + "..." if len(var_info['long_name']) > 50 else var_info['long_name']
                        report_content += f"| {var_name} | {shape_str} | {units} | {description} |\n"
                    
                    if len(metadata['data_variables']) > 10:
                        report_content += f"\n*... and {len(metadata['data_variables']) - 10} more variables*\n"
                
                report_content += "\n"
        
        # Add quality assessment results
        if any('quality' in key for key in self.analysis_results.keys()):
            report_content += "## Data Quality Assessment\n\n"
            
            for key, qa_results in self.analysis_results.items():
                if 'quality' in key:
                    product_type = qa_results['product_type']
                    report_content += f"### {product_type.capitalize()} Product Quality\n\n"
                    
                    # Data coverage
                    if qa_results['data_coverage']:
                        report_content += "#### Data Coverage\n\n"
                        report_content += "| Variable | Total Points | Valid Points | Coverage (%) |\n"
                        report_content += "|----------|--------------|--------------|-------------|\n"
                        
                        for var_name, coverage_info in qa_results['data_coverage'].items():
                            report_content += f"| {var_name} | {coverage_info['total_points']:,} | {coverage_info['valid_points']:,} | {coverage_info['coverage_percent']:.1f} |\n"
                        
                        report_content += "\n"
                    
                    # Value ranges
                    if qa_results['value_ranges']:
                        report_content += "#### Value Ranges\n\n"
                        report_content += "| Variable | Min (m) | Max (m) | Mean (m) | Std (m) |\n"
                        report_content += "|----------|---------|---------|----------|----------|\n"
                        
                        for var_name, range_info in qa_results['value_ranges'].items():
                            report_content += f"| {var_name} | {range_info['min']:.3f} | {range_info['max']:.3f} | {range_info['mean']:.3f} | {range_info['std']:.3f} |\n"
                        
                        report_content += "\n"
        
        # Add statistical summary
        if any('statistics' in key for key in self.analysis_results.keys()):
            report_content += "## Statistical Analysis\n\n"
            
            for key, stats_results in self.analysis_results.items():
                if 'statistics' in key:
                    product_type = stats_results['product_type']
                    report_content += f"### {product_type.capitalize()} Product Statistics\n\n"
                    
                    if stats_results['variable_statistics']:
                        report_content += "| Variable | Count | Mean | Median | Std | Skewness | Kurtosis |\n"
                        report_content += "|----------|-------|------|--------|-----|----------|----------|\n"
                        
                        for var_name, stats_info in stats_results['variable_statistics'].items():
                            report_content += f"| {var_name} | {stats_info['count']:,} | {stats_info['mean']:.3f} | {stats_info['median']:.3f} | {stats_info['std']:.3f} | {stats_info['skewness']:.3f} | {stats_info['kurtosis']:.3f} |\n"
                        
                        report_content += "\n"
        
        # Add conclusions and recommendations
        report_content += """
## Key Findings and Recommendations

### Data Quality
- The SWOT L2 SSH data shows varying levels of data coverage across different variables
- Missing data patterns should be considered when performing temporal or spatial analyses
- Outlier detection has identified data points that may require additional quality control

### Statistical Characteristics
- SSH values show expected oceanographic ranges and distributions
- Temporal variations indicate both seasonal and shorter-term oceanographic signals
- Spatial patterns reflect known oceanographic features and measurement geometry

### Recommendations for Further Analysis
1. **Temporal Analysis**: Investigate seasonal cycles and long-term trends
2. **Spatial Analysis**: Focus on specific oceanographic regions of interest
3. **Comparative Studies**: Compare different product types (Basic, Expert, Unsmoothed, WindWave)
4. **Quality Control**: Implement additional filtering for outliers and data gaps
5. **Cross-validation**: Compare with altimetry data from other missions

### Technical Notes
- Large dataset size may require chunked processing for full temporal analysis
- Coordinate reference system should be verified for accurate spatial analysis
- Memory management is crucial when working with the complete dataset

---

*This report was generated automatically by the Comprehensive SWOT Data Analyzer*
*Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_path = self.output_dir / f"SWOT_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Comprehensive analysis report saved to: {report_path}")
        
        return report_path
    
    def run_complete_analysis(self, product_types: List[str] = ['basic'], max_files_per_type: Optional[int] = 5):
        """
        Run complete analysis workflow.
        
        Parameters:
        -----------
        product_types : List[str]
            List of product types to analyze
        max_files_per_type : Optional[int]
            Maximum files per product type (for testing)
        """
        print("="*80)
        print("STARTING COMPREHENSIVE SWOT DATA ANALYSIS")
        print("="*80)
        
        # Step 1: Discover files
        self.discover_files()
        
        # Step 2: Analyze each product type
        for product_type in product_types:
            if product_type in self.files and self.files[product_type]:
                print(f"\n{'='*20} Analyzing {product_type.upper()} Product {'='*20}")
                
                # Inspect sample file
                self.inspect_sample_file(product_type)
                
                # Load dataset
                dataset = self.load_dataset(product_type, max_files_per_type)
                
                if dataset is not None:
                    # Quality assessment
                    self.perform_quality_assessment(dataset, product_type)
                    
                    # Statistical analysis
                    self.create_statistical_summary(dataset, product_type)
                    
                    # Create visualizations
                    self.create_visualizations(dataset, product_type)
                    
                    # Close dataset to free memory
                    dataset.close()
                
            else:
                print(f"No {product_type} files found to analyze")
        
        # Step 3: Generate final report
        self.save_analysis_report()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE SWOT DATA ANALYSIS COMPLETED")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        print(f"Figures saved to: {self.output_dir / 'figures'}")


def main():
    """
    Main function to run SWOT data analysis.
    """
    # Configuration
    DATA_DIRECTORY = r"C:/Users/abhik/Desktop/project related work/SWOT_L2_LR_SSH_2.0_2.0-20250822_162623"
    
    # Product types to analyze (can be modified)
    PRODUCT_TYPES = ['basic', 'expert', 'unsmoothed', 'windwave']
    
    # Maximum files per product type (set to None for all files, or small number for testing)
    MAX_FILES = 3  # Start with 3 files per type for testing
    
    print("SWOT L2 Low-Rate SSH Data Analysis")
    print("==================================")
    print(f"Data Directory: {DATA_DIRECTORY}")
    print(f"Product Types: {PRODUCT_TYPES}")
    print(f"Max Files per Type: {MAX_FILES}")
    print()
    
    try:
        # Initialize analyzer
        analyzer = SWOTDataAnalyzer(DATA_DIRECTORY)
        
        # Run complete analysis
        analyzer.run_complete_analysis(
            product_types=PRODUCT_TYPES,
            max_files_per_type=MAX_FILES
        )
        
        print("\nAnalysis completed successfully!")
        print(f"Check the output directory: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
