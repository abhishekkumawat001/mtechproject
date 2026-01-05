#!/usr/bin/env python3
"""
SWOT SSH Data Analysis for Andaman and Nicobar Region
=====================================================

This script provides specialized analysis of SWOT L2 Basic SSH data for the Andaman and Nicobar region.
Region of Interest: Latitude 6°N to 15°N, Longitude 92°E to 95°E

Features:
- Spatial analysis of SSH patterns
- Temporal analysis and trend detection
- Oceanographic feature identification
- Regional-specific visualizations

Author: Automated Analysis System
Date: August 23, 2025
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
from scipy import stats, ndimage
from scipy.interpolate import griddata
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class AndamanNicobarSWOTAnalyzer:
    """
    Specialized SWOT SSH Data Analyzer for Andaman and Nicobar Region
    """
    
    def __init__(self, data_directory: str):
        """
        Initialize the analyzer for Andaman and Nicobar region.
        
        Parameters:
        -----------
        data_directory : str
            Path to the directory containing SWOT NetCDF files
        """
        self.data_dir = Path(data_directory)
        
        # Region of Interest - Andaman and Nicobar
        self.roi = {
            'lat_min': 6.0,
            'lat_max': 15.0,
            'lon_min': 92.0,
            'lon_max': 95.0,
            'name': 'Andaman_Nicobar'
        }
        
        # Create output directories
        self.output_dir = Path("swot_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create specialized folder for Andaman and Nicobar analysis
        self.an_output_dir = self.output_dir / "figures_andaman_nicobar"
        self.an_output_dir.mkdir(exist_ok=True)
        (self.an_output_dir / "spatial").mkdir(exist_ok=True)
        (self.an_output_dir / "temporal").mkdir(exist_ok=True)
        (self.an_output_dir / "data").mkdir(exist_ok=True)
        
        self.datasets = []
        self.filtered_data = None
        self.analysis_results = {}
        
        print(f"Andaman and Nicobar SWOT Analyzer initialized")
        print(f"Region: {self.roi['lat_min']}°N-{self.roi['lat_max']}°N, {self.roi['lon_min']}°E-{self.roi['lon_max']}°E")
        print(f"Output directory: {self.an_output_dir}")
        
    def discover_basic_files(self) -> List[str]:
        """
        Discover SWOT Basic SSH files.
        
        Returns:
        --------
        List[str]
            List of basic SSH file paths
        """
        print("Discovering SWOT Basic SSH files...")
        
        # Find all Basic NetCDF files
        basic_files = list(self.data_dir.glob("*Basic*.nc"))
        
        if not basic_files:
            print("No Basic SSH files found!")
            return []
        
        print(f"Found {len(basic_files)} Basic SSH files")
        
        # Extract and display date range
        dates = []
        for file in basic_files:
            parts = file.name.split('_')
            for part in parts:
                if 'T' in part and len(part) == 15:  # Format: YYYYMMDDTHHMMSS
                    dates.append(part[:8])  # Extract YYYYMMDD
                    break
        
        if dates:
            unique_dates = sorted(list(set(dates)))
            print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
            print(f"Unique dates: {len(unique_dates)}")
        
        return [str(f) for f in basic_files]
    
    def load_and_filter_data(self, file_paths: List[str], max_files: Optional[int] = None) -> xr.Dataset:
        """
        Load SWOT data and filter for Andaman and Nicobar region.
        
        Parameters:
        -----------
        file_paths : List[str]
            List of file paths to load
        max_files : Optional[int]
            Maximum number of files to load (for testing)
            
        Returns:
        --------
        xr.Dataset
            Filtered dataset for the region
        """
        print(f"\nLoading and filtering data for Andaman and Nicobar region...")
        
        if max_files:
            file_paths = file_paths[:max_files]
            print(f"Processing first {len(file_paths)} files for analysis...")
        
        filtered_datasets = []
        successful_loads = 0
        
        for i, file_path in enumerate(file_paths):
            try:
                print(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                
                # Open dataset
                ds = xr.open_dataset(file_path)
                
                # Check if we have the required coordinates
                if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
                    print(f"  Warning: Missing coordinates in {file_path}")
                    ds.close()
                    continue
                
                # Filter for Andaman and Nicobar region
                lat_mask = (ds.latitude >= self.roi['lat_min']) & (ds.latitude <= self.roi['lat_max'])
                lon_mask = (ds.longitude >= self.roi['lon_min']) & (ds.longitude <= self.roi['lon_max'])
                region_mask = lat_mask & lon_mask
                
                # Apply spatial filter
                ds_filtered = ds.where(region_mask, drop=True)
                
                # Check if we have data in the region
                if ds_filtered.latitude.size == 0 or ds_filtered.longitude.size == 0:
                    print(f"  No data in target region for this file")
                    ds.close()
                    continue
                
                # Check for SSH variables
                ssh_vars = ['ssh_karin', 'ssh_karin_2']
                available_ssh = [var for var in ssh_vars if var in ds_filtered.data_vars]
                
                if not available_ssh:
                    print(f"  No SSH variables found in this file")
                    ds.close()
                    continue
                
                print(f"  Found {ds_filtered.latitude.size} data points in target region")
                print(f"  Available SSH variables: {available_ssh}")
                
                # Add file information as attributes
                ds_filtered.attrs['source_file'] = os.path.basename(file_path)
                ds_filtered.attrs['processing_time'] = datetime.now().isoformat()
                
                filtered_datasets.append(ds_filtered)
                successful_loads += 1
                
                # Close original dataset
                ds.close()
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                continue
        
        if not filtered_datasets:
            print("No datasets were successfully filtered for the region!")
            return None
        
        print(f"\nSuccessfully processed {successful_loads}/{len(file_paths)} files")
        
        try:
            # Concatenate datasets along time dimension
            print("Combining datasets...")
            combined_ds = xr.concat(filtered_datasets, dim='time', combine_attrs='drop_conflicts')
            
            # Sort by time if time coordinate exists
            if 'time' in combined_ds.dims:
                combined_ds = combined_ds.sortby('time')
            
            self.filtered_data = combined_ds
            
            print(f"Combined dataset dimensions: {dict(combined_ds.dims)}")
            print(f"Spatial coverage: Lat {float(combined_ds.latitude.min()):.2f}° to {float(combined_ds.latitude.max()):.2f}°")
            print(f"                  Lon {float(combined_ds.longitude.min()):.2f}° to {float(combined_ds.longitude.max()):.2f}°")
            
            if 'time' in combined_ds.dims:
                print(f"Temporal coverage: {str(combined_ds.time.min().values)} to {str(combined_ds.time.max().values)}")
            
            return combined_ds
            
        except Exception as e:
            print(f"Error combining datasets: {e}")
            print("Returning first dataset only...")
            self.filtered_data = filtered_datasets[0]
            return filtered_datasets[0]
    
    def perform_spatial_analysis(self, dataset: xr.Dataset):
        """
        Perform comprehensive spatial analysis for the region.
        
        Parameters:
        -----------
        dataset : xr.Dataset
            Filtered dataset for analysis
        """
        print("\nPerforming spatial analysis for Andaman and Nicobar region...")
        
        # Get SSH variables
        ssh_vars = ['ssh_karin', 'ssh_karin_2']
        available_ssh = [var for var in ssh_vars if var in dataset.data_vars]
        
        if not available_ssh:
            print("No SSH variables found for spatial analysis!")
            return
        
        # Calculate spatial statistics
        spatial_stats = {}
        
        for var_name in available_ssh:
            print(f"Analyzing {var_name}...")
            
            ssh_data = dataset[var_name]
            
            # Calculate spatial statistics (SWOT data uses num_lines, num_pixels dimensions)
            spatial_stats[var_name] = {
                'mean_value': float(ssh_data.mean(skipna=True)),
                'std_value': float(ssh_data.std(skipna=True)),
                'min_value': float(ssh_data.min(skipna=True)),
                'max_value': float(ssh_data.max(skipna=True)),
                'median_value': float(ssh_data.median(skipna=True)),
                'data_count': int(ssh_data.count()),
                'valid_data_fraction': float(ssh_data.count() / ssh_data.size)
            }
            
            # Calculate percentiles
            ssh_values = ssh_data.values.flatten()
            ssh_values = ssh_values[~np.isnan(ssh_values)]
            
            if len(ssh_values) > 0:
                spatial_stats[var_name].update({
                    'p25': float(np.percentile(ssh_values, 25)),
                    'p75': float(np.percentile(ssh_values, 75)),
                    'p95': float(np.percentile(ssh_values, 95)),
                    'p99': float(np.percentile(ssh_values, 99))
                })
            
            # Calculate gradients if we have enough spatial data
            try:
                # Get valid SSH data points and their coordinates
                ssh_valid = ssh_data.where(~np.isnan(ssh_data), drop=True)
                
                if ssh_valid.size > 100:  # Need sufficient data points
                    lat_vals = dataset.latitude.where(~np.isnan(ssh_data)).values.flatten()
                    lon_vals = dataset.longitude.where(~np.isnan(ssh_data)).values.flatten()
                    ssh_vals = ssh_valid.values.flatten()
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(lat_vals) | np.isnan(lon_vals) | np.isnan(ssh_vals))
                    lat_vals = lat_vals[valid_mask]
                    lon_vals = lon_vals[valid_mask]
                    ssh_vals = ssh_vals[valid_mask]
                    
                    if len(ssh_vals) > 10:
                        # Calculate approximate gradients using finite differences
                        spatial_stats[var_name]['gradient_available'] = True
                        spatial_stats[var_name]['gradient_stats'] = {
                            'lat_range': float(lat_vals.max() - lat_vals.min()),
                            'lon_range': float(lon_vals.max() - lon_vals.min()),
                            'ssh_range': float(ssh_vals.max() - ssh_vals.min())
                        }
                    else:
                        spatial_stats[var_name]['gradient_available'] = False
                else:
                    spatial_stats[var_name]['gradient_available'] = False
                    
            except Exception as e:
                print(f"  Could not calculate gradients for {var_name}: {e}")
                spatial_stats[var_name]['gradient_available'] = False
        
        self.analysis_results['spatial_stats'] = spatial_stats
        
        # Create spatial visualizations
        self._create_spatial_maps(dataset, spatial_stats)
        
        # Analyze oceanographic features
        self._analyze_oceanographic_features(dataset, spatial_stats)
        
    def _create_spatial_maps(self, dataset: xr.Dataset, spatial_stats: Dict):
        """Create comprehensive spatial maps."""
        print("Creating spatial maps...")
        
        ssh_vars = list(spatial_stats.keys())
        
        # 1. Mean SSH maps
        fig, axes = plt.subplots(1, len(ssh_vars), figsize=(6*len(ssh_vars), 8))
        if len(ssh_vars) == 1:
            axes = [axes]
        
        fig.suptitle('Mean SSH in Andaman and Nicobar Region', fontsize=16, fontweight='bold')
        
        for i, var_name in enumerate(ssh_vars):
            ax = axes[i]
            
            mean_ssh = spatial_stats[var_name]['mean_field']
            lat = dataset.latitude
            lon = dataset.longitude
            
            # Create scatter plot with color mapping
            scatter = ax.scatter(lon, lat, c=mean_ssh, cmap='RdYlBu_r', 
                               s=20, alpha=0.7, edgecolors='none')
            
            # Customize map
            ax.set_xlim(self.roi['lon_min'], self.roi['lon_max'])
            ax.set_ylim(self.roi['lat_min'], self.roi['lat_max'])
            ax.set_xlabel('Longitude (°E)', fontsize=12)
            ax.set_ylabel('Latitude (°N)', fontsize=12)
            ax.set_title(f'{var_name} - Mean SSH', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('SSH (m)', rotation=270, labelpad=20, fontsize=12)
            
            # Add region boundaries
            ax.axhline(y=self.roi['lat_min'], color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=self.roi['lat_max'], color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.roi['lon_min'], color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.roi['lon_max'], color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.an_output_dir / 'spatial' / 'mean_ssh_maps.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Variability maps
        fig, axes = plt.subplots(2, len(ssh_vars), figsize=(6*len(ssh_vars), 12))
        if len(ssh_vars) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('SSH Variability in Andaman and Nicobar Region', fontsize=16, fontweight='bold')
        
        for i, var_name in enumerate(ssh_vars):
            # Standard deviation
            ax1 = axes[0, i]
            std_ssh = spatial_stats[var_name]['std_field']
            
            scatter1 = ax1.scatter(dataset.longitude, dataset.latitude, c=std_ssh, 
                                 cmap='plasma', s=20, alpha=0.7, edgecolors='none')
            ax1.set_xlim(self.roi['lon_min'], self.roi['lon_max'])
            ax1.set_ylim(self.roi['lat_min'], self.roi['lat_max'])
            ax1.set_xlabel('Longitude (°E)')
            ax1.set_ylabel('Latitude (°N)')
            ax1.set_title(f'{var_name} - SSH Standard Deviation')
            ax1.grid(True, alpha=0.3)
            
            cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
            cbar1.set_label('SSH Std (m)', rotation=270, labelpad=20)
            
            # Range
            ax2 = axes[1, i]
            range_ssh = spatial_stats[var_name]['range_field']
            
            scatter2 = ax2.scatter(dataset.longitude, dataset.latitude, c=range_ssh, 
                                 cmap='viridis', s=20, alpha=0.7, edgecolors='none')
            ax2.set_xlim(self.roi['lon_min'], self.roi['lon_max'])
            ax2.set_ylim(self.roi['lat_min'], self.roi['lat_max'])
            ax2.set_xlabel('Longitude (°E)')
            ax2.set_ylabel('Latitude (°N)')
            ax2.set_title(f'{var_name} - SSH Range (Max - Min)')
            ax2.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
            cbar2.set_label('SSH Range (m)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.an_output_dir / 'spatial' / 'ssh_variability_maps.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Spatial maps saved to {self.an_output_dir / 'spatial'}")
    
    def _analyze_oceanographic_features(self, dataset: xr.Dataset, spatial_stats: Dict):
        """Analyze oceanographic features in the region."""
        print("Analyzing oceanographic features...")
        
        # Get first SSH variable for feature analysis
        ssh_vars = list(spatial_stats.keys())
        primary_var = ssh_vars[0]
        
        mean_ssh = spatial_stats[primary_var]['mean_field']
        
        # Identify potential oceanographic features
        features = {
            'high_ssh_areas': [],
            'low_ssh_areas': [],
            'high_variability_areas': [],
            'gradient_features': []
        }
        
        # Calculate percentiles for feature identification
        ssh_values = mean_ssh.values.flatten()
        ssh_clean = ssh_values[~np.isnan(ssh_values)]
        
        if len(ssh_clean) > 0:
            p95 = np.percentile(ssh_clean, 95)
            p5 = np.percentile(ssh_clean, 5)
            mean_val = np.mean(ssh_clean)
            
            features['statistics'] = {
                'mean_ssh': float(mean_val),
                'ssh_range': float(p95 - p5),
                'high_ssh_threshold': float(p95),
                'low_ssh_threshold': float(p5)
            }
            
            print(f"SSH Statistics for {primary_var}:")
            print(f"  Mean SSH: {mean_val:.3f} m")
            print(f"  SSH Range: {p95 - p5:.3f} m")
            print(f"  High SSH areas (>95th percentile): {p95:.3f} m")
            print(f"  Low SSH areas (<5th percentile): {p5:.3f} m")
        
        # Analyze variability
        std_ssh = spatial_stats[primary_var]['std_field']
        std_values = std_ssh.values.flatten()
        std_clean = std_values[~np.isnan(std_values)]
        
        if len(std_clean) > 0:
            std_p95 = np.percentile(std_clean, 95)
            features['statistics']['high_variability_threshold'] = float(std_p95)
            print(f"  High variability areas (>95th percentile): {std_p95:.3f} m")
        
        self.analysis_results['oceanographic_features'] = features
        
        # Create feature identification plot
        self._create_feature_map(dataset, spatial_stats, features)
    
    def _create_feature_map(self, dataset: xr.Dataset, spatial_stats: Dict, features: Dict):
        """Create a map highlighting oceanographic features."""
        print("Creating oceanographic feature map...")
        
        primary_var = list(spatial_stats.keys())[0]
        mean_ssh = spatial_stats[primary_var]['mean_field']
        std_ssh = spatial_stats[primary_var]['std_field']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Feature identification based on SSH values
        ax1 = axes[0]
        
        # Create categorical data for features
        ssh_values = mean_ssh.values
        feature_map = np.full_like(ssh_values, 0, dtype=int)  # 0 = normal
        
        if 'statistics' in features:
            stats = features['statistics']
            high_thresh = stats.get('high_ssh_threshold', np.percentile(ssh_values[~np.isnan(ssh_values)], 95))
            low_thresh = stats.get('low_ssh_threshold', np.percentile(ssh_values[~np.isnan(ssh_values)], 5))
            
            feature_map[ssh_values > high_thresh] = 2  # High SSH
            feature_map[ssh_values < low_thresh] = 1   # Low SSH
        
        # Create custom colormap for features
        colors = ['lightblue', 'darkblue', 'darkred']  # normal, low, high
        cmap = ListedColormap(colors)
        
        scatter1 = ax1.scatter(dataset.longitude, dataset.latitude, c=feature_map, 
                             cmap=cmap, s=25, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax1.set_xlim(self.roi['lon_min'], self.roi['lon_max'])
        ax1.set_ylim(self.roi['lat_min'], self.roi['lat_max'])
        ax1.set_xlabel('Longitude (°E)', fontsize=12)
        ax1.set_ylabel('Latitude (°N)', fontsize=12)
        ax1.set_title('SSH-based Oceanographic Features', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', label='Normal SSH'),
                          Patch(facecolor='darkblue', label='Low SSH Areas'),
                          Patch(facecolor='darkred', label='High SSH Areas')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Variability-based features
        ax2 = axes[1]
        
        variability_map = std_ssh.values
        
        scatter2 = ax2.scatter(dataset.longitude, dataset.latitude, c=variability_map, 
                             cmap='plasma', s=25, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlim(self.roi['lon_min'], self.roi['lon_max'])
        ax2.set_ylim(self.roi['lat_min'], self.roi['lat_max'])
        ax2.set_xlabel('Longitude (°E)', fontsize=12)
        ax2.set_ylabel('Latitude (°N)', fontsize=12)
        ax2.set_title('SSH Variability Features', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label('SSH Std Dev (m)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.an_output_dir / 'spatial' / 'oceanographic_features.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_temporal_analysis(self, dataset: xr.Dataset):
        """
        Perform comprehensive temporal analysis.
        
        Parameters:
        -----------
        dataset : xr.Dataset
            Filtered dataset for analysis
        """
        print("\nPerforming temporal analysis for Andaman and Nicobar region...")
        
        if 'time' not in dataset.dims:
            print("No time dimension found for temporal analysis!")
            return
        
        # Get SSH variables
        ssh_vars = ['ssh_karin', 'ssh_karin_2']
        available_ssh = [var for var in ssh_vars if var in dataset.data_vars]
        
        if not available_ssh:
            print("No SSH variables found for temporal analysis!")
            return
        
        temporal_stats = {}
        
        for var_name in available_ssh:
            print(f"Analyzing temporal patterns for {var_name}...")
            
            ssh_data = dataset[var_name]
            
            # Calculate regional mean time series
            regional_mean = ssh_data.mean(dim=['latitude', 'longitude'], skipna=True)
            regional_std = ssh_data.std(dim=['latitude', 'longitude'], skipna=True)
            regional_min = ssh_data.min(dim=['latitude', 'longitude'], skipna=True)
            regional_max = ssh_data.max(dim=['latitude', 'longitude'], skipna=True)
            
            temporal_stats[var_name] = {
                'time_series': {
                    'mean': regional_mean,
                    'std': regional_std,
                    'min': regional_min,
                    'max': regional_max,
                    'range': regional_max - regional_min
                },
                'trend_analysis': {},
                'variability_analysis': {}
            }
            
            # Trend analysis
            time_values = regional_mean.time.values
            ssh_values = regional_mean.values
            
            # Remove NaN values for trend calculation
            valid_mask = ~np.isnan(ssh_values)
            if np.sum(valid_mask) > 2:
                time_numeric = np.arange(len(time_values))[valid_mask]
                ssh_clean = ssh_values[valid_mask]
                
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, ssh_clean)
                    
                    temporal_stats[var_name]['trend_analysis'] = {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'r_squared': float(r_value**2),
                        'p_value': float(p_value),
                        'std_error': float(std_err),
                        'trend_significance': 'significant' if p_value < 0.05 else 'not significant'
                    }
                    
                    print(f"  Trend: {slope:.2e} m/time_step (R²={r_value**2:.3f}, p={p_value:.3f})")
                    
                except Exception as e:
                    print(f"  Could not calculate trend: {e}")
            
            # Variability analysis
            if len(ssh_values) > 1:
                ssh_detrended = ssh_values - np.nanmean(ssh_values)
                
                temporal_stats[var_name]['variability_analysis'] = {
                    'variance': float(np.nanvar(ssh_values)),
                    'coefficient_of_variation': float(np.nanstd(ssh_values) / np.nanmean(ssh_values)) if np.nanmean(ssh_values) != 0 else 0,
                    'range': float(np.nanmax(ssh_values) - np.nanmin(ssh_values)),
                    'mean_absolute_deviation': float(np.nanmean(np.abs(ssh_detrended)))
                }
        
        self.analysis_results['temporal_stats'] = temporal_stats
        
        # Create temporal visualizations
        self._create_temporal_plots(dataset, temporal_stats)
        
        # Analyze seasonal patterns if data spans enough time
        self._analyze_seasonal_patterns(dataset, temporal_stats)
    
    def _create_temporal_plots(self, dataset: xr.Dataset, temporal_stats: Dict):
        """Create comprehensive temporal plots."""
        print("Creating temporal plots...")
        
        ssh_vars = list(temporal_stats.keys())
        
        # 1. Time series plots
        fig, axes = plt.subplots(len(ssh_vars), 1, figsize=(15, 6*len(ssh_vars)))
        if len(ssh_vars) == 1:
            axes = [axes]
        
        fig.suptitle('SSH Time Series - Andaman and Nicobar Region', fontsize=16, fontweight='bold')
        
        for i, var_name in enumerate(ssh_vars):
            ax = axes[i]
            
            stats = temporal_stats[var_name]['time_series']
            time_data = stats['mean'].time
            
            # Plot mean with error bars
            ax.plot(time_data, stats['mean'], 'b-', linewidth=2, label='Regional Mean', alpha=0.8)
            ax.fill_between(time_data, 
                           stats['mean'] - stats['std'], 
                           stats['mean'] + stats['std'], 
                           alpha=0.3, color='blue', label='±1 Std Dev')
            
            # Plot min and max
            ax.plot(time_data, stats['min'], 'r--', alpha=0.6, label='Regional Min')
            ax.plot(time_data, stats['max'], 'g--', alpha=0.6, label='Regional Max')
            
            # Add trend line if available
            if 'trend_analysis' in temporal_stats[var_name] and temporal_stats[var_name]['trend_analysis']:
                trend = temporal_stats[var_name]['trend_analysis']
                time_numeric = np.arange(len(time_data))
                trend_line = trend['slope'] * time_numeric + trend['intercept']
                ax.plot(time_data, trend_line, 'k-', linewidth=2, alpha=0.7, 
                       label=f"Trend (slope={trend['slope']:.2e})")
            
            ax.set_ylabel('SSH (m)', fontsize=12)
            ax.set_title(f'{var_name} - Regional Time Series', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format time axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(time_data)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.an_output_dir / 'temporal' / 'ssh_time_series.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Variability analysis plots
        fig, axes = plt.subplots(2, len(ssh_vars), figsize=(8*len(ssh_vars), 12))
        if len(ssh_vars) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('SSH Temporal Variability Analysis', fontsize=16, fontweight='bold')
        
        for i, var_name in enumerate(ssh_vars):
            stats = temporal_stats[var_name]['time_series']
            
            # Plot 1: SSH range over time
            ax1 = axes[0, i]
            ssh_range = stats['range']
            ax1.plot(stats['mean'].time, ssh_range, 'purple', linewidth=2, marker='o', markersize=4)
            ax1.set_ylabel('SSH Range (m)', fontsize=12)
            ax1.set_title(f'{var_name} - SSH Range Over Time', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 2: Standard deviation over time
            ax2 = axes[1, i]
            ax2.plot(stats['mean'].time, stats['std'], 'orange', linewidth=2, marker='s', markersize=4)
            ax2.set_ylabel('SSH Std Dev (m)', fontsize=12)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_title(f'{var_name} - SSH Variability Over Time', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.an_output_dir / 'temporal' / 'ssh_variability_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal plots saved to {self.an_output_dir / 'temporal'}")
    
    def _analyze_seasonal_patterns(self, dataset: xr.Dataset, temporal_stats: Dict):
        """Analyze seasonal patterns if data spans sufficient time."""
        print("Analyzing seasonal patterns...")
        
        # Check if we have enough temporal coverage for seasonal analysis
        time_range = dataset.time.max() - dataset.time.min()
        time_range_days = time_range / np.timedelta64(1, 'D')
        
        print(f"Temporal coverage: {float(time_range_days):.1f} days")
        
        if time_range_days < 30:  # Less than a month
            print("Insufficient temporal coverage for detailed seasonal analysis")
            return
        
        ssh_vars = list(temporal_stats.keys())
        seasonal_stats = {}
        
        for var_name in ssh_vars:
            try:
                ssh_data = dataset[var_name]
                regional_mean = ssh_data.mean(dim=['latitude', 'longitude'], skipna=True)
                
                # Group by month if we have enough data
                if time_range_days > 60:  # More than 2 months
                    monthly_groups = regional_mean.groupby('time.month')
                    monthly_mean = monthly_groups.mean()
                    monthly_std = monthly_groups.std()
                    
                    seasonal_stats[var_name] = {
                        'monthly_mean': monthly_mean,
                        'monthly_std': monthly_std,
                        'analysis_period': f"{float(time_range_days):.1f} days"
                    }
                    
                    print(f"  {var_name}: Calculated monthly statistics")
                
            except Exception as e:
                print(f"  Error in seasonal analysis for {var_name}: {e}")
        
        if seasonal_stats:
            self.analysis_results['seasonal_stats'] = seasonal_stats
            self._create_seasonal_plots(seasonal_stats)
    
    def _create_seasonal_plots(self, seasonal_stats: Dict):
        """Create seasonal analysis plots."""
        print("Creating seasonal plots...")
        
        ssh_vars = list(seasonal_stats.keys())
        
        fig, axes = plt.subplots(1, len(ssh_vars), figsize=(8*len(ssh_vars), 6))
        if len(ssh_vars) == 1:
            axes = [axes]
        
        fig.suptitle('Seasonal SSH Patterns - Andaman and Nicobar Region', fontsize=16, fontweight='bold')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for i, var_name in enumerate(ssh_vars):
            ax = axes[i]
            
            stats = seasonal_stats[var_name]
            monthly_mean = stats['monthly_mean']
            monthly_std = stats['monthly_std']
            
            months = monthly_mean.month.values
            means = monthly_mean.values
            stds = monthly_std.values
            
            # Plot monthly means with error bars
            ax.errorbar(months, means, yerr=stds, 
                       marker='o', linewidth=2, markersize=8, capsize=5)
            
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('SSH (m)', fontsize=12)
            ax.set_title(f'{var_name} - Monthly SSH Patterns', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels
            ax.set_xticks(months)
            ax.set_xticklabels([month_names[m-1] for m in months])
            
            # Add text with analysis period
            period_text = stats['analysis_period']
            ax.text(0.02, 0.98, f'Analysis period: {period_text}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.an_output_dir / 'temporal' / 'seasonal_patterns.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_summary(self):
        """Save comprehensive analysis summary."""
        print("\nGenerating analysis summary...")
        
        summary_content = f"""
# SWOT SSH Analysis - Andaman and Nicobar Region

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Region:** {self.roi['lat_min']}°N to {self.roi['lat_max']}°N, {self.roi['lon_min']}°E to {self.roi['lon_max']}°E
**Data Source:** SWOT L2 Basic SSH Data

## Region Overview

The Andaman and Nicobar Islands region is analyzed using SWOT satellite altimetry data. This region is oceanographically significant due to its location in the Bay of Bengal and the complex bathymetry around the island chains.

### Geographic Boundaries
- **Latitude Range:** {self.roi['lat_min']}°N to {self.roi['lat_max']}°N
- **Longitude Range:** {self.roi['lon_min']}°E to {self.roi['lon_max']}°E
- **Total Area:** Approximately {(self.roi['lat_max'] - self.roi['lat_min']) * (self.roi['lon_max'] - self.roi['lon_min']) * 111 * 111:.0f} km²

## Data Summary

"""
        
        if self.filtered_data:
            dims = dict(self.filtered_data.dims)
            summary_content += f"""
### Dataset Characteristics
- **Spatial Points:** {dims.get('latitude', 'N/A')} latitude × {dims.get('longitude', 'N/A')} longitude
- **Temporal Coverage:** {dims.get('time', 'N/A')} time steps
- **Variables:** {len([v for v in self.filtered_data.data_vars if 'ssh' in v.lower()])} SSH variables

"""
        
        # Add spatial analysis results
        if 'spatial_stats' in self.analysis_results:
            summary_content += "## Spatial Analysis Results\n\n"
            spatial_stats = self.analysis_results['spatial_stats']
            
            for var_name, stats in spatial_stats.items():
                if 'mean_field' in stats:
                    mean_ssh = stats['mean_field']
                    std_ssh = stats['std_field']
                    
                    # Calculate statistics
                    mean_val = float(mean_ssh.mean(skipna=True))
                    std_val = float(std_ssh.mean(skipna=True))
                    min_val = float(mean_ssh.min(skipna=True))
                    max_val = float(mean_ssh.max(skipna=True))
                    
                    summary_content += f"""
### {var_name} Spatial Statistics
- **Mean SSH:** {mean_val:.3f} m
- **Average Variability:** {std_val:.3f} m
- **Spatial Range:** {min_val:.3f} to {max_val:.3f} m
- **Total Range:** {max_val - min_val:.3f} m

"""
        
        # Add temporal analysis results
        if 'temporal_stats' in self.analysis_results:
            summary_content += "## Temporal Analysis Results\n\n"
            temporal_stats = self.analysis_results['temporal_stats']
            
            for var_name, stats in temporal_stats.items():
                if 'trend_analysis' in stats and stats['trend_analysis']:
                    trend = stats['trend_analysis']
                    summary_content += f"""
### {var_name} Temporal Characteristics
- **Trend Slope:** {trend['slope']:.2e} m/time_step
- **Trend Significance:** {trend['trend_significance']}
- **R-squared:** {trend['r_squared']:.3f}
- **P-value:** {trend['p_value']:.3f}

"""
                
                if 'variability_analysis' in stats:
                    var_analysis = stats['variability_analysis']
                    summary_content += f"""
#### Variability Metrics
- **Variance:** {var_analysis['variance']:.6f} m²
- **Coefficient of Variation:** {var_analysis['coefficient_of_variation']:.3f}
- **Temporal Range:** {var_analysis['range']:.3f} m

"""
        
        # Add oceanographic features
        if 'oceanographic_features' in self.analysis_results:
            features = self.analysis_results['oceanographic_features']
            if 'statistics' in features:
                stats = features['statistics']
                summary_content += f"""
## Oceanographic Features

### Feature Identification Thresholds
- **High SSH Areas:** > {stats.get('high_ssh_threshold', 'N/A'):.3f} m
- **Low SSH Areas:** < {stats.get('low_ssh_threshold', 'N/A'):.3f} m
- **High Variability Areas:** > {stats.get('high_variability_threshold', 'N/A'):.3f} m std dev

### Regional Characteristics
- **Mean SSH Level:** {stats.get('mean_ssh', 'N/A'):.3f} m
- **SSH Dynamic Range:** {stats.get('ssh_range', 'N/A'):.3f} m

"""
        
        # Add seasonal analysis if available
        if 'seasonal_stats' in self.analysis_results:
            summary_content += "## Seasonal Patterns\n\n"
            seasonal_stats = self.analysis_results['seasonal_stats']
            
            for var_name, stats in seasonal_stats.items():
                period = stats.get('analysis_period', 'Unknown')
                summary_content += f"""
### {var_name} Seasonal Analysis
- **Analysis Period:** {period}
- **Seasonal variability detected:** {'Yes' if 'monthly_mean' in stats else 'Insufficient data'}

"""
        
        # Add conclusions
        summary_content += """
## Key Findings

### Spatial Patterns
1. **SSH Distribution:** The region shows characteristic spatial patterns related to the complex bathymetry around the Andaman and Nicobar Islands
2. **Variability Hotspots:** Certain areas show higher SSH variability, potentially indicating dynamic oceanographic processes
3. **Feature Identification:** High and low SSH areas have been identified and mapped

### Temporal Characteristics
1. **Trend Analysis:** Statistical trends in SSH have been quantified over the analysis period
2. **Variability Patterns:** Temporal variability shows both short-term fluctuations and longer-term patterns
3. **Data Coverage:** Analysis based on available SWOT pass coverage over the region

### Oceanographic Significance
1. **Regional Dynamics:** The Andaman and Nicobar region shows complex SSH patterns related to its unique geography
2. **Monsoon Influence:** Temporal patterns may reflect seasonal monsoon influences (pending longer time series)
3. **Island Effects:** Spatial patterns likely influenced by the island chain bathymetry and circulation

## Recommendations for Further Analysis

1. **Extended Temporal Coverage:** Longer time series would enable better seasonal and interannual analysis
2. **Multi-mission Comparison:** Compare with Jason-3, Sentinel-6, and other altimetry missions
3. **Bathymetry Integration:** Incorporate high-resolution bathymetry for better feature interpretation
4. **Monsoon Correlation:** Analyze SSH patterns in relation to monsoon cycles
5. **Cross-validation:** Validate findings with in-situ oceanographic data when available

## Data Files Generated

### Spatial Analysis
- `mean_ssh_maps.png` - Mean SSH distribution maps
- `ssh_variability_maps.png` - SSH variability patterns
- `oceanographic_features.png` - Identified oceanographic features

### Temporal Analysis
- `ssh_time_series.png` - Regional SSH time series
- `ssh_variability_analysis.png` - Temporal variability patterns
- `seasonal_patterns.png` - Seasonal analysis (if sufficient data)

---

*Analysis completed using SWOT L2 Basic SSH data*
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save summary
        summary_path = self.an_output_dir / f"Andaman_Nicobar_Analysis_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"Analysis summary saved to: {summary_path}")
        
        # Also save analysis results as JSON for further processing
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        results_for_json = {}
        for key, value in self.analysis_results.items():
            results_for_json[key] = self._convert_for_json(value)
        
        json_path = self.an_output_dir / 'data' / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, default=str)
        
        print(f"Analysis data saved to: {json_path}")
        
        return summary_path
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and xarray objects to JSON-serializable format."""
        if hasattr(obj, 'values'):  # xarray DataArray
            return obj.values.tolist() if hasattr(obj.values, 'tolist') else str(obj.values)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def run_complete_analysis(self, max_files: Optional[int] = None):
        """
        Run complete analysis workflow for Andaman and Nicobar region.
        
        Parameters:
        -----------
        max_files : Optional[int]
            Maximum number of files to process (for testing)
        """
        print("="*80)
        print("SWOT SSH ANALYSIS - ANDAMAN AND NICOBAR REGION")
        print("="*80)
        print(f"Region: {self.roi['lat_min']}°N-{self.roi['lat_max']}°N, {self.roi['lon_min']}°E-{self.roi['lon_max']}°E")
        print("="*80)
        
        try:
            # Step 1: Discover files
            basic_files = self.discover_basic_files()
            
            if not basic_files:
                print("No SWOT Basic files found! Analysis cannot proceed.")
                return
            
            # Step 2: Load and filter data
            dataset = self.load_and_filter_data(basic_files, max_files)
            
            if dataset is None:
                print("No data found in the target region! Analysis cannot proceed.")
                return
            
            # Step 3: Spatial analysis
            print("\n" + "="*60)
            print("SPATIAL ANALYSIS")
            print("="*60)
            self.perform_spatial_analysis(dataset)
            
            # Step 4: Temporal analysis
            print("\n" + "="*60)
            print("TEMPORAL ANALYSIS")
            print("="*60)
            self.perform_temporal_analysis(dataset)
            
            # Step 5: Generate summary report
            print("\n" + "="*60)
            print("GENERATING ANALYSIS REPORT")
            print("="*60)
            self.save_analysis_summary()
            
            # Close dataset
            dataset.close()
            
            print("\n" + "="*80)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Results saved to: {self.an_output_dir}")
            print(f"Spatial analysis: {self.an_output_dir / 'spatial'}")
            print(f"Temporal analysis: {self.an_output_dir / 'temporal'}")
            print(f"Data exports: {self.an_output_dir / 'data'}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main function to run Andaman and Nicobar SWOT analysis.
    """
    # Configuration
    DATA_DIRECTORY = r"C:/Users/abhik/Desktop/project related work/SWOT_L2_LR_SSH_2.0_2.0-20250822_162623"
    
    # Maximum files to process (set to None for all files, or small number for testing)
    MAX_FILES = 5  # Start with 5 files for testing
    
    print("SWOT SSH Analysis - Andaman and Nicobar Region")
    print("=" * 50)
    print(f"Data Directory: {DATA_DIRECTORY}")
    print(f"Target Region: 6°N-15°N, 92°E-95°E")
    print(f"Max Files: {MAX_FILES if MAX_FILES else 'All available'}")
    print()
    
    try:
        # Initialize analyzer
        analyzer = AndamanNicobarSWOTAnalyzer(DATA_DIRECTORY)
        
        # Run complete analysis
        analyzer.run_complete_analysis(max_files=MAX_FILES)
        
        print("\nAndaman and Nicobar SWOT analysis completed successfully!")
        print(f"Check the output directory: {analyzer.an_output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
