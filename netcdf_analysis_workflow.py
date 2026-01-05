#!/usr/bin/env python3
"""
NetCDF Data Analysis Workflow
============================

Complete workflow for analyzing NetCDF (.nc) files:
1. File structure examination
2. Data investigation
3. Longitude-latitude gridding
4. Time scaling analysis
5. Data visualization

Author: AI Assistant
Date: August 27, 2025
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class NetCDFAnalysisWorkflow:
    """
    Complete workflow for NetCDF file analysis
    """
    
    def __init__(self, file_path):
        """Initialize with NetCDF file path"""
        self.file_path = Path(file_path)
        self.dataset = None
        self.gridded_data = {}
        self.time_info = {}
        self.analysis_results = {}
        
        # Create output directory
        self.output_dir = self.file_path.parent / f"analysis_{self.file_path.stem}"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔍 Initializing analysis for: {self.file_path.name}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def step_1_examine_structure(self):
        """TODO 1: Examine NetCDF file structure"""
        print("\n" + "="*60)
        print("📋 STEP 1: EXAMINING FILE STRUCTURE")
        print("="*60)
        
        try:
            # Open the dataset
            self.dataset = xr.open_dataset(self.file_path)
            print(f"✅ Successfully opened: {self.file_path.name}")
            
            # Basic file information
            file_size_mb = self.file_path.stat().st_size / (1024*1024)
            print(f"📊 File size: {file_size_mb:.2f} MB")
            
            # Dimensions
            print(f"\n📐 DIMENSIONS ({len(self.dataset.dims)}):")
            for dim_name, dim_size in self.dataset.dims.items():
                print(f"   • {dim_name}: {dim_size:,}")
            
            # Coordinates
            print(f"\n📍 COORDINATES ({len(self.dataset.coords)}):")
            for coord_name, coord_var in self.dataset.coords.items():
                shape = coord_var.shape
                dtype = coord_var.dtype
                print(f"   • {coord_name}: {shape} ({dtype})")
            
            # Data variables
            print(f"\n📊 DATA VARIABLES ({len(self.dataset.data_vars)}):")
            for var_name, var_data in self.dataset.data_vars.items():
                shape = var_data.shape
                dtype = var_data.dtype
                dims = var_data.dims
                print(f"   • {var_name}: {shape} {dims} ({dtype})")
            
            # Global attributes
            print(f"\n🏷️ GLOBAL ATTRIBUTES ({len(self.dataset.attrs)}):")
            for attr_name, attr_value in list(self.dataset.attrs.items())[:10]:  # Show first 10
                if isinstance(attr_value, str) and len(attr_value) > 100:
                    attr_display = attr_value[:97] + "..."
                else:
                    attr_display = attr_value
                print(f"   • {attr_name}: {attr_display}")
            
            if len(self.dataset.attrs) > 10:
                print(f"   ... and {len(self.dataset.attrs) - 10} more attributes")
            
            # Save structure summary
            structure_info = {
                'file_name': self.file_path.name,
                'file_size_mb': file_size_mb,
                'dimensions': dict(self.dataset.dims),
                'coordinates': list(self.dataset.coords.keys()),
                'data_variables': list(self.dataset.data_vars.keys()),
                'global_attributes': dict(self.dataset.attrs)
            }
            
            structure_file = self.output_dir / "01_file_structure.txt"
            with open(structure_file, 'w') as f:
                f.write("NetCDF File Structure Analysis\n")
                f.write("="*40 + "\n\n")
                f.write(f"File: {structure_info['file_name']}\n")
                f.write(f"Size: {structure_info['file_size_mb']:.2f} MB\n\n")
                
                f.write("Dimensions:\n")
                for dim, size in structure_info['dimensions'].items():
                    f.write(f"  {dim}: {size}\n")
                
                f.write("\nData Variables:\n")
                for var in structure_info['data_variables']:
                    f.write(f"  {var}\n")
                
                f.write("\nCoordinates:\n")
                for coord in structure_info['coordinates']:
                    f.write(f"  {coord}\n")
            
            print(f"\n💾 Structure summary saved: {structure_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error examining structure: {e}")
            return False
    
    def step_2_investigate_data(self):
        """TODO 2: Investigate available data variables"""
        print("\n" + "="*60)
        print("🔍 STEP 2: INVESTIGATING DATA VARIABLES")
        print("="*60)
        
        if self.dataset is None:
            print("❌ No dataset loaded. Run step 1 first.")
            return False
        
        data_summary = {}
        
        for var_name, var_data in self.dataset.data_vars.items():
            print(f"\n📊 Analyzing variable: {var_name}")
            print(f"   Shape: {var_data.shape}")
            print(f"   Dimensions: {var_data.dims}")
            print(f"   Data type: {var_data.dtype}")
            
            # Variable attributes
            if var_data.attrs:
                print("   Attributes:")
                for attr_name, attr_value in var_data.attrs.items():
                    print(f"     - {attr_name}: {attr_value}")
            
            # Statistics for numeric data
            if np.issubdtype(var_data.dtype, np.number):
                try:
                    # Sample data for statistics (to avoid memory issues)
                    sample_size = min(10000, var_data.size)
                    flat_data = var_data.values.flatten()
                    
                    if len(flat_data) > sample_size:
                        sample_indices = np.random.choice(len(flat_data), sample_size, replace=False)
                        sample_data = flat_data[sample_indices]
                    else:
                        sample_data = flat_data
                    
                    # Remove NaN values
                    valid_data = sample_data[~np.isnan(sample_data)]
                    
                    if len(valid_data) > 0:
                        stats = {
                            'count': len(valid_data),
                            'total_points': var_data.size,
                            'valid_percentage': (len(valid_data) / len(sample_data)) * 100,
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'mean': float(np.mean(valid_data)),
                            'std': float(np.std(valid_data)),
                            'median': float(np.median(valid_data))
                        }
                        
                        print(f"   Statistics (sample of {sample_size:,}):")
                        print(f"     - Valid data: {stats['valid_percentage']:.1f}%")
                        print(f"     - Range: {stats['min']:.6f} to {stats['max']:.6f}")
                        print(f"     - Mean: {stats['mean']:.6f}")
                        print(f"     - Std: {stats['std']:.6f}")
                        print(f"     - Median: {stats['median']:.6f}")
                        
                        data_summary[var_name] = stats
                    else:
                        print("     - No valid data found")
                        data_summary[var_name] = {'error': 'No valid data'}
                        
                except Exception as e:
                    print(f"     - Statistics error: {e}")
                    data_summary[var_name] = {'error': str(e)}
            else:
                print("     - Non-numeric data")
                data_summary[var_name] = {'type': 'non-numeric'}
        
        # Save data investigation summary
        summary_file = self.output_dir / "02_data_investigation.txt"
        with open(summary_file, 'w') as f:
            f.write("Data Variables Investigation\n")
            f.write("="*40 + "\n\n")
            
            for var_name, var_data in self.dataset.data_vars.items():
                f.write(f"Variable: {var_name}\n")
                f.write(f"  Shape: {var_data.shape}\n")
                f.write(f"  Dimensions: {var_data.dims}\n")
                f.write(f"  Data type: {var_data.dtype}\n")
                
                if var_name in data_summary and 'min' in data_summary[var_name]:
                    stats = data_summary[var_name]
                    f.write(f"  Statistics:\n")
                    f.write(f"    Valid data: {stats['valid_percentage']:.1f}%\n")
                    f.write(f"    Range: {stats['min']:.6f} to {stats['max']:.6f}\n")
                    f.write(f"    Mean: {stats['mean']:.6f}\n")
                    f.write(f"    Std: {stats['std']:.6f}\n")
                
                f.write("\n")
        
        self.analysis_results['data_summary'] = data_summary
        print(f"\n💾 Data investigation saved: {summary_file}")
        return True
    
    def step_3_analyze_coordinates(self):
        """TODO 3: Analyze coordinate systems"""
        print("\n" + "="*60)
        print("📍 STEP 3: ANALYZING COORDINATE SYSTEMS")
        print("="*60)
        
        if self.dataset is None:
            print("❌ No dataset loaded. Run step 1 first.")
            return False
        
        coord_analysis = {}
        
        # Look for longitude and latitude coordinates
        lon_vars = [var for var in self.dataset.variables if 'lon' in var.lower()]
        lat_vars = [var for var in self.dataset.variables if 'lat' in var.lower()]
        
        print(f"🌍 Found longitude variables: {lon_vars}")
        print(f"🌍 Found latitude variables: {lat_vars}")
        
        # Analyze each coordinate
        for coord_name, coord_var in self.dataset.coords.items():
            print(f"\n📐 Coordinate: {coord_name}")
            
            if np.issubdtype(coord_var.dtype, np.number):
                coord_data = coord_var.values
                valid_data = coord_data[~np.isnan(coord_data)] if coord_data.dtype.kind == 'f' else coord_data
                
                if len(valid_data) > 0:
                    coord_stats = {
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'shape': coord_var.shape,
                        'dims': coord_var.dims
                    }
                    
                    print(f"   Range: {coord_stats['min']:.6f} to {coord_stats['max']:.6f}")
                    print(f"   Mean: {coord_stats['mean']:.6f}")
                    print(f"   Shape: {coord_stats['shape']}")
                    
                    # Check if regular grid
                    if len(valid_data) > 1:
                        if len(coord_var.shape) == 1:
                            diffs = np.diff(valid_data)
                            if len(diffs) > 0:
                                diff_std = np.std(diffs)
                                is_regular = diff_std < (np.mean(np.abs(diffs)) * 0.01)
                                print(f"   Regular grid: {'Yes' if is_regular else 'No'}")
                                coord_stats['is_regular'] = is_regular
                                coord_stats['resolution'] = float(np.mean(diffs)) if is_regular else None
                    
                    coord_analysis[coord_name] = coord_stats
        
        # Special analysis for longitude/latitude
        if lon_vars and lat_vars:
            print(f"\n🗺️ GEOGRAPHIC COORDINATE ANALYSIS:")
            
            # Use the first longitude and latitude variables
            lon_var = lon_vars[0]
            lat_var = lat_vars[0]
            
            lon_data = self.dataset[lon_var].values
            lat_data = self.dataset[lat_var].values
            
            # Check coordinate system
            lon_range = np.max(lon_data) - np.min(lon_data)
            lat_range = np.max(lat_data) - np.min(lat_data)
            
            print(f"   Longitude range: {np.min(lon_data):.3f}° to {np.max(lon_data):.3f}° (span: {lon_range:.3f}°)")
            print(f"   Latitude range: {np.min(lat_data):.3f}° to {np.max(lat_data):.3f}° (span: {lat_range:.3f}°)")
            
            # Determine if global or regional
            is_global = lon_range > 300 and lat_range > 150
            print(f"   Coverage: {'Global' if is_global else 'Regional'}")
            
            # Store for later use
            self.analysis_results['coordinates'] = {
                'longitude_var': lon_var,
                'latitude_var': lat_var,
                'lon_range': [float(np.min(lon_data)), float(np.max(lon_data))],
                'lat_range': [float(np.min(lat_data)), float(np.max(lat_data))],
                'is_global': is_global
            }
        
        # Save coordinate analysis
        coord_file = self.output_dir / "03_coordinate_analysis.txt"
        with open(coord_file, 'w') as f:
            f.write("Coordinate System Analysis\n")
            f.write("="*40 + "\n\n")
            
            for coord_name, stats in coord_analysis.items():
                f.write(f"Coordinate: {coord_name}\n")
                f.write(f"  Range: {stats['min']:.6f} to {stats['max']:.6f}\n")
                f.write(f"  Mean: {stats['mean']:.6f}\n")
                f.write(f"  Shape: {stats['shape']}\n")
                if 'is_regular' in stats:
                    f.write(f"  Regular grid: {stats['is_regular']}\n")
                    if stats['resolution']:
                        f.write(f"  Resolution: {stats['resolution']:.6f}\n")
                f.write("\n")
        
        print(f"\n💾 Coordinate analysis saved: {coord_file}")
        return True
    
    def step_4_create_grid(self, target_resolution=0.01):
        """TODO 4: Create proper longitude-latitude grid"""
        print("\n" + "="*60)
        print("🗺️ STEP 4: CREATING LONGITUDE-LATITUDE GRID")
        print("="*60)
        
        if self.dataset is None:
            print("❌ No dataset loaded. Run previous steps first.")
            return False
        
        if 'coordinates' not in self.analysis_results:
            print("❌ Coordinate analysis not found. Run step 3 first.")
            return False
        
        coord_info = self.analysis_results['coordinates']
        lon_var = coord_info['longitude_var']
        lat_var = coord_info['latitude_var']
        
        print(f"📍 Using coordinates: {lon_var}, {lat_var}")
        print(f"🎯 Target resolution: {target_resolution}°")
        
        # Get coordinate data
        lon_data = self.dataset[lon_var].values.flatten()
        lat_data = self.dataset[lat_var].values.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(lon_data) | np.isnan(lat_data))
        lon_clean = lon_data[valid_mask]
        lat_clean = lat_data[valid_mask]
        
        print(f"📊 Valid coordinate pairs: {len(lon_clean):,}")
        
        # Create regular grid
        lon_min, lon_max = coord_info['lon_range']
        lat_min, lat_max = coord_info['lat_range']
        
        # Add small buffer
        lon_buffer = (lon_max - lon_min) * 0.01
        lat_buffer = (lat_max - lat_min) * 0.01
        
        lon_grid = np.arange(lon_min - lon_buffer, lon_max + lon_buffer + target_resolution, target_resolution)
        lat_grid = np.arange(lat_min - lat_buffer, lat_max + lat_buffer + target_resolution, target_resolution)
        
        # Create 2D grid
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        print(f"📐 Grid dimensions: {len(lon_grid)} × {len(lat_grid)} = {len(lon_grid) * len(lat_grid):,} points")
        print(f"🌍 Grid coverage:")
        print(f"   Longitude: {lon_grid[0]:.3f}° to {lon_grid[-1]:.3f}°")
        print(f"   Latitude: {lat_grid[0]:.3f}° to {lat_grid[-1]:.3f}°")
        
        # Store grid information
        self.analysis_results['grid'] = {
            'lon_grid': lon_grid,
            'lat_grid': lat_grid,
            'lon_2d': lon_2d,
            'lat_2d': lat_2d,
            'resolution': target_resolution,
            'original_coords': (lon_clean, lat_clean)
        }
        
        # Grid each data variable
        print(f"\n🔄 Gridding data variables...")
        
        for var_name, var_data in self.dataset.data_vars.items():
            if not np.issubdtype(var_data.dtype, np.number):
                continue
                
            print(f"   Gridding: {var_name}")
            
            try:
                # Get variable data
                var_values = var_data.values.flatten()
                valid_var_mask = valid_mask & ~np.isnan(var_values)
                
                if np.sum(valid_var_mask) < 10:
                    print(f"     ⚠️ Insufficient valid data for {var_name}")
                    continue
                
                var_clean = var_values[valid_var_mask]
                lon_var_clean = lon_data[valid_var_mask]
                lat_var_clean = lat_data[valid_var_mask]
                
                # Interpolate to grid using nearest neighbor (fast and robust)
                gridded_values = griddata(
                    (lon_var_clean, lat_var_clean),
                    var_clean,
                    (lon_2d, lat_2d),
                    method='nearest',
                    fill_value=np.nan
                )
                
                self.gridded_data[var_name] = gridded_values
                print(f"     ✅ Gridded {var_name}: {np.sum(~np.isnan(gridded_values)):,} valid grid points")
                
            except Exception as e:
                print(f"     ❌ Error gridding {var_name}: {e}")
        
        # Save gridding summary
        grid_file = self.output_dir / "04_gridding_summary.txt"
        with open(grid_file, 'w') as f:
            f.write("Gridding Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Target resolution: {target_resolution}°\n")
            f.write(f"Grid dimensions: {len(lon_grid)} × {len(lat_grid)}\n")
            f.write(f"Grid coverage:\n")
            f.write(f"  Longitude: {lon_grid[0]:.3f}° to {lon_grid[-1]:.3f}°\n")
            f.write(f"  Latitude: {lat_grid[0]:.3f}° to {lat_grid[-1]:.3f}°\n")
            f.write(f"\nGridded variables:\n")
            for var_name in self.gridded_data.keys():
                valid_points = np.sum(~np.isnan(self.gridded_data[var_name]))
                f.write(f"  {var_name}: {valid_points:,} valid points\n")
        
        print(f"\n💾 Gridding summary saved: {grid_file}")
        return True
    
    def step_5_time_scaling(self):
        """TODO 5: Implement time scaling analysis"""
        print("\n" + "="*60)
        print("⏰ STEP 5: TIME SCALING ANALYSIS")
        print("="*60)
        
        if self.dataset is None:
            print("❌ No dataset loaded. Run previous steps first.")
            return False
        
        # Look for time variables
        time_vars = [var for var in self.dataset.variables if 'time' in var.lower()]
        print(f"🕒 Found time variables: {time_vars}")
        
        if not time_vars:
            print("⚠️ No time variables found in dataset")
            return True
        
        # Analyze each time variable
        for time_var in time_vars:
            print(f"\n📅 Analyzing time variable: {time_var}")
            
            time_data = self.dataset[time_var]
            print(f"   Shape: {time_data.shape}")
            print(f"   Data type: {time_data.dtype}")
            
            # Get time attributes
            if hasattr(time_data, 'attrs'):
                print("   Attributes:")
                for attr_name, attr_value in time_data.attrs.items():
                    print(f"     - {attr_name}: {attr_value}")
            
            # Try to convert to datetime
            try:
                if 'datetime64' in str(time_data.dtype):
                    # Already datetime
                    time_datetime = pd.to_datetime(time_data.values)
                else:
                    # Try to decode using xarray
                    time_datetime = pd.to_datetime(time_data.values)
                
                if len(time_datetime) > 0:
                    time_min = time_datetime.min()
                    time_max = time_datetime.max()
                    time_span = time_max - time_min
                    
                    print(f"   Time range: {time_min} to {time_max}")
                    print(f"   Time span: {time_span}")
                    
                    # Calculate sampling frequency
                    if len(time_datetime) > 1:
                        time_diffs = np.diff(time_datetime)
                        avg_interval = np.mean(time_diffs)
                        print(f"   Average interval: {avg_interval}")
                        
                        # Estimate frequency
                        freq_seconds = avg_interval.total_seconds()
                        if freq_seconds < 60:
                            freq_str = f"{freq_seconds:.1f} seconds"
                        elif freq_seconds < 3600:
                            freq_str = f"{freq_seconds/60:.1f} minutes"
                        elif freq_seconds < 86400:
                            freq_str = f"{freq_seconds/3600:.1f} hours"
                        else:
                            freq_str = f"{freq_seconds/86400:.1f} days"
                        
                        print(f"   Sampling frequency: ~{freq_str}")
                    
                    # Store time information
                    self.time_info[time_var] = {
                        'datetime_values': time_datetime,
                        'time_range': [time_min, time_max],
                        'time_span': time_span,
                        'sample_count': len(time_datetime),
                        'avg_interval': avg_interval if len(time_datetime) > 1 else None
                    }
                
            except Exception as e:
                print(f"   ❌ Error processing time data: {e}")
        
        # Save time analysis
        time_file = self.output_dir / "05_time_analysis.txt"
        with open(time_file, 'w') as f:
            f.write("Time Scaling Analysis\n")
            f.write("="*40 + "\n\n")
            
            for time_var, info in self.time_info.items():
                f.write(f"Time variable: {time_var}\n")
                f.write(f"  Time range: {info['time_range'][0]} to {info['time_range'][1]}\n")
                f.write(f"  Time span: {info['time_span']}\n")
                f.write(f"  Sample count: {info['sample_count']}\n")
                if info['avg_interval']:
                    f.write(f"  Average interval: {info['avg_interval']}\n")
                f.write("\n")
        
        print(f"\n💾 Time analysis saved: {time_file}")
        return True
    
    def step_6_design_visualization(self):
        """TODO 6: Design data visualization strategy"""
        print("\n" + "="*60)
        print("🎨 STEP 6: DESIGNING VISUALIZATION STRATEGY")
        print("="*60)
        
        # Analyze what types of plots are appropriate
        plot_strategy = {}
        
        print("📊 Analyzing visualization requirements...")
        
        # For each variable, determine best plot type
        for var_name, var_data in self.dataset.data_vars.items():
            if not np.issubdtype(var_data.dtype, np.number):
                continue
            
            var_shape = var_data.shape
            var_dims = var_data.dims
            
            strategy = {
                'shape': var_shape,
                'dimensions': var_dims,
                'plot_types': []
            }
            
            # Determine plot types based on dimensions
            if len(var_shape) == 1:
                strategy['plot_types'].extend(['line_plot', 'histogram'])
                if any('time' in str(dim).lower() for dim in var_dims):
                    strategy['plot_types'].append('time_series')
            
            elif len(var_shape) == 2:
                strategy['plot_types'].extend(['heatmap', 'contour'])
                if var_name in self.gridded_data:
                    strategy['plot_types'].extend(['geographic_map', 'filled_contour'])
            
            elif len(var_shape) >= 3:
                strategy['plot_types'].extend(['slice_plots', 'animated_sequence'])
                if any('time' in str(dim).lower() for dim in var_dims):
                    strategy['plot_types'].append('temporal_animation')
            
            # Special handling for geographic data
            if 'coordinates' in self.analysis_results:
                coord_info = self.analysis_results['coordinates']
                if (coord_info['longitude_var'] in str(var_dims) or 
                    coord_info['latitude_var'] in str(var_dims)):
                    strategy['plot_types'].append('scatter_map')
            
            plot_strategy[var_name] = strategy
            
            print(f"   📈 {var_name}: {strategy['plot_types']}")
        
        # Save strategy
        strategy_file = self.output_dir / "06_visualization_strategy.txt"
        with open(strategy_file, 'w') as f:
            f.write("Visualization Strategy\n")
            f.write("="*40 + "\n\n")
            
            for var_name, strategy in plot_strategy.items():
                f.write(f"Variable: {var_name}\n")
                f.write(f"  Shape: {strategy['shape']}\n")
                f.write(f"  Dimensions: {strategy['dimensions']}\n")
                f.write(f"  Recommended plots: {', '.join(strategy['plot_types'])}\n")
                f.write("\n")
        
        self.analysis_results['plot_strategy'] = plot_strategy
        print(f"\n💾 Visualization strategy saved: {strategy_file}")
        return True
    
    def step_7_create_basic_plots(self):
        """TODO 7: Create basic data plots"""
        print("\n" + "="*60)
        print("📊 STEP 7: CREATING BASIC PLOTS")
        print("="*60)
        
        if not hasattr(self, 'analysis_results') or 'plot_strategy' not in self.analysis_results:
            print("❌ Visualization strategy not found. Run step 6 first.")
            return False
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_count = 0
        
        for var_name, var_data in self.dataset.data_vars.items():
            if not np.issubdtype(var_data.dtype, np.number):
                continue
            
            print(f"\n📈 Creating plots for: {var_name}")
            
            try:
                # Get sample data
                var_values = var_data.values
                
                # 1D plots
                if len(var_data.shape) == 1:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Line plot
                    valid_data = var_values[~np.isnan(var_values)]
                    if len(valid_data) > 0:
                        ax1.plot(valid_data, 'b-', linewidth=1)
                        ax1.set_title(f'{var_name} - Line Plot')
                        ax1.set_xlabel('Index')
                        ax1.set_ylabel(var_name)
                        ax1.grid(True, alpha=0.3)
                        
                        # Histogram
                        ax2.hist(valid_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
                        ax2.set_title(f'{var_name} - Distribution')
                        ax2.set_xlabel(var_name)
                        ax2.set_ylabel('Frequency')
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plot_file = plots_dir / f"basic_{var_name}_1d.png"
                        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"   ✅ 1D plots saved: {plot_file}")
                        plot_count += 1
                
                # 2D plots
                elif len(var_data.shape) == 2:
                    # Sample for performance
                    max_size = 500
                    h, w = var_values.shape
                    h_step = max(1, h // max_size)
                    w_step = max(1, w // max_size)
                    sample_data = var_values[::h_step, ::w_step]
                    
                    if sample_data.size > 0:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Heatmap
                        im1 = ax1.imshow(sample_data, aspect='auto', cmap='viridis')
                        ax1.set_title(f'{var_name} - Heatmap')
                        plt.colorbar(im1, ax=ax1, label=var_name)
                        
                        # Histogram of 2D data
                        flat_data = sample_data.flatten()
                        valid_flat = flat_data[~np.isnan(flat_data)]
                        if len(valid_flat) > 0:
                            ax2.hist(valid_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
                            ax2.set_title(f'{var_name} - Distribution')
                            ax2.set_xlabel(var_name)
                            ax2.set_ylabel('Frequency')
                            ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plot_file = plots_dir / f"basic_{var_name}_2d.png"
                        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"   ✅ 2D plots saved: {plot_file}")
                        plot_count += 1
                
                # Gridded data plots
                if var_name in self.gridded_data:
                    gridded_values = self.gridded_data[var_name]
                    
                    if np.sum(~np.isnan(gridded_values)) > 0:
                        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                        
                        grid_info = self.analysis_results['grid']
                        lon_2d = grid_info['lon_2d']
                        lat_2d = grid_info['lat_2d']
                        
                        # Create geographic plot
                        im = ax.contourf(lon_2d, lat_2d, gridded_values, levels=50, cmap='viridis')
                        ax.set_xlabel('Longitude (°)')
                        ax.set_ylabel('Latitude (°)')
                        ax.set_title(f'{var_name} - Gridded Data')
                        plt.colorbar(im, ax=ax, label=var_name)
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plot_file = plots_dir / f"gridded_{var_name}.png"
                        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"   ✅ Gridded plot saved: {plot_file}")
                        plot_count += 1
                        
            except Exception as e:
                print(f"   ❌ Error creating plots for {var_name}: {e}")
        
        print(f"\n📊 Created {plot_count} plot files in: {plots_dir}")
        return True
    
    def step_8_advanced_plotting(self):
        """TODO 8: Develop advanced plotting routines"""
        print("\n" + "="*60)
        print("🎯 STEP 8: ADVANCED PLOTTING ROUTINES")
        print("="*60)
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create multi-panel summary plot
        print("📊 Creating multi-panel summary...")
        
        # Get key variables for summary
        summary_vars = []
        for var_name, var_data in self.dataset.data_vars.items():
            if (np.issubdtype(var_data.dtype, np.number) and 
                var_data.size > 1 and 
                len(summary_vars) < 4):
                summary_vars.append(var_name)
        
        if summary_vars:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, var_name in enumerate(summary_vars):
                if i >= 4:
                    break
                    
                ax = axes[i]
                var_data = self.dataset[var_name].values
                
                if len(var_data.shape) == 1:
                    # Line plot
                    valid_data = var_data[~np.isnan(var_data)]
                    ax.plot(valid_data, 'b-', linewidth=1)
                    ax.set_title(f'{var_name}')
                    ax.grid(True, alpha=0.3)
                    
                elif len(var_data.shape) == 2:
                    # Heatmap (sampled)
                    max_size = 200
                    h, w = var_data.shape
                    h_step = max(1, h // max_size)
                    w_step = max(1, w // max_size)
                    sample_data = var_data[::h_step, ::w_step]
                    
                    im = ax.imshow(sample_data, aspect='auto', cmap='viridis')
                    ax.set_title(f'{var_name}')
                    plt.colorbar(im, ax=ax, shrink=0.6)
                
                else:
                    # Flatten and histogram
                    flat_data = var_data.flatten()
                    valid_data = flat_data[~np.isnan(flat_data)]
                    if len(valid_data) > 0:
                        ax.hist(valid_data, bins=30, alpha=0.7, color='blue')
                        ax.set_title(f'{var_name} - Distribution')
                        ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(summary_vars), 4):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Data Summary: {self.file_path.name}', fontsize=16)
            plt.tight_layout()
            
            summary_file = plots_dir / "summary_overview.png"
            plt.savefig(summary_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Summary plot saved: {summary_file}")
        
        # Create geographic overview if coordinates available
        if ('coordinates' in self.analysis_results and 
            'grid' in self.analysis_results and 
            self.gridded_data):
            
            print("🗺️ Creating geographic overview...")
            
            coord_info = self.analysis_results['coordinates']
            grid_info = self.analysis_results['grid']
            
            # Get the first gridded variable
            first_var = list(self.gridded_data.keys())[0]
            gridded_values = self.gridded_data[first_var]
            
            fig = plt.figure(figsize=(15, 10))
            
            # Use cartopy if available
            try:
                ax = plt.axes(projection=ccrs.PlateCarree())
                
                # Plot data
                lon_2d = grid_info['lon_2d']
                lat_2d = grid_info['lat_2d']
                
                im = ax.contourf(lon_2d, lat_2d, gridded_values, 
                               levels=50, cmap='viridis', transform=ccrs.PlateCarree())
                
                # Add map features
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.gridlines(draw_labels=True, alpha=0.5)
                
                plt.colorbar(im, ax=ax, shrink=0.6, label=first_var)
                plt.title(f'Geographic View: {first_var}')
                
            except ImportError:
                # Fallback without cartopy
                ax = plt.axes()
                lon_2d = grid_info['lon_2d']
                lat_2d = grid_info['lat_2d']
                
                im = ax.contourf(lon_2d, lat_2d, gridded_values, levels=50, cmap='viridis')
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Latitude (°)')
                plt.colorbar(im, ax=ax, label=first_var)
                plt.title(f'Geographic View: {first_var}')
                ax.grid(True, alpha=0.3)
            
            geo_file = plots_dir / "geographic_overview.png"
            plt.savefig(geo_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Geographic plot saved: {geo_file}")
        
        # Create correlation matrix if multiple variables
        numeric_vars = [var for var in self.dataset.data_vars 
                       if np.issubdtype(self.dataset[var].dtype, np.number)]
        
        if len(numeric_vars) > 1:
            print("📈 Creating correlation analysis...")
            
            try:
                # Sample data for correlation
                corr_data = {}
                for var_name in numeric_vars[:10]:  # Limit to 10 variables
                    var_values = self.dataset[var_name].values.flatten()
                    valid_mask = ~np.isnan(var_values)
                    if np.sum(valid_mask) > 100:  # Need sufficient data
                        sample_size = min(10000, np.sum(valid_mask))
                        sample_indices = np.random.choice(np.where(valid_mask)[0], 
                                                        sample_size, replace=False)
                        corr_data[var_name] = var_values[sample_indices]
                
                if len(corr_data) > 1:
                    df = pd.DataFrame(corr_data)
                    corr_matrix = df.corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                              square=True, ax=ax, cbar_kws={'label': 'Correlation'})
                    plt.title('Variable Correlation Matrix')
                    plt.tight_layout()
                    
                    corr_file = plots_dir / "correlation_matrix.png"
                    plt.savefig(corr_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"   ✅ Correlation plot saved: {corr_file}")
                    
            except Exception as e:
                print(f"   ⚠️ Correlation analysis error: {e}")
        
        print(f"🎯 Advanced plotting completed")
        return True
    
    def step_9_export_results(self):
        """TODO 9: Export and save results"""
        print("\n" + "="*60)
        print("💾 STEP 9: EXPORTING AND SAVING RESULTS")
        print("="*60)
        
        # Export gridded data to NetCDF
        if self.gridded_data and 'grid' in self.analysis_results:
            print("📤 Exporting gridded data to NetCDF...")
            
            grid_info = self.analysis_results['grid']
            
            # Create new dataset with gridded data
            export_data = {}
            
            # Add coordinates
            export_data['longitude'] = (['x'], grid_info['lon_grid'])
            export_data['latitude'] = (['y'], grid_info['lat_grid'])
            
            # Add gridded variables
            for var_name, gridded_values in self.gridded_data.items():
                export_data[var_name] = (['y', 'x'], gridded_values)
            
            # Create dataset
            export_ds = xr.Dataset(export_data)
            
            # Add attributes
            export_ds.attrs['title'] = f'Gridded data from {self.file_path.name}'
            export_ds.attrs['source'] = str(self.file_path)
            export_ds.attrs['grid_resolution'] = grid_info['resolution']
            export_ds.attrs['created'] = datetime.now().isoformat()
            
            # Save to file
            gridded_nc_file = self.output_dir / f"gridded_{self.file_path.stem}.nc"
            export_ds.to_netcdf(gridded_nc_file)
            print(f"   ✅ Gridded NetCDF saved: {gridded_nc_file}")
        
        # Export data summaries to CSV
        if 'data_summary' in self.analysis_results:
            print("📊 Exporting data summaries...")
            
            summary_data = []
            for var_name, stats in self.analysis_results['data_summary'].items():
                if isinstance(stats, dict) and 'min' in stats:
                    summary_data.append({
                        'variable': var_name,
                        'valid_percentage': stats['valid_percentage'],
                        'min_value': stats['min'],
                        'max_value': stats['max'],
                        'mean_value': stats['mean'],
                        'std_value': stats['std'],
                        'median_value': stats['median']
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_csv = self.output_dir / "data_summary.csv"
                summary_df.to_csv(summary_csv, index=False)
                print(f"   ✅ Data summary CSV saved: {summary_csv}")
        
        # Create comprehensive report
        print("📄 Creating comprehensive report...")
        
        report_file = self.output_dir / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(f"# NetCDF Analysis Report\n\n")
            f.write(f"**File:** {self.file_path.name}  \n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Output Directory:** {self.output_dir}  \n\n")
            
            f.write("## File Structure\n\n")
            f.write(f"- **Dimensions:** {len(self.dataset.dims)}\n")
            f.write(f"- **Variables:** {len(self.dataset.data_vars)}\n")
            f.write(f"- **Coordinates:** {len(self.dataset.coords)}\n\n")
            
            if 'coordinates' in self.analysis_results:
                coord_info = self.analysis_results['coordinates']
                f.write("## Geographic Coverage\n\n")
                f.write(f"- **Longitude Range:** {coord_info['lon_range'][0]:.3f}° to {coord_info['lon_range'][1]:.3f}°\n")
                f.write(f"- **Latitude Range:** {coord_info['lat_range'][0]:.3f}° to {coord_info['lat_range'][1]:.3f}°\n")
                f.write(f"- **Coverage Type:** {'Global' if coord_info['is_global'] else 'Regional'}\n\n")
            
            if 'grid' in self.analysis_results:
                grid_info = self.analysis_results['grid']
                f.write("## Gridding Information\n\n")
                f.write(f"- **Grid Resolution:** {grid_info['resolution']}°\n")
                f.write(f"- **Grid Dimensions:** {len(grid_info['lon_grid'])} × {len(grid_info['lat_grid'])}\n")
                f.write(f"- **Gridded Variables:** {len(self.gridded_data)}\n\n")
            
            if self.time_info:
                f.write("## Temporal Information\n\n")
                for time_var, info in self.time_info.items():
                    f.write(f"**{time_var}:**\n")
                    f.write(f"- Time Range: {info['time_range'][0]} to {info['time_range'][1]}\n")
                    f.write(f"- Sample Count: {info['sample_count']:,}\n")
                    if info['avg_interval']:
                        f.write(f"- Average Interval: {info['avg_interval']}\n")
                    f.write("\n")
            
            f.write("## Generated Files\n\n")
            f.write("### Analysis Files\n")
            for analysis_file in sorted(self.output_dir.glob("0*_*.txt")):
                f.write(f"- {analysis_file.name}\n")
            
            f.write("\n### Plots\n")
            plots_dir = self.output_dir / "plots"
            if plots_dir.exists():
                for plot_file in sorted(plots_dir.glob("*.png")):
                    f.write(f"- plots/{plot_file.name}\n")
            
            f.write("\n### Data Files\n")
            for data_file in self.output_dir.glob("*.nc"):
                f.write(f"- {data_file.name}\n")
            for data_file in self.output_dir.glob("*.csv"):
                f.write(f"- {data_file.name}\n")
        
        print(f"   ✅ Analysis report saved: {report_file}")
        
        # List all created files
        print(f"\n📁 All analysis files created in: {self.output_dir}")
        all_files = list(self.output_dir.rglob("*"))
        file_count = len([f for f in all_files if f.is_file()])
        print(f"📊 Total files created: {file_count}")
        
        return True
    
    def step_10_quality_control(self):
        """TODO 10: Quality control and validation"""
        print("\n" + "="*60)
        print("✅ STEP 10: QUALITY CONTROL AND VALIDATION")
        print("="*60)
        
        validation_report = {}
        
        # Validate gridded data
        if self.gridded_data and 'grid' in self.analysis_results:
            print("🔍 Validating gridded data...")
            
            original_coords = self.analysis_results['grid']['original_coords']
            lon_orig, lat_orig = original_coords
            
            for var_name, gridded_values in self.gridded_data.items():
                print(f"   Checking {var_name}...")
                
                # Get original data
                orig_values = self.dataset[var_name].values.flatten()
                valid_mask = ~(np.isnan(lon_orig) | np.isnan(lat_orig) | np.isnan(orig_values))
                
                if np.sum(valid_mask) > 0:
                    orig_clean = orig_values[valid_mask]
                    gridded_clean = gridded_values[~np.isnan(gridded_values)]
                    
                    if len(gridded_clean) > 0:
                        # Compare statistics
                        orig_stats = {
                            'min': float(np.min(orig_clean)),
                            'max': float(np.max(orig_clean)),
                            'mean': float(np.mean(orig_clean)),
                            'std': float(np.std(orig_clean))
                        }
                        
                        grid_stats = {
                            'min': float(np.min(gridded_clean)),
                            'max': float(np.max(gridded_clean)),
                            'mean': float(np.mean(gridded_clean)),
                            'std': float(np.std(gridded_clean))
                        }
                        
                        # Calculate differences
                        diff_stats = {
                            'min_diff': abs(grid_stats['min'] - orig_stats['min']) / abs(orig_stats['min']) * 100,
                            'max_diff': abs(grid_stats['max'] - orig_stats['max']) / abs(orig_stats['max']) * 100,
                            'mean_diff': abs(grid_stats['mean'] - orig_stats['mean']) / abs(orig_stats['mean']) * 100,
                            'std_diff': abs(grid_stats['std'] - orig_stats['std']) / abs(orig_stats['std']) * 100
                        }
                        
                        validation_report[var_name] = {
                            'original_stats': orig_stats,
                            'gridded_stats': grid_stats,
                            'differences': diff_stats,
                            'data_retention': len(gridded_clean) / len(orig_clean) * 100
                        }
                        
                        print(f"     - Data retention: {validation_report[var_name]['data_retention']:.1f}%")
                        print(f"     - Mean difference: {diff_stats['mean_diff']:.1f}%")
                        
                        # Quality flags
                        quality_flags = []
                        if diff_stats['mean_diff'] > 10:
                            quality_flags.append("Large mean difference")
                        if validation_report[var_name]['data_retention'] < 50:
                            quality_flags.append("Low data retention")
                        
                        if quality_flags:
                            print(f"     ⚠️ Warnings: {', '.join(quality_flags)}")
                        else:
                            print(f"     ✅ Quality check passed")
        
        # Validate time scaling
        if self.time_info:
            print("\n🕒 Validating time scaling...")
            
            for time_var, info in self.time_info.items():
                print(f"   Checking {time_var}...")
                
                # Check for time gaps
                if info['avg_interval'] and info['sample_count'] > 1:
                    time_values = info['datetime_values']
                    time_diffs = np.diff(time_values)
                    
                    # Find large gaps (more than 2x average interval)
                    avg_interval = info['avg_interval']
                    large_gaps = time_diffs > (avg_interval * 2)
                    
                    if np.any(large_gaps):
                        gap_count = np.sum(large_gaps)
                        print(f"     ⚠️ Found {gap_count} large time gaps")
                    else:
                        print(f"     ✅ No significant time gaps")
        
        # Create validation summary
        validation_file = self.output_dir / "10_validation_report.txt"
        with open(validation_file, 'w') as f:
            f.write("Quality Control and Validation Report\n")
            f.write("="*40 + "\n\n")
            
            f.write("Gridded Data Validation:\n")
            for var_name, report in validation_report.items():
                f.write(f"\nVariable: {var_name}\n")
                f.write(f"  Data retention: {report['data_retention']:.1f}%\n")
                f.write(f"  Mean difference: {report['differences']['mean_diff']:.1f}%\n")
                f.write(f"  Range preservation:\n")
                f.write(f"    Min: {report['differences']['min_diff']:.1f}% difference\n")
                f.write(f"    Max: {report['differences']['max_diff']:.1f}% difference\n")
            
            if self.time_info:
                f.write(f"\nTime Scaling Validation:\n")
                for time_var, info in self.time_info.items():
                    f.write(f"  {time_var}: {info['sample_count']} time points\n")
        
        print(f"\n💾 Validation report saved: {validation_file}")
        
        # Final summary
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 All results saved in: {self.output_dir}")
        print(f"📊 Analysis covered {len(self.dataset.data_vars)} variables")
        if self.gridded_data:
            print(f"🗺️ Successfully gridded {len(self.gridded_data)} variables")
        
        return True
    
    def run_complete_workflow(self):
        """Run the complete analysis workflow"""
        print("🚀 Starting Complete NetCDF Analysis Workflow")
        print("="*60)
        
        workflow_steps = [
            ("Examine file structure", self.step_1_examine_structure),
            ("Investigate data variables", self.step_2_investigate_data),
            ("Analyze coordinate systems", self.step_3_analyze_coordinates),
            ("Create longitude-latitude grid", self.step_4_create_grid),
            ("Implement time scaling", self.step_5_time_scaling),
            ("Design visualization strategy", self.step_6_design_visualization),
            ("Create basic plots", self.step_7_create_basic_plots),
            ("Develop advanced plotting", self.step_8_advanced_plotting),
            ("Export and save results", self.step_9_export_results),
            ("Quality control and validation", self.step_10_quality_control)
        ]
        
        for step_name, step_function in workflow_steps:
            print(f"\n{'='*20} {step_name.upper()} {'='*20}")
            
            try:
                success = step_function()
                if success:
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"❌ {step_name} failed")
                    response = input("Continue with next step? (y/N): ").lower()
                    if response != 'y':
                        break
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                response = input("Continue with next step? (y/N): ").lower()
                if response != 'y':
                    break
        
        print(f"\n🎯 Workflow completed! Check results in: {self.output_dir}")

def main():
    """Main function with example usage"""
    print("🌊 NetCDF Analysis Workflow Tool")
    print("="*50)
    
    # Look for NetCDF files in current directory
    current_dir = Path('.')
    nc_files = list(current_dir.glob('*.nc'))
    
    if nc_files:
        print(f"\n📁 Found {len(nc_files)} NetCDF files:")
        for i, nc_file in enumerate(nc_files, 1):
            file_size = nc_file.stat().st_size / (1024*1024)
            print(f"   {i}. {nc_file.name} ({file_size:.1f} MB)")
        
        try:
            choice = input(f"\nSelect file to analyze (1-{len(nc_files)}): ").strip()
            file_index = int(choice) - 1
            
            if 0 <= file_index < len(nc_files):
                selected_file = nc_files[file_index]
                
                print(f"\n🎯 Starting analysis of: {selected_file.name}")
                
                # Create workflow instance
                workflow = NetCDFAnalysisWorkflow(selected_file)
                
                # Run complete workflow
                workflow.run_complete_workflow()
                
            else:
                print("❌ Invalid file selection")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Analysis cancelled")
    
    else:
        print("\n❌ No NetCDF files found in current directory")
        print("💡 Usage example:")
        print("   workflow = NetCDFAnalysisWorkflow('your_file.nc')")
        print("   workflow.run_complete_workflow()")

if __name__ == "__main__":
    main()
