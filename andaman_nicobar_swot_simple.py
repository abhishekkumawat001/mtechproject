#!/usr/bin/env python3
"""
Simplified SWOT SSH Analysis for Andaman and Nicobar Region
===========================================================

This script analyzes SWOT L2 Basic SSH data specifically for the Andaman and Nicobar region.
It handles the SWOT data structure properly and creates spatial and temporal visualizations.

Author: Automated Analysis System
Date: August 23, 2025
Target Region: Andaman and Nicobar Islands (6°N-15°N, 92°E-95°E)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class AndamanNicobarSWOTAnalyzer:
    """Simplified SWOT data analyzer for Andaman and Nicobar region."""
    
    def __init__(self, data_directory: str):
        self.data_dir = Path(data_directory)
        self.region = {
            'lat_min': 6.0, 'lat_max': 15.0,
            'lon_min': 92.0, 'lon_max': 95.0
        }
        
        # Create output directory
        self.output_dir = Path("swot_analysis_output") / "figures_andaman_nicobar"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Andaman Nicobar SWOT Analyzer initialized")
        print(f"Region: {self.region['lat_min']}°N-{self.region['lat_max']}°N, "
              f"{self.region['lon_min']}°E-{self.region['lon_max']}°E")
        print(f"Output directory: {self.output_dir}")
    
    def find_basic_files(self):
        """Find all Basic SSH files."""
        pattern = "*Basic*.nc"
        files = list(self.data_dir.glob(pattern))
        files.sort()
        print(f"Found {len(files)} Basic SSH files")
        return files
    
    def process_single_file(self, file_path):
        """Process a single SWOT file and extract Andaman region data."""
        try:
            # Load dataset
            ds = xr.open_dataset(file_path)
            
            # Check for required coordinates
            if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
                return None
            
            # Filter for Andaman region
            lat_mask = ((ds.latitude >= self.region['lat_min']) & 
                       (ds.latitude <= self.region['lat_max']))
            lon_mask = ((ds.longitude >= self.region['lon_min']) & 
                       (ds.longitude <= self.region['lon_max']))
            region_mask = lat_mask & lon_mask
            
            # Apply mask
            ds_region = ds.where(region_mask, drop=True)
            
            # Check if we have data
            if ds_region.latitude.size == 0:
                return None
            
            # Extract time from filename
            filename = file_path.name
            # SWOT filename format: SWOT_L2_LR_SSH_Basic_CCC_PPP_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_PGC0_01.nc
            parts = filename.split('_')
            time_str = parts[5]  # Start time: YYYYMMDDTHHMMSS
            date_str = time_str[:8]  # YYYYMMDD
            
            # Get SSH data
            ssh_vars = ['ssh_karin', 'ssh_karin_2']
            available_ssh = [var for var in ssh_vars if var in ds_region.data_vars]
            
            if not available_ssh:
                return None
            
            # Prepare output data
            result = {
                'date': date_str,
                'datetime': pd.to_datetime(time_str, format='%Y%m%dT%H%M%S'),
                'file': filename,
                'latitude': ds_region.latitude.values.flatten(),
                'longitude': ds_region.longitude.values.flatten(),
                'n_points': ds_region.latitude.size
            }
            
            # Add SSH data
            for var in available_ssh:
                result[var] = ds_region[var].values.flatten()
            
            return result
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def analyze_region_data(self, max_files=10):
        """Analyze SWOT data for Andaman and Nicobar region."""
        print(f"\nAnalyzing SWOT data for Andaman and Nicobar region...")
        
        # Find files
        files = self.find_basic_files()
        if not files:
            print("No Basic SSH files found!")
            return
        
        # Process files
        print(f"Processing up to {max_files} files...")
        all_data = []
        
        for i, file_path in enumerate(files[:max_files]):
            print(f"Processing file {i+1}/{min(len(files), max_files)}: {file_path.name}")
            
            result = self.process_single_file(file_path)
            if result:
                all_data.append(result)
                print(f"  Found {result['n_points']} data points in region")
        
        if not all_data:
            print("No valid data found in the region!")
            return
        
        print(f"\nSuccessfully processed {len(all_data)} files")
        
        # Combine data
        self.combined_data = self._combine_data(all_data)
        
        # Perform analysis
        self._spatial_analysis()
        self._temporal_analysis()
        
        print(f"\nAnalysis complete! Check results in: {self.output_dir}")
    
    def _combine_data(self, all_data):
        """Combine data from all files."""
        combined = {
            'dates': [],
            'datetimes': [],
            'latitude': [],
            'longitude': [],
            'ssh_karin': [],
            'ssh_karin_2': []
        }
        
        for data in all_data:
            n_points = len(data['latitude'])
            combined['dates'].extend([data['date']] * n_points)
            combined['datetimes'].extend([data['datetime']] * n_points)
            combined['latitude'].extend(data['latitude'])
            combined['longitude'].extend(data['longitude'])
            
            if 'ssh_karin' in data:
                combined['ssh_karin'].extend(data['ssh_karin'])
            else:
                combined['ssh_karin'].extend([np.nan] * n_points)
                
            if 'ssh_karin_2' in data:
                combined['ssh_karin_2'].extend(data['ssh_karin_2'])
            else:
                combined['ssh_karin_2'].extend([np.nan] * n_points)
        
        # Convert to numpy arrays and create DataFrame
        df = pd.DataFrame({
            'date': combined['dates'],
            'datetime': combined['datetimes'],
            'latitude': np.array(combined['latitude']),
            'longitude': np.array(combined['longitude']),
            'ssh_karin': np.array(combined['ssh_karin']),
            'ssh_karin_2': np.array(combined['ssh_karin_2'])
        })
        
        # Remove rows with all NaN SSH values
        df = df.dropna(subset=['ssh_karin', 'ssh_karin_2'], how='all')
        
        print(f"Combined dataset: {len(df)} data points")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
        print(f"Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
        
        return df
    
    def _spatial_analysis(self):
        """Perform spatial analysis."""
        print("\n" + "="*50)
        print("SPATIAL ANALYSIS")
        print("="*50)
        
        df = self.combined_data
        
        # Calculate statistics for each SSH variable
        ssh_vars = ['ssh_karin', 'ssh_karin_2']
        
        for var in ssh_vars:
            if var in df.columns:
                data = df[var].dropna()
                if len(data) > 0:
                    print(f"\n{var} Statistics:")
                    print(f"  Count: {len(data)}")
                    print(f"  Mean: {data.mean():.3f} m")
                    print(f"  Std: {data.std():.3f} m")
                    print(f"  Min: {data.min():.3f} m")
                    print(f"  Max: {data.max():.3f} m")
                    print(f"  Range: {data.max() - data.min():.3f} m")
        
        # Create spatial plots
        self._create_spatial_plots(df)
    
    def _temporal_analysis(self):
        """Perform temporal analysis."""
        print("\n" + "="*50)
        print("TEMPORAL ANALYSIS")
        print("="*50)
        
        df = self.combined_data
        
        # Group by date and calculate daily statistics
        daily_stats = df.groupby('date').agg({
            'ssh_karin': ['count', 'mean', 'std', 'min', 'max'],
            'ssh_karin_2': ['count', 'mean', 'std', 'min', 'max'],
            'latitude': ['min', 'max'],
            'longitude': ['min', 'max']
        }).round(3)
        
        print("\nDaily Statistics Summary:")
        print(daily_stats)
        
        # Save daily statistics
        daily_stats.to_csv(self.output_dir / "daily_statistics.csv")
        
        # Create temporal plots
        self._create_temporal_plots(df)
    
    def _create_spatial_plots(self, df):
        """Create spatial visualization plots."""
        print("Creating spatial plots...")
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SWOT SSH Spatial Analysis - Andaman and Nicobar Region', fontsize=16)
        
        # Plot 1: SSH Karin scatter plot
        ax1 = axes[0, 0]
        ssh1_valid = df[df['ssh_karin'].notna()]
        if len(ssh1_valid) > 0:
            scatter = ax1.scatter(ssh1_valid['longitude'], ssh1_valid['latitude'], 
                                c=ssh1_valid['ssh_karin'], cmap='RdYlBu_r', s=1, alpha=0.6)
            ax1.set_title('SSH Karin Distribution')
            ax1.set_xlabel('Longitude (°E)')
            ax1.set_ylabel('Latitude (°N)')
            plt.colorbar(scatter, ax=ax1, label='SSH (m)')
        
        # Plot 2: SSH Karin 2 scatter plot
        ax2 = axes[0, 1]
        ssh2_valid = df[df['ssh_karin_2'].notna()]
        if len(ssh2_valid) > 0:
            scatter = ax2.scatter(ssh2_valid['longitude'], ssh2_valid['latitude'], 
                                c=ssh2_valid['ssh_karin_2'], cmap='RdYlBu_r', s=1, alpha=0.6)
            ax2.set_title('SSH Karin 2 Distribution')
            ax2.set_xlabel('Longitude (°E)')
            ax2.set_ylabel('Latitude (°N)')
            plt.colorbar(scatter, ax=ax2, label='SSH (m)')
        
        # Plot 3: SSH distribution histogram
        ax3 = axes[1, 0]
        if 'ssh_karin' in df.columns:
            ssh1_data = df['ssh_karin'].dropna()
            if len(ssh1_data) > 0:
                ax3.hist(ssh1_data, bins=50, alpha=0.7, label='SSH Karin', density=True)
        if 'ssh_karin_2' in df.columns:
            ssh2_data = df['ssh_karin_2'].dropna()
            if len(ssh2_data) > 0:
                ax3.hist(ssh2_data, bins=50, alpha=0.7, label='SSH Karin 2', density=True)
        ax3.set_title('SSH Value Distribution')
        ax3.set_xlabel('SSH (m)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Data coverage map
        ax4 = axes[1, 1]
        ax4.scatter(df['longitude'], df['latitude'], s=0.5, alpha=0.5, c='blue')
        ax4.set_title('Data Coverage')
        ax4.set_xlabel('Longitude (°E)')
        ax4.set_ylabel('Latitude (°N)')
        ax4.grid(True, alpha=0.3)
        
        # Add region boundaries
        for ax in [ax1, ax2, ax4]:
            ax.axhline(y=self.region['lat_min'], color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=self.region['lat_max'], color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.region['lon_min'], color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.region['lon_max'], color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "spatial_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Spatial plot saved: {self.output_dir / 'spatial_analysis.png'}")
    
    def _create_temporal_plots(self, df):
        """Create temporal visualization plots."""
        print("Creating temporal plots...")
        
        # Group by date for temporal analysis
        daily_data = df.groupby('datetime').agg({
            'ssh_karin': ['count', 'mean', 'std'],
            'ssh_karin_2': ['count', 'mean', 'std']
        })
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SWOT SSH Temporal Analysis - Andaman and Nicobar Region', fontsize=16)
        
        # Plot 1: Time series of mean SSH
        ax1 = axes[0, 0]
        if ('ssh_karin', 'mean') in daily_data.columns:
            daily_data[('ssh_karin', 'mean')].plot(ax=ax1, label='SSH Karin', marker='o')
        if ('ssh_karin_2', 'mean') in daily_data.columns:
            daily_data[('ssh_karin_2', 'mean')].plot(ax=ax1, label='SSH Karin 2', marker='s')
        ax1.set_title('Daily Mean SSH')
        ax1.set_ylabel('SSH (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time series of SSH standard deviation
        ax2 = axes[0, 1]
        if ('ssh_karin', 'std') in daily_data.columns:
            daily_data[('ssh_karin', 'std')].plot(ax=ax2, label='SSH Karin', marker='o')
        if ('ssh_karin_2', 'std') in daily_data.columns:
            daily_data[('ssh_karin_2', 'std')].plot(ax=ax2, label='SSH Karin 2', marker='s')
        ax2.set_title('Daily SSH Standard Deviation')
        ax2.set_ylabel('SSH Std (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Data count over time
        ax3 = axes[1, 0]
        if ('ssh_karin', 'count') in daily_data.columns:
            daily_data[('ssh_karin', 'count')].plot(ax=ax3, label='SSH Karin', marker='o')
        if ('ssh_karin_2', 'count') in daily_data.columns:
            daily_data[('ssh_karin_2', 'count')].plot(ax=ax3, label='SSH Karin 2', marker='s')
        ax3.set_title('Daily Data Point Count')
        ax3.set_ylabel('Number of Points')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: SSH correlation
        ax4 = axes[1, 1]
        # Create correlation plot if both variables have data
        ssh1 = df['ssh_karin'].dropna()
        ssh2 = df['ssh_karin_2'].dropna()
        
        if len(ssh1) > 0 and len(ssh2) > 0:
            # Find common indices
            common_df = df[['ssh_karin', 'ssh_karin_2']].dropna()
            if len(common_df) > 10:
                ax4.scatter(common_df['ssh_karin'], common_df['ssh_karin_2'], 
                           alpha=0.5, s=1)
                ax4.set_xlabel('SSH Karin (m)')
                ax4.set_ylabel('SSH Karin 2 (m)')
                ax4.set_title('SSH Karin vs SSH Karin 2')
                
                # Calculate correlation
                correlation = common_df['ssh_karin'].corr(common_df['ssh_karin_2'])
                ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Temporal plot saved: {self.output_dir / 'temporal_analysis.png'}")
    
    def save_summary_report(self):
        """Save a summary report of the analysis."""
        if not hasattr(self, 'combined_data'):
            print("No data to summarize!")
            return
        
        df = self.combined_data
        
        # Create summary statistics
        summary = {
            'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Region': f"{self.region['lat_min']}°N-{self.region['lat_max']}°N, {self.region['lon_min']}°E-{self.region['lon_max']}°E",
            'Total Data Points': len(df),
            'Date Range': f"{df['datetime'].min()} to {df['datetime'].max()}",
            'Unique Dates': len(df['date'].unique()),
            'Latitude Range': f"{df['latitude'].min():.3f}° to {df['latitude'].max():.3f}°",
            'Longitude Range': f"{df['longitude'].min():.3f}° to {df['longitude'].max():.3f}°"
        }
        
        # Add SSH statistics
        for var in ['ssh_karin', 'ssh_karin_2']:
            if var in df.columns:
                data = df[var].dropna()
                if len(data) > 0:
                    summary[f'{var}_count'] = len(data)
                    summary[f'{var}_mean'] = f"{data.mean():.3f} m"
                    summary[f'{var}_std'] = f"{data.std():.3f} m"
                    summary[f'{var}_range'] = f"{data.min():.3f} to {data.max():.3f} m"
        
        # Save summary
        summary_df = pd.DataFrame(list(summary.items()), columns=['Parameter', 'Value'])
        summary_df.to_csv(self.output_dir / "analysis_summary.csv", index=False)
        
        print("\nAnalysis Summary:")
        print("-" * 50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print(f"\nSummary saved: {self.output_dir / 'analysis_summary.csv'}")


def main():
    """Main function to run the analysis."""
    # Configuration
    DATA_DIRECTORY = r"C:/Users/abhik/Desktop/project related work/SWOT_L2_LR_SSH_2.0_2.0-20250822_162623"
    MAX_FILES = 10  # Adjust as needed
    
    print("SWOT SSH Analysis - Andaman and Nicobar Region")
    print("=" * 60)
    print(f"Data Directory: {DATA_DIRECTORY}")
    print(f"Target Region: 6°N-15°N, 92°E-95°E")
    print(f"Max Files: {MAX_FILES}")
    print()
    
    try:
        # Initialize analyzer
        analyzer = AndamanNicobarSWOTAnalyzer(DATA_DIRECTORY)
        
        # Run analysis
        analyzer.analyze_region_data(max_files=MAX_FILES)
        
        # Save summary
        analyzer.save_summary_report()
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
