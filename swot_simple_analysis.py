"""
Simplified SWOT Data Analysis Script
A cleaned-up version with configuration file support and better error handling.
"""

import os
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data processing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to import optional libraries
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("Warning: xarray not available. Some features may be limited.")

try:
    import netCDF4 as nc
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False
    print("Warning: netCDF4 not available. Using fallback methods.")

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Spectral analysis will be limited.")

# Import configuration
try:
    from swot_config import *
    print("Configuration loaded from swot_config.py")
except ImportError:
    print("Configuration file not found. Using default settings.")
    # Default configuration
    DATA_DIRECTORY = r"c:\Users\abhik\Desktop\project related work\SWOT_L2_LR_SSH_2.0_2.0-20250815_094335"
    BOX_HALF_DEG = 0.5
    USE_SSHA_FIRST = True
    KEEP_QUAL_EQ_ZERO_ONLY = True
    ROLLING_MEAN_COUNT = 25
    TARGET_LAT = None
    TARGET_LON = None
    MAX_FILES_FOR_INTERNAL_WAVES = 3
    PROGRESS_UPDATE_INTERVAL = 10

def extract_timestamp_from_filename(filename):
    """Extract timestamp from SWOT filename."""
    pattern = r'(\d{8}T\d{6})_\d{8}T\d{6}_PIC2_01\.nc$'
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def parse_timestamp(timestamp_str):
    """Convert timestamp string to datetime object."""
    try:
        return datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
    except ValueError:
        return None

def get_sorted_nc_files(directory_path):
    """Get all .nc files sorted by their timestamp in chronological order."""
    nc_files = glob.glob(os.path.join(directory_path, "*.nc"))
    
    file_timestamps = []
    for file_path in nc_files:
        filename = os.path.basename(file_path)
        timestamp_str = extract_timestamp_from_filename(filename)
        if timestamp_str:
            dt = parse_timestamp(timestamp_str)
            if dt:
                file_timestamps.append((dt, file_path, filename))
    
    file_timestamps.sort(key=lambda x: x[0])
    return file_timestamps

def extract_ssh_data_simple(file_path):
    """Simple SSH data extraction with fallback methods."""
    try:
        if HAS_XARRAY:
            # Try xarray first
            ds = xr.open_dataset(file_path, mask_and_scale=True)
            
            # Find SSH/SSHA variables
            ssh_vars = ['ssha_karin_2', 'ssha_karin', 'ssh_karin_2', 'ssh_karin']
            ssh_var = None
            for var in ssh_vars:
                if var in ds.variables:
                    ssh_var = var
                    break
            
            if ssh_var is None:
                print(f"No SSH variables found in {os.path.basename(file_path)}")
                ds.close()
                return None
            
            # Get coordinates
            lat_vars = ['latitude', 'lat', 'latitude_karin']
            lon_vars = ['longitude', 'lon', 'longitude_karin']
            
            lat_var = None
            lon_var = None
            for var in lat_vars:
                if var in ds.variables:
                    lat_var = var
                    break
            for var in lon_vars:
                if var in ds.variables:
                    lon_var = var
                    break
            
            if lat_var is None or lon_var is None:
                print(f"No coordinate variables found in {os.path.basename(file_path)}")
                ds.close()
                return None
            
            # Extract data
            ssh_data = ds[ssh_var].values
            lat_data = ds[lat_var].values
            lon_data = ds[lon_var].values
            
            # Calculate statistics
            ssh_flat = ssh_data.flatten()
            ssh_clean = ssh_flat[~np.isnan(ssh_flat)]
            
            if len(ssh_clean) == 0:
                ds.close()
                return None
            
            result = {
                'filename': os.path.basename(file_path),
                'ssh_mean': np.mean(ssh_clean),
                'ssh_std': np.std(ssh_clean),
                'ssh_min': np.min(ssh_clean),
                'ssh_max': np.max(ssh_clean),
                'ssh_median': np.median(ssh_clean),
                'valid_points': len(ssh_clean),
                'total_points': len(ssh_flat),
                'coverage_pct': (len(ssh_clean) / len(ssh_flat)) * 100,
                'lat_min': np.nanmin(lat_data),
                'lat_max': np.nanmax(lat_data),
                'lon_min': np.nanmin(lon_data),
                'lon_max': np.nanmax(lon_data)
            }
            
            ds.close()
            return result
            
        elif HAS_NETCDF4 and 'nc' in globals():
            # Fallback to netCDF4
            with nc.Dataset(file_path, 'r') as dataset:
                # Find SSH variables
                ssh_vars = ['ssha_karin_2', 'ssha_karin', 'ssh_karin_2', 'ssh_karin']
                ssh_var = None
                for var in ssh_vars:
                    if var in dataset.variables:
                        ssh_var = var
                        break
                
                if ssh_var is None:
                    return None
                
                ssh_data = dataset.variables[ssh_var][:]
                ssh_flat = ssh_data.flatten()
                ssh_clean = ssh_flat[~np.isnan(ssh_flat)]
                
                if len(ssh_clean) == 0:
                    return None
                
                result = {
                    'filename': os.path.basename(file_path),
                    'ssh_mean': np.mean(ssh_clean),
                    'ssh_std': np.std(ssh_clean),
                    'ssh_min': np.min(ssh_clean),
                    'ssh_max': np.max(ssh_clean),
                    'ssh_median': np.median(ssh_clean),
                    'valid_points': len(ssh_clean),
                    'total_points': len(ssh_flat),
                    'coverage_pct': (len(ssh_clean) / len(ssh_flat)) * 100
                }
                
                return result
        else:
            print("Neither xarray nor netCDF4 available. Cannot process files.")
            return None
            
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def create_summary_plots(analysis_results):
    """Create summary visualization plots."""
    if not analysis_results:
        print("No data available for plotting.")
        return
    
    df = pd.DataFrame(analysis_results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SWOT Data Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. Time series of mean SSH
    axes[0, 0].plot(df.index, df['ssh_mean'], 'b-o', markersize=3)
    axes[0, 0].set_xlabel('File Index')
    axes[0, 0].set_ylabel('Mean SSH/SSHA (m)')
    axes[0, 0].set_title('Mean SSH/SSHA Time Series')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. SSH distribution
    axes[0, 1].hist(df['ssh_mean'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df['ssh_mean'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["ssh_mean"].mean():.4f}m')
    axes[0, 1].set_xlabel('Mean SSH/SSHA (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('SSH/SSHA Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Variability analysis
    axes[0, 2].plot(df.index, df['ssh_std'], 'r-o', markersize=3)
    axes[0, 2].set_xlabel('File Index')
    axes[0, 2].set_ylabel('SSH/SSHA Std Dev (m)')
    axes[0, 2].set_title('SSH/SSHA Variability')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Data coverage
    axes[1, 0].plot(df.index, df['coverage_pct'], 'g-o', markersize=3)
    axes[1, 0].set_xlabel('File Index')
    axes[1, 0].set_ylabel('Data Coverage (%)')
    axes[1, 0].set_title('Data Quality (Coverage)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Range analysis
    axes[1, 1].plot(df.index, df['ssh_min'], 'g--', label='Min', alpha=0.7)
    axes[1, 1].plot(df.index, df['ssh_max'], 'r--', label='Max', alpha=0.7)
    amplitude = df['ssh_max'] - df['ssh_min']
    axes[1, 1].plot(df.index, amplitude, 'purple', marker='o', markersize=3, label='Range')
    axes[1, 1].set_xlabel('File Index')
    axes[1, 1].set_ylabel('SSH/SSHA (m)')
    axes[1, 1].set_title('SSH/SSHA Range Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Mean vs Variability
    axes[1, 2].scatter(df['ssh_mean'], df['ssh_std'], alpha=0.7, s=50)
    axes[1, 2].set_xlabel('Mean SSH/SSHA (m)')
    axes[1, 2].set_ylabel('SSH/SSHA Std Dev (m)')
    axes[1, 2].set_title('Mean vs Variability')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main analysis function."""
    print("Simplified SWOT Data Analysis")
    print("=" * 50)
    print(f"Data directory: {DATA_DIRECTORY}")
    print(f"Analysis settings:")
    print(f"  - Box size: {BOX_HALF_DEG*2:.1f} degrees")
    print(f"  - Prefer SSHA: {USE_SSHA_FIRST}")
    print(f"  - Quality control: {KEEP_QUAL_EQ_ZERO_ONLY}")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIRECTORY):
        print(f"Error: Data directory not found: {DATA_DIRECTORY}")
        print("Please update the DATA_DIRECTORY in swot_config.py")
        return
    
    # Get sorted files
    sorted_files = get_sorted_nc_files(DATA_DIRECTORY)
    
    if not sorted_files:
        print("No NetCDF files found in the specified directory!")
        print("Please check the directory path and file naming convention.")
        return
    
    print(f"Found {len(sorted_files)} NetCDF files")
    print(f"Date range: {sorted_files[0][0]} to {sorted_files[-1][0]}")
    print(f"Time span: {(sorted_files[-1][0] - sorted_files[0][0]).days} days")
    
    # Process all files
    print(f"\nProcessing files...")
    analysis_results = []
    
    for i, (timestamp, file_path, filename) in enumerate(sorted_files):
        if i % PROGRESS_UPDATE_INTERVAL == 0:
            print(f"Processing {i+1}/{len(sorted_files)}: {filename}")
        
        result = extract_ssh_data_simple(file_path)
        if result:
            result['timestamp'] = timestamp
            result['file_index'] = i
            analysis_results.append(result)
    
    if not analysis_results:
        print("No valid data extracted from any files!")
        print("Please check the file format and variable names.")
        return
    
    print(f"\n[SUCCESS] Processed {len(analysis_results)}/{len(sorted_files)} files successfully")
    
    # Create summary statistics
    df = pd.DataFrame(analysis_results)
    
    print(f"\nDATA SUMMARY")
    print("=" * 40)
    print(f"SSH/SSHA Statistics:")
    print(f"  Overall mean: {df['ssh_mean'].mean():.4f} +/- {df['ssh_mean'].std():.4f} m")
    print(f"  Range: {df['ssh_mean'].min():.4f} to {df['ssh_mean'].max():.4f} m")
    print(f"  Median variability: {df['ssh_std'].median():.4f} m")
    print(f"  Average data coverage: {df['coverage_pct'].mean():.1f}%")
    print(f"  Total valid measurements: {df['valid_points'].sum():,}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    fig = create_summary_plots(analysis_results)
    
    # Export results
    output_dir = os.path.dirname(DATA_DIRECTORY)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    csv_file = os.path.join(output_dir, f'swot_analysis_summary_{timestamp_str}.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"\n[SUCCESS] Analysis Complete!")
    print(f"Results saved to: {csv_file}")
    print(f"\nKey Findings:")
    print(f"- Analyzed {len(analysis_results)} files successfully")
    print(f"- SSH/SSHA range: {df['ssh_mean'].min():.4f} to {df['ssh_mean'].max():.4f} m")
    print(f"- Average variability: {df['ssh_std'].mean():.4f} m")
    print(f"- Data quality: {df['coverage_pct'].mean():.1f}% average coverage")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the error message and try again.")
        input("Press Enter to exit...")
