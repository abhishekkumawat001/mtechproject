"""
SWOT Data Analysis Script
This script loops through SWOT L2 SSH NetCDF files in chronological order
for data analysis and visualization.
"""

import os
import glob
import re
from datetime import datetime
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from SWOT filename
    Example: SWOT_L2_LR_SSH_Basic_026_230_20250101T004209_20250101T013337_PIC2_01.nc
    Returns the start timestamp: 20250101T004209
    """
    # Pattern to match the timestamp in the filename
    pattern = r'(\d{8}T\d{6})_\d{8}T\d{6}_PIC2_01\.nc$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def parse_timestamp(timestamp_str):
    """
    Convert timestamp string to datetime object
    Format: YYYYMMDDTHHMMSS
    """
    try:
        return datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
    except ValueError:
        return None

def get_sorted_nc_files(directory_path):
    """
    Get all .nc files sorted by their timestamp in chronological order
    """
    # Get all .nc files in the directory
    nc_files = glob.glob(os.path.join(directory_path, "*.nc"))
    
    # Extract timestamps and create list of tuples (datetime, filepath)
    file_timestamps = []
    for file_path in nc_files:
        filename = os.path.basename(file_path)
        timestamp_str = extract_timestamp_from_filename(filename)
        if timestamp_str:
            dt = parse_timestamp(timestamp_str)
            if dt:
                file_timestamps.append((dt, file_path))
    
    # Sort by datetime
    file_timestamps.sort(key=lambda x: x[0])
    
    return file_timestamps

def analyze_swot_file(file_path):
    """
    Analyze a single SWOT NetCDF file and extract key information
    """
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            # Get basic information about the dataset
            print(f"\nAnalyzing: {os.path.basename(file_path)}")
            print(f"Dimensions: {list(dataset.dimensions.keys())}")
            print(f"Variables: {list(dataset.variables.keys())}")
            
            # Extract some key variables (adjust based on actual file structure)
            data_info = {
                'file_path': file_path,
                'dimensions': dict(dataset.dimensions),
                'variables': list(dataset.variables.keys())
            }
            
            # Try to extract SSH data if available
            if 'ssh_karin_2' in dataset.variables:
                ssh_data = dataset.variables['ssh_karin_2'][:]
                data_info['ssh_mean'] = np.nanmean(ssh_data)
                data_info['ssh_std'] = np.nanstd(ssh_data)
                data_info['ssh_min'] = np.nanmin(ssh_data)
                data_info['ssh_max'] = np.nanmax(ssh_data)
            
            # Extract latitude and longitude if available
            if 'latitude_karin' in dataset.variables and 'longitude_karin' in dataset.variables:
                lat = dataset.variables['latitude_karin'][:]
                lon = dataset.variables['longitude_karin'][:]
                data_info['lat_range'] = (np.nanmin(lat), np.nanmax(lat))
                data_info['lon_range'] = (np.nanmin(lon), np.nanmax(lon))
            
            return data_info
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def plot_ssh_data(file_path):
    """
    Create a basic plot of SSH data from a NetCDF file
    """
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            if 'ssh_karin_2' in dataset.variables:
                ssh_data = dataset.variables['ssh_karin_2'][:]
                
                plt.figure(figsize=(12, 8))
                
                # Plot 1: SSH data heatmap
                plt.subplot(2, 2, 1)
                plt.imshow(ssh_data, cmap='viridis', aspect='auto')
                plt.colorbar(label='SSH (m)')
                plt.title(f'SSH Data - {os.path.basename(file_path)}')
                
                # Plot 2: SSH histogram
                plt.subplot(2, 2, 2)
                ssh_flat = ssh_data.flatten()
                ssh_clean = ssh_flat[~np.isnan(ssh_flat)]
                plt.hist(ssh_clean, bins=50, alpha=0.7)
                plt.xlabel('SSH (m)')
                plt.ylabel('Frequency')
                plt.title('SSH Distribution')
                
                # Plot 3: SSH along track (if 2D data)
                if len(ssh_data.shape) == 2:
                    plt.subplot(2, 2, 3)
                    along_track_mean = np.nanmean(ssh_data, axis=1)
                    plt.plot(along_track_mean)
                    plt.xlabel('Along-track point')
                    plt.ylabel('Mean SSH (m)')
                    plt.title('Mean SSH Along Track')
                
                # Plot 4: SSH across track (if 2D data)
                if len(ssh_data.shape) == 2:
                    plt.subplot(2, 2, 4)
                    across_track_mean = np.nanmean(ssh_data, axis=0)
                    plt.plot(across_track_mean)
                    plt.xlabel('Across-track point')
                    plt.ylabel('Mean SSH (m)')
                    plt.title('Mean SSH Across Track')
                
                plt.tight_layout()
                plt.show()
                
    except Exception as e:
        print(f"Error plotting {file_path}: {str(e)}")

def main():
    """
    Main function to process all SWOT files in chronological order
    """
    # Set the directory containing your SWOT files
    data_directory = r"c:\Users\abhik\Desktop\project related work\SWOT_L2_LR_SSH_2.0_2.0-20250815_094335"
    
    print("SWOT Data Analysis - Processing files in chronological order")
    print("=" * 60)
    
    # Get sorted files
    sorted_files = get_sorted_nc_files(data_directory)
    
    if not sorted_files:
        print("No NetCDF files found in the specified directory!")
        return
    
    print(f"Found {len(sorted_files)} NetCDF files")
    print(f"Date range: {sorted_files[0][0]} to {sorted_files[-1][0]}")
    print()
    
    # Store analysis results
    analysis_results = []
    
    # Process each file in chronological order
    for i, (timestamp, file_path) in enumerate(sorted_files):
        print(f"Processing file {i+1}/{len(sorted_files)}")
        print(f"Timestamp: {timestamp}")
        
        # Analyze the file
        result = analyze_swot_file(file_path)
        if result:
            result['timestamp'] = timestamp
            analysis_results.append(result)
        
        # Optional: Plot the first few files for visualization
        if i < 3:  # Plot first 3 files as examples
            plot_ssh_data(file_path)
        
        print("-" * 40)
    
    # Create summary analysis
    if analysis_results:
        print("\nSummary Analysis:")
        print("=" * 40)
        
        # Convert to DataFrame for easier analysis
        summary_data = []
        for result in analysis_results:
            if 'ssh_mean' in result:
                summary_data.append({
                    'timestamp': result['timestamp'],
                    'ssh_mean': result['ssh_mean'],
                    'ssh_std': result['ssh_std'],
                    'ssh_min': result['ssh_min'],
                    'ssh_max': result['ssh_max']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Plot time series of SSH statistics
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(df['timestamp'], df['ssh_mean'], 'b-o', markersize=3)
            plt.xlabel('Date')
            plt.ylabel('Mean SSH (m)')
            plt.title('Mean SSH Time Series')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 2)
            plt.plot(df['timestamp'], df['ssh_std'], 'r-o', markersize=3)
            plt.xlabel('Date')
            plt.ylabel('SSH Standard Deviation (m)')
            plt.title('SSH Variability Time Series')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 3)
            plt.plot(df['timestamp'], df['ssh_min'], 'g-o', markersize=3, label='Min')
            plt.plot(df['timestamp'], df['ssh_max'], 'm-o', markersize=3, label='Max')
            plt.xlabel('Date')
            plt.ylabel('SSH (m)')
            plt.title('SSH Range Time Series')
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 4)
            plt.scatter(df['ssh_mean'], df['ssh_std'], alpha=0.6)
            plt.xlabel('Mean SSH (m)')
            plt.ylabel('SSH Standard Deviation (m)')
            plt.title('Mean vs Variability')
            
            plt.tight_layout()
            plt.show()
            
            # Save summary to CSV
            csv_file = os.path.join(os.path.dirname(data_directory), 'swot_ssh_summary.csv')
            df.to_csv(csv_file, index=False)
            print(f"Summary data saved to: {csv_file}")

if __name__ == "__main__":
    main()
