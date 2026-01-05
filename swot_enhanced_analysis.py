"""
Enhanced SWOT Data Analysis Script
Advanced analysis for SWOT L2 SSH data with internal wave detection,
quality control, and sophisticated visualization techniques.

Based on SWOT L2 LR SSH v2.0 best practices for internal wave analysis.
"""

import os
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Enhanced imports for advanced analysis
import xarray as xr
import netCDF4 as nc
import pandas as pd
import numpy as np
from pathlib import Path

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Scientific computation
from scipy import signal, stats
from scipy.ndimage import gaussian_filter

# Configuration
TARGET_LAT = None  # Will be auto-detected
TARGET_LON = None  # Will be auto-detected
BOX_HALF_DEG = 0.5  # Half-width of the lat/lon box in degrees
USE_SSHA_FIRST = True  # Prefer SSHA if available for wave analysis
KEEP_QUAL_EQ_ZERO_ONLY = True  # Strict QC: keep only qual==0
ROLLING_MEAN_COUNT = 25  # Window for high-pass filtering

def to_180(lon):
    """Convert 0..360 to -180..180 if needed."""
    lon2 = ((lon + 180) % 360) - 180
    return lon2

def pick_var(ds, names):
    """Return first variable present in 'names'."""
    for n in names:
        if n in ds.variables:
            return ds[n]
    raise KeyError(f"None of {names} found in dataset.")

def get_ssha_varnames(prefer_ssha=True):
    """Return (data_var_candidates, qual_var_suffix) for SSH/SSHA."""
    if prefer_ssha:
        return (["ssha_karin_2", "ssha_karin"], "_qual")
    else:
        return (["ssh_karin_2", "ssh_karin"], "_qual")

def extract_timestamp_from_filename(filename):
    """Extract timestamp from SWOT filename."""
    pattern = r'(\d{8}T\d{6})_\d{8}T\d{6}_PIC2_01\.nc$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

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

def subset_and_qc_advanced(file_path, target_lat, target_lon, box_half_deg):
    """
    Advanced SWOT data extraction with quality control and geographic subsetting.
    Based on PO.DAAC best practices for SWOT L2 LR SSH v2.0.
    """
    try:
        ds = xr.open_dataset(file_path, mask_and_scale=True)
        
        # Get coordinate variables with fallback options
        lat = ds["latitude"] if "latitude" in ds.variables else ds["lat"]
        lon = ds["longitude"] if "longitude" in ds.variables else ds["lon"]
        
        # Normalize longitude to -180..180 if necessary
        if float(lon.max()) > 180.0:
            lon = to_180(lon)
        
        # Choose SSHA (preferred) or SSH
        data_candidates, qual_suffix = get_ssha_varnames(prefer_ssha=USE_SSHA_FIRST)
        try:
            data_var = pick_var(ds, data_candidates)
        except KeyError:
            # Fallback to SSH if SSHA not present
            data_var = pick_var(ds, ["ssh_karin_2", "ssh_karin"])
        
        # Quality variable: <varname>_qual if present
        qual_name = f"{data_var.name}{qual_suffix}"
        qual = ds[qual_name] if qual_name in ds.variables else None
        
        # Build geographic mask
        mask = ((lat >= target_lat - box_half_deg) & (lat <= target_lat + box_half_deg) &
                (lon >= target_lon - box_half_deg) & (lon <= target_lon + box_half_deg))
        
        # Apply quality control (keep only good data: qual==0)
        if qual is not None and KEEP_QUAL_EQ_ZERO_ONLY:
            mask = mask & (qual == 0)
        
        # Create small dataset with mask applied
        small = xr.Dataset(
            dict(val=data_var.where(mask)),
            coords=dict(latitude=lat, longitude=lon, time=ds["time"])
        )
        
        # Convert to table format
        tab = small.to_dataframe().reset_index()
        tab = tab.dropna(subset=["val"])
        tab.rename(columns={"val": "ssh"}, inplace=True)
        
        # Add metadata
        tab["file"] = os.path.basename(file_path)
        for meta_name in ["cycle_number", "pass_number"]:
            if meta_name in ds.variables:
                try:
                    v = ds[meta_name].values
                    tab[meta_name] = v.item() if np.ndim(v) == 0 else v[0]
                except Exception:
                    pass
        
        ds.close()
        return tab
        
    except Exception as e:
        print(f"[WARN] Error processing {os.path.basename(file_path)}: {e}")
        return None

def high_pass_analysis(file_path, target_lat, target_lon, window_count=25):
    """
    Perform high-pass filtering analysis for internal wave detection.
    """
    try:
        ds = xr.open_dataset(file_path, mask_and_scale=True)
        lat = ds["latitude"] if "latitude" in ds.variables else ds["lat"]
        lon = ds["longitude"] if "longitude" in ds.variables else ds["lon"]
        
        if float(lon.max()) > 180.0:
            lon = to_180(lon)
        
        # Get SSH/SSHA data
        data_candidates, _ = get_ssha_varnames(prefer_ssha=USE_SSHA_FIRST)
        try:
            ssh_var = pick_var(ds, data_candidates)
        except KeyError:
            ssh_var = pick_var(ds, ["ssh_karin_2", "ssh_karin"])
        
        # Find pixel nearest to target point
        if len(ssh_var.dims) == 2:
            # Get first line for distance calculation
            lon0 = lon.isel({ssh_var.dims[0]: 0})
            lat0 = lat.isel({ssh_var.dims[0]: 0})
            
            # Calculate distance
            d2 = (np.cos(np.deg2rad(target_lat)) * (lon0 - target_lon))**2 + (lat0 - target_lat)**2
            j = int(d2.argmin())  # across-track pixel index
            
            # Extract 1D series along track at that pixel
            track = ssh_var.isel({ssh_var.dims[1]: j}).to_series().dropna()
            
            if len(track) >= window_count * 2:
                # Apply rolling mean for high-pass filtering
                roll = track.rolling(window=window_count, center=True, 
                                   min_periods=max(3, window_count//3)).mean()
                hp = track - roll
                
                # Calculate statistics
                hp_clean = hp.dropna()
                if len(hp_clean) > 0:
                    stats = {
                        'rms_amplitude': np.sqrt(np.mean(hp_clean**2)),
                        'peak_to_peak': hp_clean.max() - hp_clean.min(),
                        'std_dev': hp_clean.std(),
                        'data_points': len(hp_clean)
                    }
                    
                    ds.close()
                    return hp, stats
        
        ds.close()
        return None, None
        
    except Exception as e:
        print(f"[WARN] High-pass analysis failed for {os.path.basename(file_path)}: {e}")
        return None, None

def create_comprehensive_plots(all_data, daily_stats, monthly_stats):
    """
    Create comprehensive visualization suite.
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # Main time series
    plt.subplot(3, 3, 1)
    plt.scatter(all_data['time'], all_data['ssh'], c=all_data['ssh'], 
               cmap='RdYlBu_r', s=1, alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel('SSH/SSHA (m)')
    plt.title('All SSH/SSHA Measurements', fontweight='bold')
    plt.colorbar(label='SSH/SSHA (m)')
    
    # Daily statistics
    plt.subplot(3, 3, 2)
    plt.plot(daily_stats.index, daily_stats['ssh_mean'], 'b-o', markersize=3)
    plt.fill_between(daily_stats.index, 
                     daily_stats['ssh_mean'] - daily_stats['ssh_std'],
                     daily_stats['ssh_mean'] + daily_stats['ssh_std'],
                     alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('SSH/SSHA (m)')
    plt.title('Daily Statistics (Mean +/- Std)', fontweight='bold')
    plt.xticks(rotation=45)
    
    # Geographic distribution
    plt.subplot(3, 3, 3)
    scatter = plt.scatter(all_data['longitude'], all_data['latitude'], 
                         c=all_data['ssh'], cmap='RdYlBu_r', s=5, alpha=0.7)
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('Geographic Distribution', fontweight='bold')
    plt.colorbar(scatter, label='SSH/SSHA (m)')
    
    # SSH distribution
    plt.subplot(3, 3, 4)
    plt.hist(all_data['ssh'], bins=50, alpha=0.7, density=True)
    plt.axvline(all_data['ssh'].mean(), color='red', linestyle='--', 
               label=f'Mean: {all_data["ssh"].mean():.4f}m')
    plt.xlabel('SSH/SSHA (m)')
    plt.ylabel('Density')
    plt.title('SSH/SSHA Distribution', fontweight='bold')
    plt.legend()
    
    # Amplitude time series
    plt.subplot(3, 3, 5)
    amplitude = daily_stats['ssh_max'] - daily_stats['ssh_min']
    plt.plot(daily_stats.index, amplitude, 'purple', marker='o', markersize=3)
    plt.xlabel('Date')
    plt.ylabel('Daily Amplitude (m)')
    plt.title('Daily SSH/SSHA Amplitude', fontweight='bold')
    plt.xticks(rotation=45)
    
    # Monthly comparison
    plt.subplot(3, 3, 6)
    monthly_means = monthly_stats['ssh_mean']
    monthly_stds = monthly_stats['ssh_std']
    x_pos = range(len(monthly_means))
    plt.bar(x_pos, monthly_means, yerr=monthly_stds, capsize=5, alpha=0.7)
    plt.xlabel('Month')
    plt.ylabel('Mean SSH/SSHA (m)')
    plt.title('Monthly Statistics', fontweight='bold')
    plt.xticks(x_pos, [str(m) for m in monthly_stats.index], rotation=45)
    
    # Data density
    plt.subplot(3, 3, 7)
    daily_counts = all_data.groupby(all_data['time'].dt.date).size()
    plt.plot(daily_counts.index, daily_counts.values, 'green', marker='o', markersize=3)
    plt.xlabel('Date')
    plt.ylabel('Measurements per Day')
    plt.title('Data Density', fontweight='bold')
    plt.xticks(rotation=45)
    
    # Mean vs variability
    plt.subplot(3, 3, 8)
    plt.scatter(daily_stats['ssh_mean'], daily_stats['ssh_std'], 
               c=range(len(daily_stats)), cmap='viridis', alpha=0.7)
    plt.xlabel('Daily Mean SSH/SSHA (m)')
    plt.ylabel('Daily Std Dev (m)')
    plt.title('Mean vs Variability', fontweight='bold')
    plt.colorbar(label='Time Index')
    
    # Power spectral density
    plt.subplot(3, 3, 9)
    if len(daily_stats) > 10:
        ssh_series = daily_stats['ssh_mean'].values - daily_stats['ssh_mean'].mean()
        frequencies, psd = signal.periodogram(ssh_series, fs=1.0)
        plt.loglog(frequencies[1:], psd[1:], 'b-', alpha=0.7)
        plt.xlabel('Frequency (cycles/day)')
        plt.ylabel('Power Spectral Density')
        plt.title('PSD of Daily SSH/SSHA', fontweight='bold')
        
        # Add period markers
        for period in [7, 14, 30]:
            freq = 1.0 / period
            if frequencies[1] < freq < frequencies[-1]:
                plt.axvline(freq, color='red', linestyle='--', alpha=0.5)
                plt.text(freq, psd.max() * 0.1, f'{period}d', rotation=90)
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor spectral analysis', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Power Spectral Density', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Enhanced main function with advanced SWOT analysis capabilities.
    """
    global TARGET_LAT, TARGET_LON
    
    # Data directory
    data_directory = r"c:\Users\abhik\Desktop\project related work\SWOT_L2_LR_SSH_2.0_2.0-20250815_094335"
    
    print("Enhanced SWOT L2 SSH Data Analysis")
    print("=" * 60)
    print("Features:")
    print("- Quality-controlled data extraction")
    print("- Internal wave analysis with high-pass filtering")
    print("- Advanced statistical analysis")
    print("- Comprehensive visualizations")
    print("- Export capabilities for further research")
    print("=" * 60)
    
    # Get sorted files
    sorted_files = get_sorted_nc_files(data_directory)
    
    if not sorted_files:
        print("No NetCDF files found!")
        return
    
    print(f"Found {len(sorted_files)} NetCDF files")
    print(f"Date range: {sorted_files[0][0]} to {sorted_files[-1][0]}")
    
    # Auto-detect target location from first file
    try:
        ds_sample = xr.open_dataset(sorted_files[0][1])
        lat_data = ds_sample["latitude"] if "latitude" in ds_sample else ds_sample["lat"]
        lon_data = ds_sample["longitude"] if "longitude" in ds_sample else ds_sample["lon"]
        TARGET_LAT = float(lat_data.mean())
        TARGET_LON = float(lon_data.mean())
        ds_sample.close()
        print(f"Auto-detected target: {TARGET_LAT:.3f}deg N, {TARGET_LON:.3f}deg E")
    except Exception as e:
        print(f"Warning: Could not auto-detect target location: {e}")
        TARGET_LAT, TARGET_LON = 0.0, 0.0
    
    # Process all files
    print(f"\nProcessing files in {BOX_HALF_DEG*2:.1f}deg box around target...")
    all_rows = []
    internal_wave_stats = []
    
    for i, (timestamp, file_path, filename) in enumerate(sorted_files):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(sorted_files)}: {filename}")
        
        # Extract data with QC
        df = subset_and_qc_advanced(file_path, TARGET_LAT, TARGET_LON, BOX_HALF_DEG)
        if df is not None and len(df) > 0:
            df['timestamp'] = timestamp
            all_rows.append(df)
        
        # Internal wave analysis for first few files
        if i < 3:
            hp_data, hw_stats = high_pass_analysis(file_path, TARGET_LAT, TARGET_LON, ROLLING_MEAN_COUNT)
            if hw_stats:
                hw_stats['file'] = filename
                hw_stats['timestamp'] = timestamp
                internal_wave_stats.append(hw_stats)
    
    if not all_rows:
        print("No data extracted! Check target coordinates and box size.")
        return
    
    # Combine all data
    all_data = pd.concat(all_rows, ignore_index=True)
    all_data['time'] = pd.to_datetime(all_data['time'])
    all_data = all_data.sort_values('time')
    
    print(f"\n[SUCCESS] Combined dataset: {len(all_data):,} measurements")
    print(f"[SUCCESS] Time span: {(all_data['time'].max() - all_data['time'].min()).days} days")
    
    # Calculate statistics
    daily_stats = all_data.groupby(all_data['time'].dt.date).agg({
        'ssh': ['min', 'max', 'mean', 'std', 'count']
    }).round(4)
    daily_stats.columns = ['ssh_min', 'ssh_max', 'ssh_mean', 'ssh_std', 'ssh_count']
    daily_stats.index = pd.to_datetime(daily_stats.index)
    
    monthly_stats = all_data.groupby(all_data['time'].dt.to_period('M')).agg({
        'ssh': ['min', 'max', 'mean', 'std', 'count']
    }).round(4)
    monthly_stats.columns = ['ssh_min', 'ssh_max', 'ssh_mean', 'ssh_std', 'ssh_count']
    
    # Print summary
    print(f"\nDATA SUMMARY")
    print("=" * 40)
    print(f"SSH/SSHA statistics:")
    print(f"  Mean: {all_data['ssh'].mean():.4f} +/- {all_data['ssh'].std():.4f} m")
    print(f"  Range: {all_data['ssh'].min():.4f} to {all_data['ssh'].max():.4f} m")
    print(f"  Total amplitude: {all_data['ssh'].max() - all_data['ssh'].min():.4f} m")
    
    if internal_wave_stats:
        print(f"\nInternal Wave Analysis ({len(internal_wave_stats)} files):")
        hw_df = pd.DataFrame(internal_wave_stats)
        print(f"  Average RMS amplitude: {hw_df['rms_amplitude'].mean():.4f} m")
        print(f"  Max peak-to-peak: {hw_df['peak_to_peak'].max():.4f} m")
    
    # Create visualizations
    print(f"\nCreating comprehensive visualizations...")
    create_comprehensive_plots(all_data, daily_stats, monthly_stats)
    
    # Export results
    output_dir = os.path.dirname(data_directory)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export main dataset
    main_csv = os.path.join(output_dir, f'swot_enhanced_analysis_{timestamp_str}.csv')
    all_data.to_csv(main_csv, index=False)
    
    # Export statistics
    stats_csv = os.path.join(output_dir, f'swot_daily_stats_{timestamp_str}.csv')
    daily_stats.to_csv(stats_csv)
    
    hw_csv = None
    if internal_wave_stats:
        hw_csv = os.path.join(output_dir, f'swot_internal_waves_{timestamp_str}.csv')
        pd.DataFrame(internal_wave_stats).to_csv(hw_csv, index=False)
    
    print(f"\n>>> ENHANCED ANALYSIS COMPLETE! <<<")
    print(f"[SUCCESS] Main dataset: {main_csv}")
    print(f"[SUCCESS] Daily statistics: {stats_csv}")
    if hw_csv:
        print(f"[SUCCESS] Internal wave stats: {hw_csv}")
    
    print(f"\nKey results:")
    print(f"- {len(all_data):,} quality-controlled measurements")
    print(f"- {len(daily_stats)} days of data")
    print(f"- SSH/SSHA range: {all_data['ssh'].min():.4f} to {all_data['ssh'].max():.4f} m")
    if internal_wave_stats:
        print(f"- Internal wave analysis on {len(internal_wave_stats)} files")

if __name__ == "__main__":
    main()
