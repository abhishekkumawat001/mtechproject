"""
SWOT Internal Wave Visualization Generator
Creates publication-quality internal wave analysis plots from SWOT L2 SSH data.

This script generates various types of internal wave visualizations including:
- Swath plots showing wave patterns
- High-pass filtered signals
- Spectral analysis plots
- Geographic context maps
- Along-track wave analysis
"""

import os
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configuration for internal wave analysis
class InternalWaveConfig:
    def __init__(self):
        self.data_directory = r"c:\Users\abhik\Desktop\project related work\SWOT_L2_LR_SSH_2.0_2.0-20250815_094335"
        self.output_directory = r"c:\Users\abhik\Desktop\project related work\figures_internalwaves"
        
        # Analysis parameters
        self.high_pass_window = 50  # km equivalent for high-pass filtering
        self.along_track_resolution = 0.25  # km approximate along-track resolution
        self.across_track_resolution = 2.0  # km approximate across-track resolution
        
        # Geographic focus area (can be auto-detected or manually set)
        self.target_lat = None  # Will be auto-detected
        self.target_lon = None  # Will be auto-detected
        self.focus_box_size = 2.0  # degrees for geographic focus
        
        # Visualization parameters
        self.wave_colormap = 'RdBu_r'
        self.bathymetry_colormap = 'viridis_r'
        self.figure_dpi = 150
        self.save_figures = True

config = InternalWaveConfig()

def load_bathymetry_data():
    """
    Load bathymetry data for context. 
    This is a placeholder - you would load actual bathymetry data here.
    Common sources: GEBCO, ETOPO1, or regional bathymetry datasets.
    """
    # Placeholder bathymetry data
    # In practice, you would load from a netCDF file or other data source
    print("Note: Using synthetic bathymetry data.")
    print("For real analysis, load bathymetry from GEBCO, ETOPO1, or regional datasets.")
    
    # Create synthetic bathymetry for demonstration
    lats = np.linspace(-5, 5, 100)
    lons = np.linspace(90, 100, 100)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Synthetic depth with some features
    depth = -1000 - 3000 * np.exp(-((lon_grid - 95)**2 + (lat_grid - 0)**2) / 4)
    depth += 500 * np.sin(lon_grid * 0.5) * np.cos(lat_grid * 0.8)
    
    return lat_grid, lon_grid, depth

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

def load_swot_data(file_path):
    """Load SWOT data with proper variable handling."""
    try:
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
            return None
        
        # Get coordinates
        lat = ds['latitude'] if 'latitude' in ds.variables else ds['lat']
        lon = ds['longitude'] if 'longitude' in ds.variables else ds['lon']
        
        # Normalize longitude if needed
        if float(lon.max()) > 180.0:
            lon = ((lon + 180) % 360) - 180
        
        # Get quality flags if available
        qual_var = f"{ssh_var}_qual"
        qual = ds[qual_var] if qual_var in ds.variables else None
        
        return {
            'ssh': ds[ssh_var],
            'latitude': lat,
            'longitude': lon,
            'quality': qual,
            'time': ds['time'] if 'time' in ds.variables else None,
            'dataset': ds
        }
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def apply_high_pass_filter(data_2d, window_size_points=50, axis=0):
    """Apply high-pass filter along specified axis to isolate internal waves."""
    filtered_data = np.full_like(data_2d, np.nan)
    
    if axis == 0:  # Along-track filtering
        for j in range(data_2d.shape[1]):
            track = data_2d[:, j]
            valid_mask = ~np.isnan(track)
            if np.sum(valid_mask) > window_size_points * 2:
                # Apply Gaussian high-pass filter
                smoothed = gaussian_filter1d(track, sigma=window_size_points/3, mode='nearest')
                high_pass = track - smoothed
                filtered_data[:, j] = high_pass
    else:  # Across-track filtering
        for i in range(data_2d.shape[0]):
            track = data_2d[i, :]
            valid_mask = ~np.isnan(track)
            if np.sum(valid_mask) > window_size_points * 2:
                smoothed = gaussian_filter1d(track, sigma=window_size_points/3, mode='nearest')
                high_pass = track - smoothed
                filtered_data[i, :] = high_pass
    
    return filtered_data

def create_internal_wave_swath_plot(swot_data, high_pass_data, filename):
    """Create a comprehensive swath plot showing internal wave patterns."""
    
    ssh = swot_data['ssh'].values
    lat = swot_data['latitude'].values
    lon = swot_data['longitude'].values
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Original SSH swath
    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(ssh, cmap=config.wave_colormap, aspect='auto', 
                     extent=[0, ssh.shape[1], ssh.shape[0], 0])
    ax1.set_title(f'Original SSH/SSHA - {os.path.basename(filename)}', fontweight='bold')
    ax1.set_xlabel('Across-track pixel')
    ax1.set_ylabel('Along-track line')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('SSH/SSHA (m)')
    
    # 2. High-pass filtered (internal waves)
    ax2 = fig.add_subplot(gs[1, :2])
    im2 = ax2.imshow(high_pass_data, cmap=config.wave_colormap, aspect='auto',
                     extent=[0, ssh.shape[1], ssh.shape[0], 0])
    ax2.set_title('High-pass Filtered (Internal Waves)', fontweight='bold')
    ax2.set_xlabel('Across-track pixel')
    ax2.set_ylabel('Along-track line')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('High-pass SSH/SSHA (m)')
    
    # 3. Geographic context
    ax3 = fig.add_subplot(gs[0, 2])
    valid_mask = ~np.isnan(ssh) & ~np.isnan(lat) & ~np.isnan(lon)
    if np.any(valid_mask):
        scatter = ax3.scatter(lon[valid_mask], lat[valid_mask], 
                             c=ssh[valid_mask], cmap=config.wave_colormap, s=1, alpha=0.7)
        ax3.set_xlabel('Longitude (deg)')
        ax3.set_ylabel('Latitude (deg)')
        ax3.set_title('Geographic Coverage', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add bathymetry contours if available
        try:
            lat_bathy, lon_bathy, depth_bathy = load_bathymetry_data()
            cs = ax3.contour(lon_bathy, lat_bathy, depth_bathy, 
                           levels=[-3000, -2000, -1000, -500], colors='gray', alpha=0.5)
            ax3.clabel(cs, inline=True, fontsize=8)
        except:
            pass
    
    # 4. Along-track profile
    ax4 = fig.add_subplot(gs[1, 2])
    if ssh.shape[1] > 10:  # Ensure we have enough across-track pixels
        center_pixel = ssh.shape[1] // 2
        along_track_original = ssh[:, center_pixel]
        along_track_filtered = high_pass_data[:, center_pixel]
        
        valid_orig = ~np.isnan(along_track_original)
        valid_filt = ~np.isnan(along_track_filtered)
        
        if np.any(valid_orig):
            ax4.plot(np.where(valid_orig)[0], along_track_original[valid_orig], 
                    'b-', label='Original', alpha=0.7, linewidth=1)
        if np.any(valid_filt):
            ax4.plot(np.where(valid_filt)[0], along_track_filtered[valid_filt], 
                    'r-', label='High-pass', linewidth=1.5)
        
        ax4.set_xlabel('Along-track index')
        ax4.set_ylabel('SSH/SSHA (m)')
        ax4.set_title(f'Center Track Profile\n(Pixel {center_pixel})', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Wave statistics
    ax5 = fig.add_subplot(gs[2, :])
    
    # Calculate statistics along track
    wave_amplitude = []
    wave_wavelength = []
    track_indices = []
    
    for i in range(0, ssh.shape[0], 20):  # Sample every 20 points
        if i + 20 < ssh.shape[0]:
            segment = high_pass_data[i:i+20, :]
            if not np.all(np.isnan(segment)):
                # Calculate RMS amplitude
                rms_amp = np.sqrt(np.nanmean(segment**2))
                wave_amplitude.append(rms_amp)
                track_indices.append(i + 10)
                
                # Estimate dominant wavelength using autocorrelation
                center_track = segment[:, segment.shape[1]//2]
                if not np.all(np.isnan(center_track)):
                    try:
                        autocorr = np.correlate(center_track - np.nanmean(center_track), 
                                              center_track - np.nanmean(center_track), mode='full')
                        autocorr = autocorr[autocorr.size // 2:]
                        # Find first minimum after the zero lag
                        if len(autocorr) > 5:
                            min_idx = np.argmin(autocorr[1:5]) + 1
                            wavelength_points = min_idx * 2  # Half wavelength to full wavelength
                            wavelength_km = wavelength_points * config.along_track_resolution
                            wave_wavelength.append(wavelength_km)
                        else:
                            wave_wavelength.append(np.nan)
                    except:
                        wave_wavelength.append(np.nan)
                else:
                    wave_wavelength.append(np.nan)
    
    if wave_amplitude:
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(track_indices, wave_amplitude, 'r-o', markersize=4, 
                        label='Wave Amplitude (RMS)', linewidth=2)
        ax5.set_xlabel('Along-track index')
        ax5.set_ylabel('RMS Amplitude (m)', color='red')
        ax5.tick_params(axis='y', labelcolor='red')
        
        if any(~np.isnan(wave_wavelength)):
            line2 = ax5_twin.plot(track_indices, wave_wavelength, 'b-s', markersize=4, 
                                 label='Est. Wavelength (km)', linewidth=2)
            ax5_twin.set_ylabel('Wavelength (km)', color='blue')
            ax5_twin.tick_params(axis='y', labelcolor='blue')
        
        ax5.set_title('Internal Wave Characteristics Along Track', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax5.get_legend_handles_labels()
        if 'ax5_twin' in locals():
            lines2, labels2 = ax5_twin.get_legend_handles_labels()
            ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax5.legend()
    
    plt.suptitle(f'Internal Wave Analysis: {os.path.basename(filename)}', 
                 fontsize=16, fontweight='bold')
    
    if config.save_figures:
        if not os.path.exists(config.output_directory):
            os.makedirs(config.output_directory)
        
        timestamp = extract_timestamp_from_filename(filename)
        output_name = f'internal_waves_{timestamp}.png'
        output_path = os.path.join(config.output_directory, output_name)
        plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    return fig

def create_spectral_analysis_plot(swot_data, filename):
    """Create spectral analysis plot for internal wave frequency analysis."""
    
    ssh = swot_data['ssh'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Spectral Analysis: {os.path.basename(filename)}', fontsize=14, fontweight='bold')
    
    # 1. Along-track power spectral density
    if ssh.shape[1] > 10:
        center_pixel = ssh.shape[1] // 2
        along_track = ssh[:, center_pixel]
        valid_mask = ~np.isnan(along_track)
        
        if np.sum(valid_mask) > 50:
            clean_signal = along_track[valid_mask]
            clean_signal = clean_signal - np.mean(clean_signal)
            
            # Calculate PSD
            frequencies, psd = signal.periodogram(clean_signal, fs=1/config.along_track_resolution)
            
            # Convert to wavelength
            wavelengths = 1 / frequencies[1:]  # Skip zero frequency
            
            axes[0, 0].loglog(wavelengths, psd[1:], 'b-', alpha=0.8)
            axes[0, 0].set_xlabel('Wavelength (km)')
            axes[0, 0].set_ylabel('Power Spectral Density')
            axes[0, 0].set_title('Along-track PSD')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add internal wave band annotation
            axes[0, 0].axvspan(10, 100, alpha=0.2, color='red', label='Internal Wave Band')
            axes[0, 0].legend()
    
    # 2. Across-track analysis
    if ssh.shape[0] > 10:
        mid_line = ssh.shape[0] // 2
        across_track = ssh[mid_line, :]
        valid_mask = ~np.isnan(across_track)
        
        if np.sum(valid_mask) > 20:
            clean_signal = across_track[valid_mask]
            clean_signal = clean_signal - np.mean(clean_signal)
            
            frequencies, psd = signal.periodogram(clean_signal, fs=1/config.across_track_resolution)
            wavelengths = 1 / frequencies[1:]
            
            axes[0, 1].loglog(wavelengths, psd[1:], 'g-', alpha=0.8)
            axes[0, 1].set_xlabel('Wavelength (km)')
            axes[0, 1].set_ylabel('Power Spectral Density')
            axes[0, 1].set_title('Across-track PSD')
            axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 2D Spectral analysis (if enough data)
    try:
        valid_data = ssh[~np.isnan(ssh)]
        if len(valid_data) > 100:
            # Simple 2D FFT analysis
            ssh_filled = np.nan_to_num(ssh, nan=np.nanmean(ssh))
            fft_2d = np.fft.fft2(ssh_filled)
            power_2d = np.abs(fft_2d)**2
            
            # Plot 2D power spectrum
            im = axes[1, 0].imshow(np.log10(np.fft.fftshift(power_2d)), 
                                  cmap='viridis', aspect='auto')
            axes[1, 0].set_title('2D Power Spectrum (log scale)')
            axes[1, 0].set_xlabel('Across-track frequency')
            axes[1, 0].set_ylabel('Along-track frequency')
            plt.colorbar(im, ax=axes[1, 0])
    except:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor 2D analysis', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. Wave direction analysis
    axes[1, 1].text(0.5, 0.5, 'Wave Direction Analysis\n(Requires multiple passes)', 
                   ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Wave Propagation Direction')
    
    plt.tight_layout()
    
    if config.save_figures:
        timestamp = extract_timestamp_from_filename(filename)
        output_name = f'spectral_analysis_{timestamp}.png'
        output_path = os.path.join(config.output_directory, output_name)
        plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    return fig

def main():
    """Main function to generate internal wave visualizations."""
    print("SWOT Internal Wave Visualization Generator")
    print("=" * 50)
    
    # Get SWOT files
    nc_files = glob.glob(os.path.join(config.data_directory, "*.nc"))
    if not nc_files:
        print(f"No NetCDF files found in {config.data_directory}")
        return
    
    # Sort files chronologically
    file_timestamps = []
    for file_path in nc_files:
        filename = os.path.basename(file_path)
        timestamp_str = extract_timestamp_from_filename(filename)
        if timestamp_str:
            dt = parse_timestamp(timestamp_str)
            if dt:
                file_timestamps.append((dt, file_path, filename))
    
    file_timestamps.sort(key=lambda x: x[0])
    
    print(f"Found {len(file_timestamps)} SWOT files")
    
    # Process first few files for internal wave analysis
    max_files = min(3, len(file_timestamps))
    
    for i, (timestamp, file_path, filename) in enumerate(file_timestamps[:max_files]):
        print(f"\nProcessing {i+1}/{max_files}: {filename}")
        
        # Load SWOT data
        swot_data = load_swot_data(file_path)
        if swot_data is None:
            continue
        
        ssh = swot_data['ssh'].values
        
        # Apply high-pass filter
        print("  Applying high-pass filter for internal wave isolation...")
        window_points = int(config.high_pass_window / config.along_track_resolution)
        high_pass_data = apply_high_pass_filter(ssh, window_points, axis=0)
        
        # Create comprehensive swath plot
        print("  Creating internal wave swath visualization...")
        fig1 = create_internal_wave_swath_plot(swot_data, high_pass_data, filename)
        
        # Create spectral analysis plot
        print("  Creating spectral analysis plot...")
        fig2 = create_spectral_analysis_plot(swot_data, filename)
        
        # Close dataset
        swot_data['dataset'].close()
        
        print(f"  Completed analysis for {filename}")
    
    print(f"\nInternal wave visualization complete!")
    print(f"Figures saved to: {config.output_directory}")
    
    # Print recommendations for additional datasets
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR ENHANCED INTERNAL WAVE ANALYSIS")
    print("="*60)
    print("To create more comprehensive internal wave visualizations, consider adding:")
    print("\n1. BATHYMETRY DATA:")
    print("   - GEBCO global bathymetry (15 arc-second resolution)")
    print("   - ETOPO1 global relief model")
    print("   - Regional high-resolution bathymetry")
    print("   - Purpose: Show topographic context for wave generation")
    
    print("\n2. TIDAL DATA:")
    print("   - FES2014 global tidal atlas")
    print("   - TPXO tidal models")
    print("   - Regional tidal predictions")
    print("   - Purpose: Separate tidal and internal wave signals")
    
    print("\n3. OCEANOGRAPHIC CONTEXT:")
    print("   - Argo float profiles (temperature/salinity)")
    print("   - Ocean reanalysis data (HYCOM, GLORYS)")
    print("   - Buoyancy frequency profiles")
    print("   - Purpose: Calculate theoretical wave speeds and modes")
    
    print("\n4. METEOROLOGICAL DATA:")
    print("   - Wind stress from ERA5 or similar")
    print("   - Surface atmospheric pressure")
    print("   - Purpose: Identify forcing mechanisms")
    
    print("\n5. MULTIPLE SWOT PASSES:")
    print("   - Repeat pass data for same region")
    print("   - Crossing tracks for 2D analysis")
    print("   - Purpose: Track wave propagation and evolution")

if __name__ == "__main__":
    main()
