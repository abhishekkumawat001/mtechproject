#!/usr/bin/env python3
"""
GHRSST MUR JPL L4 SST Analysis and Visualization Script
========================================================
Product: MUR-JPL-L4-GLOB-v4.1 (Multi-scale Ultra-high Resolution SST)
Resolution: 0.01° global grid (~36000 x 18000 pixels per file)
Frequency: Daily files

This script:
  1. Explores the NetCDF dataset structure
  2. Creates monthly mean SST maps (12-panel figure)
  3. Creates daily SST maps for a user-specified date range

Author: Generated for geospatial SST analysis
"""

import os
import re
import glob
import warnings
from datetime import datetime
from math import ceil
from calendar import month_name

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIGURATION BLOCK (User edits only here)
# =============================================================================

# Paths
DATA_DIR = r"C:\Users\abhik\Downloads\New folder\MUR-JPL-L4-GLOB-v4.1_4.1-20260316_132446"
OUTPUT_DIR = r"C:\Users\abhik\Downloads\New folder\SST_Plots"

# Region of interest: [lon_min, lon_max, lat_min, lat_max] or None for global
# Example: [40, 100, -10, 30] for Indian Ocean
REGION = [40, 100, -10, 30]  # Set to None for global view

# Year to analyze
YEAR = 2024

# Colormap settings
SST_CMAP = "RdYlBu_r"  # Colormap for SST (red=warm, blue=cold)
SST_VMIN = -2.0        # Minimum SST in °C
SST_VMAX = 32.0        # Maximum SST in °C

# Performance: spatial downsampling factor (1=full resolution, 5=fast, 10=very fast)
DOWNSAMPLE = 5

# Daily plot configuration
DAILY_MONTH = 11       # Month number (1-12) for daily visualization
DAILY_DAY_START = 1    # Start day of range
DAILY_DAY_END = 30     # End day of range
DAYS_PER_ROW = 5       # Number of subplot columns in daily panel figure

# =============================================================================
# BLOCK 1: HELPER FUNCTIONS
# =============================================================================

def extract_date_from_filename(filepath):
    """
    Parse YYYYMMDD from the first 8 characters of the filename.

    Parameters
    ----------
    filepath : str
        Full path to the NetCDF file

    Returns
    -------
    datetime
        Python datetime object representing the file date
    """
    filename = os.path.basename(filepath)
    match = re.match(r'^(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    else:
        raise ValueError(f"Cannot parse date from filename: {filename}")


def load_sst(filepath, region=None, downsample=1):
    """
    Load SST data from a NetCDF file with optional region subsetting and downsampling.

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    region : list or None
        [lon_min, lon_max, lat_min, lat_max] for subsetting, or None for full domain
    downsample : int
        Spatial subsampling factor (1=full resolution)

    Returns
    -------
    xr.DataArray
        SST data in degrees Celsius, with time dimension squeezed
    """
    ds = xr.open_dataset(filepath)

    # Extract SST and squeeze the time dimension (always size 1)
    sst = ds['analysed_sst'].squeeze('time')

    # Subset to region if specified
    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        sst = sst.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    # Convert Kelvin to Celsius
    sst = sst - 273.15

    # Apply spatial downsampling
    if downsample > 1:
        sst = sst[::downsample, ::downsample]

    # Close the dataset to free memory
    ds.close()

    return sst


def make_sst_map(ax, sst_data, title, vmin, vmax, cmap, show_cbar=False):
    """
    Create an SST map on a Cartopy axis with coastlines, borders, and gridlines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cartopy GeoAxes object
    sst_data : xr.DataArray
        SST data to plot (must have lat/lon coordinates)
    title : str
        Title for the subplot
    vmin, vmax : float
        Colorbar range (min/max SST in °C)
    cmap : str
        Matplotlib colormap name
    show_cbar : bool
        Whether to add an individual colorbar (default False for shared colorbar)

    Returns
    -------
    matplotlib.collections.QuadMesh
        The pcolormesh image handle (useful for shared colorbar)
    """
    # Get coordinate arrays
    lon = sst_data.coords['lon'].values
    lat = sst_data.coords['lat'].values

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--', zorder=2)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}

    # Plot SST data
    img = ax.pcolormesh(lon, lat, sst_data.values,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='auto', zorder=0)

    # Set title
    ax.set_title(title, fontsize=10, fontweight='bold')

    # Optional individual colorbar
    if show_cbar:
        plt.colorbar(img, ax=ax, orientation='horizontal',
                     label='SST (°C)', pad=0.05, shrink=0.8)

    return img


def get_files_for_month(all_files, year, month):
    """
    Filter files to get those matching a specific year and month.

    Parameters
    ----------
    all_files : list
        List of all NetCDF file paths
    year : int
        Year to filter (e.g., 2024)
    month : int
        Month to filter (1-12)

    Returns
    -------
    dict
        Dictionary mapping day number (int) to filepath
    """
    month_files = {}
    for filepath in all_files:
        try:
            file_date = extract_date_from_filename(filepath)
            if file_date.year == year and file_date.month == month:
                month_files[file_date.day] = filepath
        except ValueError:
            continue
    return month_files


# =============================================================================
# BLOCK 2: DATA EXPLORER
# =============================================================================

def explore_dataset(data_dir):
    """
    Explore the GHRSST MUR dataset: file count, date range, and structure.

    Parameters
    ----------
    data_dir : str
        Directory containing the NetCDF files

    Returns
    -------
    list
        Sorted list of all NetCDF file paths
    """
    print("=" * 70)
    print("BLOCK 2: DATA EXPLORATION")
    print("=" * 70)

    # Find all NetCDF files
    pattern = os.path.join(data_dir, "*.nc")
    all_files = sorted(glob.glob(pattern))

    print(f"\nTotal files found: {len(all_files)}")

    if len(all_files) == 0:
        print("ERROR: No .nc files found in the specified directory!")
        print(f"Directory: {data_dir}")
        return []

    # Parse dates from filenames
    dates = []
    for f in all_files:
        try:
            dates.append(extract_date_from_filename(f))
        except ValueError:
            pass

    if dates:
        dates_sorted = sorted(dates)
        print(f"Date range: {dates_sorted[0].strftime('%Y-%m-%d')} → "
              f"{dates_sorted[-1].strftime('%Y-%m-%d')}")

        # List available months
        months_avail = sorted(set((d.year, d.month) for d in dates))
        print(f"Months available ({len(months_avail)} total):")
        for y, m in months_avail:
            count = sum(1 for d in dates if d.year == y and d.month == m)
            print(f"  {y}-{m:02d}: {count} files")

    # Open one sample file to explore structure
    print("\n" + "-" * 50)
    print("SAMPLE FILE STRUCTURE")
    print("-" * 50)

    sample_file = all_files[0]
    print(f"\nSample file: {os.path.basename(sample_file)}\n")

    ds = xr.open_dataset(sample_file)
    print(ds)

    print("\n" + "-" * 50)
    print("VARIABLE DETAILS")
    print("-" * 50)
    for var_name in ds.data_vars:
        var = ds[var_name]
        units = var.attrs.get('units', 'N/A')
        long_name = var.attrs.get('long_name', 'N/A')
        print(f"\n  {var_name}:")
        print(f"    Units: {units}")
        print(f"    Long name: {long_name}")
        print(f"    Shape: {var.shape}")

    ds.close()
    print("\n")

    return all_files


# =============================================================================
# BLOCK 3: MONTHLY AVERAGE PLOTS
# =============================================================================

def create_monthly_mean_plots(all_files, output_dir, year, region, downsample,
                               vmin, vmax, cmap):
    """
    Create a 3x4 panel figure showing monthly mean SST for each month.

    Parameters
    ----------
    all_files : list
        List of all NetCDF file paths
    output_dir : str
        Directory to save the output figure
    year : int
        Year to process
    region : list or None
        [lon_min, lon_max, lat_min, lat_max] or None for global
    downsample : int
        Spatial downsampling factor
    vmin, vmax : float
        Colorbar limits
    cmap : str
        Colormap name

    Returns
    -------
    str
        Path to the saved figure
    """
    print("=" * 70)
    print("BLOCK 3: MONTHLY AVERAGE PLOTS")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with 3x4 subplot grid
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    img_handle = None
    files_processed = 0

    # Process each month
    for month_idx in range(12):
        month_num = month_idx + 1
        ax = axes[month_idx]

        print(f"Processing {month_name[month_num]} {year}...", end=" ")

        # Get files for this month
        month_files = get_files_for_month(all_files, year, month_num)

        if len(month_files) == 0:
            # No data for this month
            ax.set_title(f"{month_name[month_num]} (no data)", fontsize=10)
            ax.axis('off')
            print("no data")
            continue

        # Load all daily SST arrays for this month
        daily_sst_list = []
        for day, filepath in sorted(month_files.items()):
            try:
                sst = load_sst(filepath, region, downsample)
                daily_sst_list.append(sst)
                files_processed += 1
            except Exception as e:
                print(f"\n    Warning: Failed to load day {day}: {e}")

        if len(daily_sst_list) == 0:
            ax.set_title(f"{month_name[month_num]} (load error)", fontsize=10)
            ax.axis('off')
            print("load error")
            continue

        # Stack and compute monthly mean
        monthly_stack = xr.concat(daily_sst_list, dim='day')
        monthly_mean = monthly_stack.mean(dim='day')

        # Plot
        img_handle = make_sst_map(ax, monthly_mean, month_name[month_num],
                                   vmin, vmax, cmap, show_cbar=False)

        # Set extent if region is specified
        if region is not None:
            ax.set_extent(region, crs=ccrs.PlateCarree())

        print(f"done ({len(daily_sst_list)} days)")

    # Add shared colorbar at bottom
    if img_handle is not None:
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = fig.colorbar(img_handle, cax=cbar_ax, orientation='horizontal',
                            extend='both')
        cbar.set_label('Sea Surface Temperature (°C)', fontsize=12)

    # Main title
    region_str = "Global" if region is None else f"Region [{region[0]}°-{region[1]}°E, {region[2]}°-{region[3]}°N]"
    fig.suptitle(f"MUR GHRSST — Monthly Mean SST (°C) | {year}\n{region_str}",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    # Save figure
    output_path = os.path.join(output_dir, f"SST_Monthly_Mean_{year}.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\nMonthly plot saved: {output_path}")
    print(f"Total files processed: {files_processed}")

    return output_path


# =============================================================================
# BLOCK 4: DAILY PLOTS (Flexible Date Range)
# =============================================================================

def create_daily_plots(all_files, output_dir, year, month, day_start, day_end,
                       days_per_row, region, downsample, vmin, vmax, cmap):
    """
    Create a panel figure showing daily SST maps for a specified date range.

    Parameters
    ----------
    all_files : list
        List of all NetCDF file paths
    output_dir : str
        Directory to save the output figure
    year : int
        Year to process
    month : int
        Month number (1-12)
    day_start, day_end : int
        Day range (inclusive)
    days_per_row : int
        Number of subplot columns
    region : list or None
        [lon_min, lon_max, lat_min, lat_max] or None for global
    downsample : int
        Spatial downsampling factor
    vmin, vmax : float
        Colorbar limits
    cmap : str
        Colormap name

    Returns
    -------
    str
        Path to the saved figure
    """
    print("\n" + "=" * 70)
    print("BLOCK 4: DAILY PLOTS")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get files for the specified month
    month_files = get_files_for_month(all_files, year, month)

    # Filter to the requested day range
    days_to_plot = []
    for day in range(day_start, day_end + 1):
        if day in month_files:
            days_to_plot.append((day, month_files[day]))

    if len(days_to_plot) == 0:
        print(f"ERROR: No data files found for {month_name[month]} {year}, "
              f"days {day_start}-{day_end}")
        return None

    print(f"\nPlotting {len(days_to_plot)} days for {month_name[month]} {year}")
    print(f"Day range: {day_start} - {day_end}")

    # Compute subplot grid dimensions
    n_days = len(days_to_plot)
    ncols = min(days_per_row, n_days)
    nrows = ceil(n_days / ncols)

    # Create figure
    fig_width = 4 * ncols
    fig_height = 3.5 * nrows + 1  # Extra space for colorbar

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, fig_height),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Handle single row/column cases
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    img_handle = None

    # Plot each day
    for idx, (day, filepath) in enumerate(days_to_plot):
        ax = axes_flat[idx]

        date_str = f"{year}-{month:02d}-{day:02d}"
        print(f"  Plotting {month_name[month]} {day:02d}...", end=" ")

        try:
            sst = load_sst(filepath, region, downsample)
            img_handle = make_sst_map(ax, sst, date_str, vmin, vmax, cmap,
                                       show_cbar=False)

            # Set extent if region is specified
            if region is not None:
                ax.set_extent(region, crs=ccrs.PlateCarree())

            print("done")

        except Exception as e:
            ax.set_title(f"{date_str}\n(error)", fontsize=9)
            ax.axis('off')
            print(f"error: {e}")

    # Hide unused axes
    for idx in range(n_days, len(axes_flat)):
        axes_flat[idx].axis('off')

    # Add shared colorbar at bottom
    if img_handle is not None:
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.015])
        cbar = fig.colorbar(img_handle, cax=cbar_ax, orientation='horizontal',
                            extend='both')
        cbar.set_label('Sea Surface Temperature (°C)', fontsize=11)

    # Main title
    region_str = "Global" if region is None else f"Region [{region[0]}°-{region[1]}°E, {region[2]}°-{region[3]}°N]"
    fig.suptitle(f"MUR GHRSST — Daily SST (°C) | {month_name[month]} {year} "
                 f"(Day {day_start}–{day_end})\n{region_str}",
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    # Save figure
    output_filename = f"SST_Daily_{year}_{month:02d}_Day{day_start:02d}-{day_end:02d}.png"
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\nDaily plot saved: {output_path}")

    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for the GHRSST MUR SST analysis script."""

    print("\n" + "=" * 70)
    print("GHRSST MUR JPL L4 SST ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Region: {'Global' if REGION is None else REGION}")
    print(f"  Year: {YEAR}")
    print(f"  Downsample factor: {DOWNSAMPLE}")
    print(f"  SST range: {SST_VMIN}°C to {SST_VMAX}°C")
    print(f"  Colormap: {SST_CMAP}")
    print(f"  Daily plot: {month_name[DAILY_MONTH]} {YEAR}, "
          f"days {DAILY_DAY_START}-{DAILY_DAY_END}")
    print()

    # Block 2: Explore dataset
    all_files = explore_dataset(DATA_DIR)

    if len(all_files) == 0:
        print("Exiting: No data files found.")
        return

    # Block 3: Monthly mean plots
    monthly_plot_path = create_monthly_mean_plots(
        all_files=all_files,
        output_dir=OUTPUT_DIR,
        year=YEAR,
        region=REGION,
        downsample=DOWNSAMPLE,
        vmin=SST_VMIN,
        vmax=SST_VMAX,
        cmap=SST_CMAP
    )

    # Block 4: Daily plots
    daily_plot_path = create_daily_plots(
        all_files=all_files,
        output_dir=OUTPUT_DIR,
        year=YEAR,
        month=DAILY_MONTH,
        day_start=DAILY_DAY_START,
        day_end=DAILY_DAY_END,
        days_per_row=DAYS_PER_ROW,
        region=REGION,
        downsample=DOWNSAMPLE,
        vmin=SST_VMIN,
        vmax=SST_VMAX,
        cmap=SST_CMAP
    )

    # Final summary
    print("\n" + "=" * 70)
    print("OUTPUT SUMMARY")
    print("=" * 70)
    print(f"\nMonthly mean plot: {monthly_plot_path}")
    print(f"Daily plot: {daily_plot_path}")
    print(f"Total files in dataset: {len(all_files)}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
