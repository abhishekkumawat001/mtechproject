#!/usr/bin/env python3
"""
GHRSST MUR JPL L4 SST Analysis and Visualization Script
========================================================
Product: MUR-JPL-L4-GLOB-v4.1 (Multi-scale Ultra-high Resolution SST)
Resolution: 0.01° global grid (~36000 x 18000 pixels per file)
Frequency: Daily files

This script:
  1. Interactively collects all configuration at runtime
  2. Explores the NetCDF dataset structure
  3. Creates monthly mean SST maps (12-panel figure)
  4. Creates daily SST maps for a user-specified date range

Author: Abhishek kumawat
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# INTERACTIVE INPUT HELPERS
# =============================================================================

def ask(prompt_text, default=None, cast=str, choices=None):
    """
    Prompt the user for input with an optional default and type cast.

    Parameters
    ----------
    prompt_text : str
    default : any
        Shown in brackets; returned if user presses Enter with no input.
    cast : callable
        Type to cast the answer to (e.g. int, float, str).
    choices : list or None
        If provided, only accept values in this list (case-insensitive for str).
    """
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {prompt_text}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        if raw == "":
            print("    (required — please enter a value)")
            continue
        try:
            value = cast(raw)
        except (ValueError, TypeError):
            print(f"    (invalid — expected {cast.__name__})")
            continue
        if choices is not None:
            check = value.lower() if isinstance(value, str) else value
            if check not in [c.lower() if isinstance(c, str) else c for c in choices]:
                print(f"    (choose from: {', '.join(str(c) for c in choices)})")
                continue
        return value


def ask_yes_no(prompt_text, default=True):
    """Ask a yes/no question, returns bool."""
    default_str = "y" if default else "n"
    raw = ask(prompt_text + " (y/n)", default=default_str, cast=str, choices=["y", "n"])
    return raw.lower() == "y"


def ask_path(prompt_text, must_exist=False):
    """Ask for a directory path, expanding ~ and env vars."""
    while True:
        raw = input(f"  {prompt_text}: ").strip().strip('"').strip("'")
        if not raw:
            print("    (required — please enter a path)")
            continue
        path = os.path.expandvars(os.path.expanduser(raw))
        if must_exist and not os.path.isdir(path):
            print(f"    (directory not found: {path})")
            continue
        return path


def ask_region():
    """
    Ask the user for a lat/lon bounding box or global.
    Returns [lon_min, lon_max, lat_min, lat_max] or None for global.
    """
    global_plot = ask_yes_no("Plot global domain (skip lat/lon input)?", default=False)
    if global_plot:
        return None

    print("  Enter bounding box (decimal degrees):")
    lon_min = ask("    Lon min (e.g. 40 for 40°E)", cast=float)
    lon_max = ask("    Lon max (e.g. 100 for 100°E)", cast=float)
    lat_min = ask("    Lat min (e.g. -10 for 10°S)", cast=float)
    lat_max = ask("    Lat max (e.g.  30 for 30°N)", cast=float)
    return [lon_min, lon_max, lat_min, lat_max]


def ask_year_selection(available_years):
    """
    Show available years and let the user choose one or all.
    Returns a list of ints.
    """
    years_str = ", ".join(str(y) for y in sorted(available_years))
    print(f"\n  Years found in dataset: {years_str}")
    print("  Enter a single year (e.g. 2024) or 'all' for every year found.")
    while True:
        raw = input("  Year(s) to analyze: ").strip()
        if raw.lower() == "all":
            return sorted(available_years)
        try:
            year = int(raw)
            if year in available_years:
                return [year]
            print(f"    (year {year} not in dataset — choose from: {years_str})")
        except ValueError:
            print("    (enter a 4-digit year or 'all')")


def ask_month():
    """Ask for a month number 1-12, return int."""
    while True:
        raw = input("  Month (1-12): ").strip()
        try:
            m = int(raw)
            if 1 <= m <= 12:
                return m
            print("    (must be 1–12)")
        except ValueError:
            print("    (enter a number 1–12)")


def collect_config(available_years):
    """
    Interactively collect all runtime settings.
    Returns a dict with all configuration values.
    """
    cfg = {}

    # ---- Directories ----
    print("\n--- Directories ---")
    cfg["data_dir"] = ask_path("Data directory (contains .nc files)", must_exist=True)
    cfg["output_dir"] = ask_path("Output directory (will be created if absent)")

    # ---- Region ----
    print("\n--- Region ---")
    cfg["region"] = ask_region()

    # ---- Year selection ----
    print("\n--- Year selection ---")
    cfg["years"] = ask_year_selection(available_years)

    # ---- Plot types ----
    print("\n--- Plot types ---")
    print("  [1] Monthly mean maps only  (12-panel, one per year)")
    print("  [2] Daily maps only         (specific month & day range)")
    print("  [3] Both monthly and daily")
    plot_choice = ask("Choice", default="3", cast=str, choices=["1", "2", "3"])
    cfg["do_monthly"] = plot_choice in ("1", "3")
    cfg["do_daily"]   = plot_choice in ("2", "3")

    # ---- Daily plot settings ----
    if cfg["do_daily"]:
        print("\n--- Daily plot settings ---")
        if len(cfg["years"]) > 1:
            print("  (daily plot will be generated for each selected year)")
        cfg["daily_month"]     = ask_month()
        cfg["daily_day_start"] = ask("Start day", default=1,  cast=int)
        cfg["daily_day_end"]   = ask("End day",   default=30, cast=int)
        cfg["days_per_row"]    = ask("Subplots per row", default=5, cast=int)

    # ---- Colormap / rendering ----
    print("\n--- Colormap & rendering (press Enter to accept defaults) ---")
    cfg["cmap"]       = ask("Colormap name",         default="RdYlBu_r")
    cfg["vmin"]       = ask("SST min (°C)",           default=-2.0, cast=float)
    cfg["vmax"]       = ask("SST max (°C)",           default=32.0, cast=float)
    cfg["downsample"] = ask("Downsample factor (1=full res, 5=fast)", default=5, cast=int)

    return cfg


# =============================================================================
# BLOCK 1: HELPER FUNCTIONS
# =============================================================================

def extract_date_from_filename(filepath):
    """Parse YYYYMMDD from the first 8 characters of the filename."""
    filename = os.path.basename(filepath)
    match = re.match(r'^(\d{8})', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d')
    raise ValueError(f"Cannot parse date from filename: {filename}")


def load_sst(filepath, region=None, downsample=1):
    """Load SST data from a NetCDF file, subset to region, return in °C."""
    ds = xr.open_dataset(filepath)
    sst = ds['analysed_sst'].squeeze('time')

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        sst = sst.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    sst = sst - 273.15  # Kelvin → Celsius

    if downsample > 1:
        sst = sst[::downsample, ::downsample]

    ds.close()
    return sst


def make_sst_map(ax, sst_data, title, vmin, vmax, cmap, show_cbar=False):
    """Draw an SST map on a Cartopy axis; returns the image handle."""
    lon = sst_data.coords['lon'].values
    lat = sst_data.coords['lat'].values

    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,         zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle='--', zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}

    img = ax.pcolormesh(lon, lat, sst_data.values,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='auto', zorder=0)

    ax.set_title(title, fontsize=10, fontweight='bold')

    if show_cbar:
        plt.colorbar(img, ax=ax, orientation='horizontal',
                     label='SST (°C)', pad=0.05, shrink=0.8)

    return img


def get_files_for_month(all_files, year, month):
    """Return {day: filepath} for files matching year and month."""
    month_files = {}
    for filepath in all_files:
        try:
            d = extract_date_from_filename(filepath)
            if d.year == year and d.month == month:
                month_files[d.day] = filepath
        except ValueError:
            continue
    return month_files


def scan_dataset(data_dir):
    """
    Scan data_dir for *.nc files, return (sorted file list, set of years).
    """
    pattern = os.path.join(data_dir, "*.nc")
    all_files = sorted(glob.glob(pattern))

    years = set()
    for f in all_files:
        try:
            years.add(extract_date_from_filename(f).year)
        except ValueError:
            pass

    return all_files, years


# =============================================================================
# BLOCK 2: DATA EXPLORER
# =============================================================================

def explore_dataset(all_files, years):
    """Print dataset summary (file count, date range, structure)."""
    print("\n" + "=" * 70)
    print("DATASET EXPLORATION")
    print("=" * 70)
    print(f"\n  Total .nc files found : {len(all_files)}")
    print(f"  Years available       : {', '.join(str(y) for y in sorted(years))}")

    if not all_files:
        return

    dates = []
    for f in all_files:
        try:
            dates.append(extract_date_from_filename(f))
        except ValueError:
            pass

    if dates:
        dates_sorted = sorted(dates)
        print(f"  Date range            : {dates_sorted[0]:%Y-%m-%d}  →  "
              f"{dates_sorted[-1]:%Y-%m-%d}")

    # Sample file structure
    print("\n  Sample file structure:")
    ds = xr.open_dataset(all_files[0])
    print(f"    File : {os.path.basename(all_files[0])}")
    for var_name in ds.data_vars:
        v = ds[var_name]
        print(f"    Var  : {var_name}  shape={v.shape}  "
              f"units={v.attrs.get('units','?')}")
    ds.close()


# =============================================================================
# BLOCK 3: MONTHLY AVERAGE PLOTS
# =============================================================================

def create_monthly_mean_plots(all_files, cfg, year):
    """
    Create a 3×4 panel figure of monthly mean SST for the given year.
    Saved to <output_dir>/SST_Monthly_Mean_<year>.png
    """
    print(f"\n{'='*70}")
    print(f"MONTHLY MEAN MAPS — {year}")
    print("=" * 70)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    img_handle = None

    for month_idx in range(12):
        month_num = month_idx + 1
        ax = axes[month_idx]
        print(f"  {month_name[month_num]:>12} {year} ... ", end="", flush=True)

        month_files = get_files_for_month(all_files, year, month_num)

        if not month_files:
            ax.set_title(f"{month_name[month_num]} (no data)", fontsize=10)
            ax.axis('off')
            print("no data")
            continue

        daily_sst_list = []
        for day, filepath in sorted(month_files.items()):
            try:
                daily_sst_list.append(load_sst(filepath, cfg["region"],
                                                cfg["downsample"]))
            except Exception as e:
                print(f"\n    Warning day {day}: {e}", end="")

        if not daily_sst_list:
            ax.set_title(f"{month_name[month_num]} (load error)", fontsize=10)
            ax.axis('off')
            print("load error")
            continue

        monthly_mean = xr.concat(daily_sst_list, dim='day').mean(dim='day')
        img_handle = make_sst_map(ax, monthly_mean, month_name[month_num],
                                   cfg["vmin"], cfg["vmax"], cfg["cmap"])

        if cfg["region"] is not None:
            ax.set_extent(cfg["region"], crs=ccrs.PlateCarree())

        print(f"done ({len(daily_sst_list)} days)")

    if img_handle is not None:
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = fig.colorbar(img_handle, cax=cbar_ax, orientation='horizontal',
                            extend='both')
        cbar.set_label('Sea Surface Temperature (°C)', fontsize=12)

    region_label = ("Global" if cfg["region"] is None
                    else (f"[{cfg['region'][0]}°–{cfg['region'][1]}°E, "
                          f"{cfg['region'][2]}°–{cfg['region'][3]}°N]"))
    fig.suptitle(f"MUR GHRSST — Monthly Mean SST (°C) | {year}\n{region_label}",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_path = os.path.join(cfg["output_dir"], f"SST_Monthly_Mean_{year}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out_path}")
    return out_path


# =============================================================================
# BLOCK 4: DAILY PLOTS
# =============================================================================

def create_daily_plots(all_files, cfg, year):
    """
    Create a panel figure of daily SST maps for the configured month & day range.
    Saved to <output_dir>/SST_Daily_<year>_<month>_Day<start>-<end>.png
    """
    month    = cfg["daily_month"]
    day_start = cfg["daily_day_start"]
    day_end   = cfg["daily_day_end"]

    print(f"\n{'='*70}")
    print(f"DAILY MAPS — {month_name[month]} {year}, "
          f"days {day_start}–{day_end}")
    print("=" * 70)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    month_files = get_files_for_month(all_files, year, month)
    days_to_plot = [(d, month_files[d])
                    for d in range(day_start, day_end + 1)
                    if d in month_files]

    if not days_to_plot:
        print(f"  No data files found for {month_name[month]} {year}, "
              f"days {day_start}–{day_end}")
        return None

    print(f"  Plotting {len(days_to_plot)} day(s)...")

    ncols = min(cfg["days_per_row"], len(days_to_plot))
    nrows = ceil(len(days_to_plot) / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4 * ncols, 3.5 * nrows + 1),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()
    img_handle = None

    for idx, (day, filepath) in enumerate(days_to_plot):
        ax = axes_flat[idx]
        date_str = f"{year}-{month:02d}-{day:02d}"
        print(f"    {date_str} ... ", end="", flush=True)
        try:
            sst = load_sst(filepath, cfg["region"], cfg["downsample"])
            img_handle = make_sst_map(ax, sst, date_str,
                                       cfg["vmin"], cfg["vmax"], cfg["cmap"])
            if cfg["region"] is not None:
                ax.set_extent(cfg["region"], crs=ccrs.PlateCarree())
            print("done")
        except Exception as e:
            ax.set_title(f"{date_str}\n(error)", fontsize=9)
            ax.axis('off')
            print(f"error: {e}")

    for idx in range(len(days_to_plot), len(axes_flat)):
        axes_flat[idx].axis('off')

    if img_handle is not None:
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.015])
        cbar = fig.colorbar(img_handle, cax=cbar_ax, orientation='horizontal',
                            extend='both')
        cbar.set_label('Sea Surface Temperature (°C)', fontsize=11)

    region_label = ("Global" if cfg["region"] is None
                    else (f"[{cfg['region'][0]}°–{cfg['region'][1]}°E, "
                          f"{cfg['region'][2]}°–{cfg['region'][3]}°N]"))
    fig.suptitle(f"MUR GHRSST — Daily SST (°C) | {month_name[month]} {year} "
                 f"(Day {day_start}–{day_end})\n{region_label}",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    fname = (f"SST_Daily_{year}_{month:02d}"
             f"_Day{day_start:02d}-{day_end:02d}.png")
    out_path = os.path.join(cfg["output_dir"], fname)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out_path}")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  GHRSST MUR JPL L4 SST ANALYSIS  —  Interactive Mode")
    print("=" * 70)

    # ---- Quick scan to discover years before asking full config ----
    print("\nEnter the data directory to scan for available years first.")
    data_dir = ask_path("Data directory (contains .nc files)", must_exist=True)

    print("\nScanning dataset...")
    all_files, available_years = scan_dataset(data_dir)

    if not all_files:
        print(f"\nERROR: No .nc files found in: {data_dir}")
        return

    explore_dataset(all_files, available_years)

    # ---- Collect the rest of the config ----
    # Override data_dir with the one already entered above
    cfg = collect_config(available_years)
    cfg["data_dir"] = data_dir          # already entered, skip re-asking

    print("\n" + "=" * 70)
    print("STARTING ANALYSIS")
    print("=" * 70)
    region_label = ("Global" if cfg["region"] is None
                    else str(cfg["region"]))
    print(f"  Data dir   : {cfg['data_dir']}")
    print(f"  Output dir : {cfg['output_dir']}")
    print(f"  Region     : {region_label}")
    print(f"  Year(s)    : {cfg['years']}")
    print(f"  Monthly    : {cfg['do_monthly']}")
    print(f"  Daily      : {cfg['do_daily']}")
    if cfg["do_daily"]:
        print(f"  Daily month: {month_name[cfg['daily_month']]}  "
              f"days {cfg['daily_day_start']}–{cfg['daily_day_end']}")

    saved_files = []

    for year in cfg["years"]:
        if cfg["do_monthly"]:
            path = create_monthly_mean_plots(all_files, cfg, year)
            if path:
                saved_files.append(path)

        if cfg["do_daily"]:
            path = create_daily_plots(all_files, cfg, year)
            if path:
                saved_files.append(path)

    print("\n" + "=" * 70)
    print("DONE — output files:")
    for p in saved_files:
        print(f"  {p}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
