import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
import json
import os
import glob
import re
import xarray as xr
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
SST_DATA_DIR = r"C:\Users\abhik\Downloads\New_folder\MUR-JPL-L4-GLOB-v4.1_4.1-20260316_132446"
MSLA_ROOT_DIR = r"C:\Users\abhik\Downloads\SEALEVEL_GLO_PHY_L4_MY_008_047_cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_2024"

# SST: 0.05°/pixel ≈ 5.55 km/pixel at 19°N
FILTER_SCALES = {
    'submeso': 0.2,   # ~22 km
    'meso':    0.8,   # ~89 km
    'large':   2.5,   # ~278 km
}

# MSLA: 0.125°/pixel ≈ 13.9 km/pixel at 19°N
FILTER_SCALES_MSLA = {
    'submeso': 0.5,   # ~56 km
    'meso':    1.5,   # ~167 km
    'large':   3.5,   # ~389 km
}

DS_MSLA = None # Global cache for MSLA dataset

# =============================================================================
# Helper – get peak date (same as previous implementations)
# =============================================================================

def _get_peak_date(cyc):
    """Return a datetime.date representing the cyclone's peak."""
    dates = cyc.get('dates', {})
    if dates.get('peak'):
        return pd.to_datetime(dates['peak']).date()
    if dates.get('formation') and dates.get('dissipation'):
        t0 = pd.to_datetime(dates['formation'])
        t1 = pd.to_datetime(dates['dissipation'])
        return (t0 + (t1 - t0) / 2).date()
    windows = _get_analysis_window_ranges(cyc)
    if windows:
        _, s, e = windows[-1]
        t0, t1 = pd.to_datetime(s), pd.to_datetime(e)
        return (t0 + (t1 - t0) / 2).date()
    return None

def _make_square_bbox(bbox, track):
    """Ensure bbox is square and centered around the track."""
    lons = [t[0] for t in track]
    lats = [t[1] for t in track]
    lon_min, lon_max = min(lons) - 2, max(lons) + 2
    lat_min, lat_max = min(lats) - 2, max(lats) + 2
    lon_cent = (lon_min + lon_max) / 2
    lat_cent = (lat_min + lat_max) / 2
    hw = max((lon_max - lon_min), (lat_max - lat_min)) / 2
    return {'lon_min': lon_cent - hw, 'lon_max': lon_cent + hw,
            'lat_min': lat_cent - hw, 'lat_max': lat_cent + hw}

def _get_analysis_window_ranges(cyc):
    """Extract pre, during, post windows from cyclone dict."""
    ad = cyc.get('analysis_dates', {})
    if ad:
        return [(k, v[0], v[1]) for k, v in ad.items()]
    
    from datetime import timedelta
    import pandas as pd
    dates = cyc.get('dates', {})
    if not dates.get('formation') or not dates.get('dissipation'):
        return []
    fmt = '%Y-%m-%d'
    formation = pd.to_datetime(dates['formation'])
    pre_start = (formation - timedelta(days=5)).strftime(fmt)
    pre_end = (formation - timedelta(days=1)).strftime(fmt)
    return [('pre', pre_start, pre_end), ('during', dates['formation'], dates['dissipation'])]

def _get_analysis_date_list_from_range(start_str, end_str):
    """Generate list of dates between start and end."""
    return pd.date_range(start_str, end_str).tolist()

# =============================================================================
# DATA LOADING & DECOMPOSITION
# =============================================================================

def load_sst_for_date(date, region):
    if isinstance(date, pd.Timestamp): date = date.to_pydatetime()
    date_str = date.strftime('%Y%m%d')
    pattern = os.path.join(SST_DATA_DIR, f"{date_str}*.nc")
    files = glob.glob(pattern)
    if not files: return None
    ds = xr.open_dataset(files[0])
    sst = ds['analysed_sst'].squeeze('time').sel(
        lon=slice(region['lon_min'], region['lon_max']),
        lat=slice(region['lat_min'], region['lat_max'])
    ) - 273.15
    ds.close()
    return sst

def load_msla_dataset(bbox):
    pattern = os.path.join(MSLA_ROOT_DIR, "*.nc")
    file_list = sorted(glob.glob(pattern))
    datasets = []
    for fp in file_list:
        d = xr.open_dataset(fp).sel(
            latitude=slice(bbox['lat_min'], bbox['lat_max']),
            longitude=slice(bbox['lon_min'], bbox['lon_max'])
        ).load()
        datasets.append(d)
    if not datasets:
        raise ValueError(f"No MSLA files found in {MSLA_ROOT_DIR}")
    return xr.concat(datasets, dim="time")

def load_msla_for_date(date, bbox):
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    date_str = date.strftime('%Y%m%d')
    pattern = os.path.join(MSLA_ROOT_DIR, f'*_{date_str}_*.nc')
    files = glob.glob(pattern)
    if not files:
        return None
    ds = xr.open_dataset(files[0])
    sla = ds['sla'].squeeze('time', drop=True).sel(
        latitude=slice(bbox['lat_min'], bbox['lat_max']), 
        longitude=slice(bbox['lon_min'], bbox['lon_max'])
    )
    # Don't close immediately if you need later? Xarray handles it
    return sla
def scale_decomposition(field_2d, grid_res, filter_scales, ref_lat=19.0):
    data = field_2d.values.astype(float).copy() if hasattr(field_2d, 'values') else np.asarray(field_2d).copy()
    mask = np.isnan(data)
    data_filled = np.where(mask, np.nanmean(data), data)
    dlat_km = 111.0 * grid_res
    dlon_km = 111.0 * np.cos(np.deg2rad(ref_lat)) * grid_res
    def _sig(scale_deg):
        skm = 111.0 * scale_deg
        return (skm / dlat_km, skm / dlon_km)
    s_sub, s_meso, s_large = _sig(filter_scales['submeso']), _sig(filter_scales['meso']), _sig(filter_scales['large'])
    meso_lp = gaussian_filter(data_filled, sigma=s_meso)
    large_lp = gaussian_filter(data_filled, sigma=s_large)
    return {'large_scale': np.where(mask, np.nan, large_lp),
            'meso': np.where(mask, np.nan, meso_lp - large_lp),
            'submeso': np.where(mask, np.nan, data_filled - meso_lp)}

def decompose_sst_field(sst_data):
    return scale_decomposition(sst_data, 0.05, FILTER_SCALES, float(sst_data.lat.mean()))

def decompose_msla_field(msla_data):
    return scale_decomposition(msla_data, 0.125, FILTER_SCALES_MSLA, float(msla_data.latitude.mean()))

# =============================================================================
# MULTI‑SCALE ANOMALY CALCULATION (Baseline-subtracted)
# =============================================================================

def calculate_multiscale_anomalies(cyclones_dict,
                                   output_csv='cyclone_multiscale_anomalies.csv'):
    """Compute mean SST & MSLA anomalies for each spatial scale.
    For Large-scale, it subtracts a pre-cyclone baseline to get the true anomaly.
    """
    records = []

    for cyc_name, cyc in cyclones_dict.items():
        print(f"Processing {cyc_name} ...", end=' ', flush=True)

        bbox = cyc.get('bbox')
        track = cyc.get('track')
        if not bbox or not track:
            print("skipped (no bbox/track)")
            continue

        # make sure bbox is square and covers the whole track
        bbox = _make_square_bbox(bbox, track)
        peak_date = _get_peak_date(cyc)
        if peak_date is None:
            print("skipped (no peak date)")
            continue

        windows = _get_analysis_window_ranges(cyc)
        if not windows:
            print("skipped (no analysis window)")
            continue

        # --- 1. Compute Baselines (from Pre window) ---
        pre_win = next((w for w in windows if w[0].lower() == 'pre'), windows[0])
        _, pre_start, pre_end = pre_win
        pre_dates = _get_analysis_date_list_from_range(pre_start, pre_end)
        
        pre_sst_large, pre_msla_large = [], []
        for d in pre_dates:
            s = load_sst_for_date(d, bbox)
            if s is not None:
                dec = decompose_sst_field(s)
                if dec and 'large_scale' in dec: pre_sst_large.append(np.nanmean(dec['large_scale']))
            m = load_msla_for_date(d, bbox)
            if m is not None:
                dec = decompose_msla_field(m)
                if dec and 'large_scale' in dec: pre_msla_large.append(np.nanmean(dec['large_scale']))
        
        sst_base = np.nanmean(pre_sst_large) if pre_sst_large else np.nan
        msla_base = np.nanmean(pre_msla_large) if pre_msla_large else np.nan
        
        if np.isnan(sst_base): print(f"Warning: No pre-cyclone SST data for {cyc_name}. Large-scale will be absolute.")
        if np.isnan(msla_base): print(f"Warning: No pre-cyclone MSLA data for {cyc_name}. Large-scale will be absolute.")
        
        sst_base = sst_base if not np.isnan(sst_base) else 0
        msla_base = msla_base if not np.isnan(msla_base) else 0

        # --- 2. Compute Target Signal (from During/Post window) ---
        _, start_str, end_str = windows[-1]
        date_list = _get_analysis_date_list_from_range(start_str, end_str)

        # containers for daily means per scale
        sst_large_vals, sst_meso_vals, sst_sub_vals = [], [], []
        msla_large_vals, msla_meso_vals, msla_sub_vals = [], [], []

        for date in date_list:
            # ----- SST -----
            sst_raw = load_sst_for_date(date, bbox)
            if sst_raw is not None:
                dec = decompose_sst_field(sst_raw)
                if dec:
                    if 'large_scale' in dec:
                        sst_large_vals.append(np.nanmean(dec['large_scale']))
                    if 'meso' in dec:
                        sst_meso_vals.append(np.nanmean(dec['meso']))
                    if 'submeso' in dec:
                        sst_sub_vals.append(np.nanmean(dec['submeso']))

            # ----- MSLA -----
            msla_raw = load_msla_for_date(date, bbox)
            if msla_raw is not None:
                dec = decompose_msla_field(msla_raw)
                if dec:
                    if 'large_scale' in dec:
                        msla_large_vals.append(np.nanmean(dec['large_scale']))
                    if 'meso' in dec:
                        msla_meso_vals.append(np.nanmean(dec['meso']))
                    if 'submeso' in dec:
                        msla_sub_vals.append(np.nanmean(dec['submeso']))

        # ---- final mean per scale (relative to baseline for Large scale) ----
        rec = {
            'Cyclone': cyc_name,
            'Peak_Date': str(peak_date),
            'SST_Large': np.nanmean(sst_large_vals) - sst_base if sst_large_vals else np.nan,
            'SST_Meso': np.nanmean(sst_meso_vals),
            'SST_Submeso': np.nanmean(sst_sub_vals),
            'MSLA_Large': np.nanmean(msla_large_vals) - msla_base if msla_large_vals else np.nan,
            'MSLA_Meso': np.nanmean(msla_meso_vals),
            'MSLA_Submeso': np.nanmean(msla_sub_vals),
        }
        print(f"SST_L={rec['SST_Large']:.3f} M={rec['SST_Meso']:.3f} Sub={rec['SST_Submeso']:.3f}")
        records.append(rec)

    df = pd.DataFrame(records)
    if not df.empty:
        df['Peak_Date'] = pd.to_datetime(df['Peak_Date'])
        df.to_csv(output_csv, index=False)
        print(f"\nSaved → {output_csv}")
    return df

# =============================================================================
# PLOT helpers – now using dates on the x‑axis
# =============================================================================

def _bar_plot_per_date(df, col, title, y_label, save_name, save=True):
    """Generic bar plot where the x‑axis is the peak date.
    Positive values are coloured warm/red, negatives cold/blue.
    """
    # keep only rows with both the anomaly and a peak date
    valid = df.dropna(subset=[col, 'Peak_Date']).copy()
    if valid.empty:
        print(f"No data for {col}")
        return
    # ensure dates are pandas Timestamp objects
    valid['Peak_Date'] = pd.to_datetime(valid['Peak_Date'])
    # sort chronologically
    valid = valid.sort_values('Peak_Date')

    colors = ['#d62728' if v >= 0 else '#1f77b4' for v in valid[col]]

    # Optimized dimensions for visibility and layout stability
    fig_width = min(20, max(12, len(valid) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    # width=0.7 to give some breathing room between bars
    bars = ax.bar(valid['Peak_Date'], valid[col], color=colors, 
                  edgecolor=None, linewidth=0, width=0.7)
    ax.axhline(0, color='gray', linewidth=0.8, zorder=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Peak Date', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    # strictly show Jan 2024 to Dec 2024
    ax.set_xlim(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=45)
    ax.grid(axis='y', alpha=0.3)
    # annotate each bar with cyclone name
    ymin, ymax = valid[col].min(), valid[col].max()
    offset = 0.02 * (ymax - ymin) if ymax != ymin else 0.01
    for bar, name in zip(bars, valid['Cyclone']):
        height = bar.get_height()
        if height >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    name,
                    ha='right', va='bottom', fontsize=6,
                    rotation=45, rotation_mode='anchor', color='#333333')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height - offset,
                    name,
                    ha='right', va='top', fontsize=6,
                    rotation=45, rotation_mode='anchor', color='#333333')
    
    # Manual adjustment to prevent "Tight layout" warnings and excessive whitespace
    plt.subplots_adjust(bottom=0.25, top=0.9, left=0.08, right=0.98)
    if save:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()

def plot_sst_scales(df, save=True):
    """Three bar plots – one per SST scale, x‑axis = peak date."""
    scales = [
        ('SST_Large',   'Large‑Scale SST Anomaly'),
        ('SST_Meso',    'Mesoscale SST Anomaly'),
        ('SST_Submeso', 'Sub‑mesoscale SST Anomaly')
    ]
    for col, title in scales:
        _bar_plot_per_date(df, col, title, 'SST Anomaly (°C)',
                           f"{col}_vs_Date.png", save)

def plot_msla_scales(df, save=True):
    """Three bar plots – one per MSLA scale, x‑axis = peak date."""
    scales = [
        ('MSLA_Large',   'Large‑Scale MSLA Anomaly'),
        ('MSLA_Meso',    'Mesoscale MSLA Anomaly'),
        ('MSLA_Submeso', 'Sub‑mesoscale MSLA Anomaly')
    ]
    for col, title in scales:
        _bar_plot_per_date(df, col, title, 'MSLA Anomaly (m)',
                           f"{col}_vs_Date.png", save)

# =============================================================================
# SEASONAL CLASSIFICATION
# =============================================================================

SEASON_MAP = {
    1: 'JFM', 2: 'JFM', 3: 'JFM',       # Winter
    4: 'AMJ', 5: 'AMJ', 6: 'AMJ',        # Pre-Monsoon
    7: 'JAS', 8: 'JAS', 9: 'JAS',        # Monsoon
    10: 'OND', 11: 'OND', 12: 'OND',     # Post-Monsoon
}

SEASON_ORDER = ['JFM', 'AMJ', 'JAS', 'OND']

SEASON_COLORS = {
    'JFM': '#2196F3',   # blue  – Winter
    'AMJ': '#FF9800',   # orange – Pre-Monsoon
    'JAS': '#4CAF50',   # green – Monsoon
    'OND': '#E91E63',   # pink  – Post-Monsoon
}

SEASON_LABELS = {
    'JFM': 'JFM (Jan–Mar)',
    'AMJ': 'AMJ (Apr–Jun)',
    'JAS': 'JAS (Jul–Sep)',
    'OND': 'OND (Oct–Dec)',
}


def _assign_season(df):
    """Add a 'Season' column based on the month of Peak_Date."""
    df = df.copy()
    df['Peak_Date'] = pd.to_datetime(df['Peak_Date'])
    df['Season'] = df['Peak_Date'].dt.month.map(SEASON_MAP)
    return df


# =============================================================================
# SEASONAL BAR PLOT – one bar per cyclone, colour = season
# =============================================================================

def _seasonal_bar_plot(df, col, title, y_label, save_name, save=True):
    """Bar plot with x = peak date, bars coloured by season, cyclone names."""
    valid = df.dropna(subset=[col, 'Peak_Date']).copy()
    if valid.empty:
        print(f"No data for {col}"); return

    valid['Peak_Date'] = pd.to_datetime(valid['Peak_Date'])
    valid = _assign_season(valid)
    valid = valid.sort_values('Peak_Date')

    # Optimized dimensions for visibility and layout stability
    fig_width = min(20, max(14, len(valid) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # draw seasonal background bands
    # draw seasonal background bands for 2024
    year = 2024
    season_spans = [
        ('JFM', f'{year}-01-01', f'{year}-03-31'),
        ('AMJ', f'{year}-04-01', f'{year}-06-30'),
        ('JAS', f'{year}-07-01', f'{year}-09-30'),
        ('OND', f'{year}-10-01', f'{year}-12-31'),
    ]
    for sname, s0, s1 in season_spans:
        ax.axvspan(pd.to_datetime(s0), pd.to_datetime(s1),
                   alpha=0.08, color=SEASON_COLORS[sname], zorder=0)
        # season label at top (using transform to keep it at the top of the plot)
        mid = pd.to_datetime(s0) + (pd.to_datetime(s1) - pd.to_datetime(s0)) / 2
        ax.text(mid, 0.98, SEASON_LABELS[sname],
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=9, fontweight='bold',
                color=SEASON_COLORS[sname], alpha=0.8)

    # bars coloured by season, remove edge to prevent thin bars looking white
    bar_colors = [SEASON_COLORS[s] for s in valid['Season']]
    bars = ax.bar(valid['Peak_Date'], valid[col],
                  color=bar_colors, edgecolor=None, linewidth=0, width=0.8, zorder=3)

    ax.axhline(0, color='gray', linewidth=0.8, zorder=2)

    # annotate cyclone names
    ymin, ymax = valid[col].min(), valid[col].max()
    offset = 0.03 * (ymax - ymin) if ymax != ymin else 0.01
    for bar, name in zip(bars, valid['Cyclone']):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (offset if h >= 0 else -offset),
                name,
                ha='right', va='bottom' if h >= 0 else 'top',
                fontsize=6, rotation=45, rotation_mode='anchor', color='#333333')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Peak Date', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=45)
    ax.grid(axis='y', alpha=0.25, zorder=0)

    # season legend
    from matplotlib.patches import Patch
    handles = [Patch(color=SEASON_COLORS[s], label=SEASON_LABELS[s])
               for s in SEASON_ORDER]
    ax.legend(handles=handles, fontsize=8, framealpha=0.9, loc='upper right')

    # Manual adjustment to prevent "Tight layout" warnings and excessive whitespace
    plt.subplots_adjust(bottom=0.25, top=0.88, left=0.08, right=0.98)
    if save:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


def plot_seasonal_summary(df, save=True):
    """Two grouped bar charts showing the mean anomaly per season
    for each spatial scale (Large / Meso / Sub-meso).
    """
    dfs = _assign_season(df)

    for var, cols, y_label, fname in [
        ('SST',  ['SST_Large', 'SST_Meso', 'SST_Submeso'],
         'Mean SST Anomaly (°C)', 'SST_Seasonal_Summary.png'),
        ('MSLA', ['MSLA_Large', 'MSLA_Meso', 'MSLA_Submeso'],
         'Mean MSLA Anomaly (m)', 'MSLA_Seasonal_Summary.png'),
    ]:
        means = dfs.groupby('Season')[cols].mean().reindex(SEASON_ORDER)
        counts = dfs.groupby('Season')['Cyclone'].count().reindex(SEASON_ORDER).fillna(0).astype(int)

        x = np.arange(len(SEASON_ORDER))
        width = 0.25
        scale_labels = ['Large', 'Meso', 'Sub-meso']
        scale_colors = ['#5C6BC0', '#26A69A', '#FFA726']

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (c, label, clr) in enumerate(zip(cols, scale_labels, scale_colors)):
            vals = means[c].fillna(0).values
            bars = ax.bar(x + i * width, vals, width,
                          label=label, color=clr, edgecolor='white', zorder=3)
            # value labels on bars
            ax.margins(y=0.2)
            for bar, v in zip(bars, vals):
                if not np.isnan(v) and v != 0:
                    h = bar.get_height()
                    offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + (offset if h >= 0 else -offset),
                            f'{v:.4f}', ha='center', 
                            va='bottom' if h >= 0 else 'top',
                            fontsize=7, color='#333333', fontweight='bold')

        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{SEASON_LABELS[s]}\n(n={counts[s]})"
                            for s in SEASON_ORDER], fontsize=9)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{var} Anomaly — Seasonal Mean by Scale',
                     fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis='y', alpha=0.25, zorder=0)

        plt.tight_layout()
        if save:
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"Saved: {fname}")
        plt.show()

def plot_anomaly_correlation(df, save=True):
    """Scatter of SST vs MSLA (Meso-scale) across all cyclones."""
    col_sst  = 'SST_Meso'
    col_msla = 'MSLA_Meso'
    
    valid = df.dropna(subset=[col_sst, col_msla]).copy()
    if valid.empty:
        print("No paired data for correlation plotting."); return

    sst  = valid[col_sst].values
    msla = valid[col_msla].values
    corr, pval = pearsonr(sst, msla)

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', 'H',
               'X', 'd', 'P', '8', '*']
    cmap = plt.get_cmap('tab20', len(valid))

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color='gray', linewidth=0.8, alpha=0.5, zorder=1)

    for i, (_, row) in enumerate(valid.iterrows()):
        x, y = row[col_sst], row[col_msla]
        ax.scatter(x, y,
                   marker=markers[i % len(markers)],
                   color=cmap(i), s=100, zorder=4,
                   edgecolors='white', linewidths=0.5,
                   label=row['Cyclone'])
        
        x_off = 5 if x < np.mean(sst) else -5
        y_off = 5 if y < np.mean(msla) else -5
        ha = 'left' if x_off > 0 else 'right'
        va = 'bottom' if y_off > 0 else 'top'

        ax.annotate(row['Cyclone'],
                    xy=(x, y),
                    xytext=(x_off, y_off), textcoords='offset points',
                    ha=ha, va=va,
                    fontsize=6.5, color='#333333', fontweight='medium')

    m, b   = np.polyfit(sst, msla, 1)
    xs     = np.linspace(sst.min(), sst.max(), 200)
    ax.plot(xs, m * xs + b, color='crimson', linestyle='--',
            linewidth=1.8, label=f'Linear fit (r={corr:.2f})', zorder=3)

    ax.set_xlabel('Mean SST Anomaly (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean MSLA Anomaly (m)',  fontsize=12, fontweight='bold')
    ax.set_title('SST ↔ MSLA Correlation — All Cyclones (Meso-scale)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=7, framealpha=0.8, ncol=1, title="Cyclones", title_fontsize=8)

    stats_text = (f'Pearson r = {corr:.3f}\n'
                  f'p-value   = {pval:.4f}\n'
                  f'n = {len(valid)}')
    ax.text(0.03, 0.97, stats_text,
            transform=ax.transAxes, va='top', ha='left', fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_box_aspect(0.8)
    plt.tight_layout()
    if save:
        plt.savefig('SST_MSLA_Correlation.png', dpi=150, bbox_inches='tight')
        print("Saved: SST_MSLA_Correlation.png")
    plt.show()

# =============================================================================
# QUICK RUNNER
# =============================================================================
if __name__ == "__main__":
    if 'CYCLONES' not in locals() and 'CYCLONES' not in globals():
        json_path = 'cyclone_info_2024_padded.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                CYCLONES = json.load(f)
        else:
            CYCLONES = {}

    if CYCLONES:
        # Uncomment to re-calculate anomalies (requires access to netCDF data paths)
        df_multi = calculate_multiscale_anomalies(CYCLONES)
        try:
            if 'df_multi' not in locals():
                df_multi = pd.read_csv('cyclone_multiscale_anomalies.csv')
            plot_sst_scales(df_multi)
            plot_msla_scales(df_multi)
            plot_seasonal_summary(df_multi)
            plot_anomaly_correlation(df_multi)
        except FileNotFoundError:
            print("CSV not found.")
