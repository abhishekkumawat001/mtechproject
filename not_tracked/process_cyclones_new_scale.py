import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import gc

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150})

# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
SST_DATA_DIR = r"C:\Users\abhik\Downloads\New_folder\MUR-JPL-L4-GLOB-v4.1_4.1-20260316_132446"
MSLA_ROOT_DIR = r"C:\Users\abhik\Downloads\SEALEVEL_GLO_PHY_L4_MY_008_047_cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_2024"
OUTPUT_ROOT_DIR = r"C:\Users\abhik\Desktop\project related work\Cyclone_Analysis_Outputs_newscale_500large"
CYCLONE_JSON_PATH = r"C:\Users\abhik\Desktop\project related work\cyclone_info_2024_padded.json"
CSV_FILE_PATH = r"C:\Users\abhik\Desktop\project related work\cyclone_multiscale_anomalies.csv"

# New filter scales
FILTER_SCALES = {
    'submeso': 0.25,  # ~28 km
    'meso':    1.0,   # ~111 km
    'large':   3.0,   # ~333 km
}

FILTER_SCALES_MSLA = {
    'submeso': 0.625,  # ~69 km
    'meso':    1.5,    # ~167 km
    'large':   3.0,    # ~333 km
}

ANALYSIS_COLOR = '#444444'
ANALYSIS_TEXT = 'white'

# =============================================================================
# LOADERS & GEOMETRY
# =============================================================================
def _make_square_bbox(bbox, track=None, pad_deg=0.6):
    lon_min, lon_max = bbox['lon_min'], bbox['lon_max']
    lat_min, lat_max = bbox['lat_min'], bbox['lat_max']

    if track:
        track_lons = [t[0] for t in track]
        track_lats = [t[1] for t in track]
        lon_min = min(lon_min, min(track_lons))
        lon_max = max(lon_max, max(track_lons))
        lat_min = min(lat_min, min(track_lats))
        lat_max = max(lat_max, max(track_lats))

    lon_min -= pad_deg
    lon_max += pad_deg
    lat_min -= pad_deg
    lat_max += pad_deg

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    span = max(lon_span, lat_span, 0.01)

    lon_mid = (lon_min + lon_max) / 2.0
    lat_mid = (lat_min + lat_max) / 2.0
    half = span / 2.0

    return {
        'lon_min': lon_mid - half,
        'lon_max': lon_mid + half,
        'lat_min': lat_mid - half,
        'lat_max': lat_mid + half,
    }

def load_sst_for_date(date, region):
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    date_str = date.strftime('%Y%m%d')
    pattern = os.path.join(SST_DATA_DIR, f"{date_str}*.nc")
    files = glob.glob(pattern)
    if not files:
        return None
    try:
        ds = xr.open_dataset(files[0])
        sst = ds['analysed_sst'].squeeze('time').sel(
            lon=slice(region['lon_min'], region['lon_max']),
            lat=slice(region['lat_min'], region['lat_max'])
        ).load() - 273.15
        ds.close()
        return sst
    except Exception as e:
        print(f"Error loading SST for {date_str}: {e}")
        return None

def load_msla_for_date(date, bbox):
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    date_str = date.strftime('%Y%m%d')
    pattern = os.path.join(MSLA_ROOT_DIR, f'*_{date_str}_*.nc')
    files = glob.glob(pattern)
    if not files:
        return None
    try:
        ds = xr.open_dataset(files[0])
        sla = ds['sla'].squeeze('time', drop=True).sel(
            latitude=slice(bbox['lat_min'], bbox['lat_max']), 
            longitude=slice(bbox['lon_min'], bbox['lon_max'])
        ).load()
        ds.close()
        return sla
    except Exception as e:
        print(f"Error loading MSLA for {date_str}: {e}")
        return None

def _get_analysis_date_list_from_range(start_str, end_str):
    return [d.to_pydatetime() for d in pd.date_range(start_str, end_str, freq='D')]

# =============================================================================
# SCALE DECOMPOSITION
# =============================================================================
def scale_decomposition(field_2d, grid_res, filter_scales, ref_lat=19.0):
    data = field_2d.values.astype(float).copy() if hasattr(field_2d, 'values') else np.asarray(field_2d).copy()
    mask = np.isnan(data)
    if np.all(mask):
        return None
    data_filled = np.where(mask, np.nanmean(data), data)
    dlat_km = 111.0 * grid_res
    dlon_km = 111.0 * np.cos(np.deg2rad(ref_lat)) * grid_res
    
    def _sig(scale_deg):
        skm = 111.0 * scale_deg
        return (skm / dlat_km, skm / dlon_km)
        
    s_meso = _sig(filter_scales['meso'])
    s_large = _sig(filter_scales['large'])
    
    meso_lp = gaussian_filter(data_filled, sigma=s_meso)
    large_lp = gaussian_filter(data_filled, sigma=s_large)
    
    return {
        'large_scale': np.where(mask, np.nan, large_lp),
        'meso': np.where(mask, np.nan, meso_lp - large_lp),
        'submeso': np.where(mask, np.nan, data_filled - meso_lp)
    }

def decompose_sst_field(sst_data):
    ref_lat = float(sst_data.lat.mean())
    return scale_decomposition(sst_data, 0.05, FILTER_SCALES, ref_lat)

def decompose_msla_field(msla_data):
    ref_lat = float(msla_data.latitude.mean())
    return scale_decomposition(msla_data, 0.125, FILTER_SCALES_MSLA, ref_lat)

# =============================================================================
# PLOT HELPERS (1-ROW PANEL)
# =============================================================================
def _add_phase_date_label(ax, phase_upper, date_txt, phase_color):
    ax.text(
        0.5, 1.16, phase_upper,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=8.2, fontweight='bold', color='white',
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor=phase_color,
            edgecolor='none',
            alpha=0.95
        ),
        path_effects=[pe.withStroke(linewidth=1.4, foreground='black')],
        clip_on=False
    )

    ax.text(
        0.5, 1.05, date_txt,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=7.6, fontweight='bold', color='black',
        clip_on=False
    )

def _draw_evolution_panel(ax, field, lons, lats, bbox,
                          cmap, vmin, vmax, diverge,
                          track_data=None, date=None,
                          add_left_labels=False, add_bottom_labels=False):
    ax.set_extent([bbox['lon_min'], bbox['lon_max'],
                   bbox['lat_min'], bbox['lat_max']],
                  crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='#d4c9a8', zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', zorder=4)

    gl = ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.35, linestyle='--')
    if add_left_labels:
        gl.left_labels = True
        gl.ylabel_style = {'size': 6}
    if add_bottom_labels:
        gl.bottom_labels = True
        gl.xlabel_style = {'size': 6}

    if field is not None and lons is not None and lats is not None:
        # Check if vmin/vmax are valid for TwoSlopeNorm
        if diverge and vmin < 0 and vmax > 0:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(lons, lats, field, cmap=cmap, norm=norm,
                           transform=ccrs.PlateCarree(), shading='auto', zorder=1)
    else:
        im = None
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='gray')

    # track overlays
    if track_data and date is not None:
        all_lons = [e[0] for e in track_data]
        all_lats = [e[1] for e in track_data]
        all_dts  = [datetime.strptime(e[2], '%Y-%m-%d %H:%M') for e in track_data]

        ax.plot(all_lons, all_lats, '--', color='white', lw=0.8, alpha=0.4,
                transform=ccrs.PlateCarree(), zorder=5)

        past_lons = [lo for lo, dt in zip(all_lons, all_dts) if dt.date() <= date.date()]
        past_lats = [la for la, dt in zip(all_lats, all_dts) if dt.date() <= date.date()]
        if past_lons:
            ax.plot(past_lons, past_lats, '-o', color='yellow', lw=1.2, ms=2.5,
                    transform=ccrs.PlateCarree(), zorder=6)
            ax.plot(past_lons[-1], past_lats[-1], '*', color='red', ms=7.0,
                    transform=ccrs.PlateCarree(), zorder=7)

    return im

def plot_evolution_one_row(storm_name, window_label, dates, lons, lats, field_history, bbox, track, title, unit, cmap, vmin, vmax, diverge, save_path, cyc):
    n_cols = len(dates)
    panel_w = 3.0
    fig_w   = max(26, n_cols * panel_w + 2.0)
    fig_h   = 4.6 + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        2, n_cols,
        height_ratios=[0.055, 1],
        hspace=0.22, wspace=0.03,
        left=0.05, right=0.90,
        top=0.91, bottom=0.05
    )

    # header row
    hax = fig.add_subplot(gs[0, :])
    hax.set_facecolor(ANALYSIS_COLOR)
    hax.set_axis_off()
    hax.text(0.5, 0.5, f"{storm_name} - {window_label}", transform=hax.transAxes,
             ha='center', va='center',
             fontsize=12, fontweight='bold',
             color=ANALYSIS_TEXT,
             path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    cax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
    im_to_cb = None

    dates_info = cyc.get('dates', {})
    formation = dates_info.get('formation')
    dissipation = dates_info.get('dissipation')
    formation_dt = datetime.strptime(formation, '%Y-%m-%d') if formation else None
    dissipation_dt = datetime.strptime(dissipation, '%Y-%m-%d') if dissipation else None

    phase_colors = {
        'pre': '#4a90d9',
        'during': '#e05c2a',
        'post': '#3daa5e'
    }

    for ci, date in enumerate(dates):
        ax = fig.add_subplot(gs[1, ci], projection=ccrs.PlateCarree())

        for spine in ax.spines.values():
            spine.set_edgecolor(ANALYSIS_COLOR)
            spine.set_linewidth(1.2)

        field = field_history[ci]

        im = _draw_evolution_panel(
            ax, field, lons, lats, bbox,
            cmap=cmap, vmin=vmin, vmax=vmax, diverge=diverge,
            track_data=track, date=date,
            add_left_labels=(ci == 0),
            add_bottom_labels=True
        )
        if im is not None:
            im_to_cb = im

        # Add phase/date label
        phase = 'during'
        if formation_dt and dissipation_dt:
            if date < formation_dt:
                phase = 'pre'
            elif date <= dissipation_dt:
                phase = 'during'
            else:
                phase = 'post'
        phase_upper = phase.upper()
        phase_color = phase_colors.get(phase, '#888888')
        date_txt = date.strftime('%b %d')
        _add_phase_date_label(ax, phase_upper, date_txt, phase_color)

    if im_to_cb is not None:
        cb = fig.colorbar(im_to_cb, cax=cax, orientation='vertical', extend='both')
        cb.set_label(unit, fontsize=8.0)
        cb.ax.tick_params(labelsize=7.0)

    fig.legend(
        handles=[
            plt.Line2D([0], [0], color='white',  lw=0.8, linestyle='--', label='Full track'),
            plt.Line2D([0], [0], color='yellow', lw=1.2, marker='o', ms=3, label='Track to date'),
            plt.Line2D([0], [0], color='red',    lw=0, marker='*', ms=8, label='Current position'),
        ],
        loc='lower right', bbox_to_anchor=(0.915, 0.055),
        fontsize=7.0, framealpha=0.85, edgecolor='#cccccc', ncol=1
    )

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    gc.collect()

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
def main():
    if not os.path.exists(CYCLONE_JSON_PATH):
        print(f"Error: JSON path {CYCLONE_JSON_PATH} does not exist.")
        return

    with open(CYCLONE_JSON_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Flatten storm database
    CYCLONES = {}
    for basin, b_data in raw_data.items():
        for storm in b_data.get('storms', []):
            name = storm.get('name', '').upper()
            CYCLONES[name] = storm

    records = []

    for storm_name, cyc in CYCLONES.items():
        dates_info = cyc.get('dates', {})
        start_analysis_str = dates_info.get('start_analysis_date')
        end_analysis_str = dates_info.get('end_analysis_date')
        peak_date_str = dates_info.get('peak')

        if not start_analysis_str or not end_analysis_str or not peak_date_str:
            print(f"Skipping {storm_name}: incomplete dates in JSON.")
            continue

        print(f"\n========================================\nProcessing {storm_name}...")
        
        # Bounding box and track
        track = cyc.get('track', [])
        bbox = _make_square_bbox(cyc['bbox'], track=track)
        ref_lat = (bbox['lat_min'] + bbox['lat_max']) / 2.0

        # Create output dir
        storm_out_dir = os.path.join(OUTPUT_ROOT_DIR, storm_name)
        os.makedirs(storm_out_dir, exist_ok=True)

        # Date lists
        analysis_dates = _get_analysis_date_list_from_range(start_analysis_str, end_analysis_str)
        
        # 1. Decompose daily fields for the padded ANALYSIS window
        sst_history = {'raw': [], 'large_scale': [], 'meso': [], 'submeso': [], 'submeso_grad': []}
        msla_history = {'raw': [], 'large_scale': [], 'meso': [], 'submeso': [], 'submeso_grad': []}
        
        sst_lons, sst_lats = None, None
        msla_lons, msla_lats = None, None

        for d in analysis_dates:
            # SST
            sst_raw = load_sst_for_date(d, bbox)
            if sst_raw is not None:
                if sst_lons is None:
                    sst_lons, sst_lats = sst_raw.lon.values, sst_raw.lat.values
                dec_sst = decompose_sst_field(sst_raw)
                if dec_sst:
                    # Daily submeso gradient
                    dy, dx = np.gradient(dec_sst['submeso'])
                    dy_km = dy / (111.0 * 0.05)
                    dx_km = dx / (111.0 * np.cos(np.deg2rad(ref_lat)) * 0.05)
                    sst_sub_grad = np.sqrt(dy_km**2 + dx_km**2)
                    
                    sst_history['raw'].append(sst_raw.values)
                    sst_history['large_scale'].append(dec_sst['large_scale'])
                    sst_history['meso'].append(dec_sst['meso'])
                    sst_history['submeso'].append(dec_sst['submeso'])
                    sst_history['submeso_grad'].append(sst_sub_grad)
                else:
                    for k in sst_history: sst_history[k].append(None)
            else:
                for k in sst_history: sst_history[k].append(None)

            # MSLA
            msla_raw = load_msla_for_date(d, bbox)
            if msla_raw is not None:
                if msla_lons is None:
                    msla_lons, msla_lats = msla_raw.longitude.values, msla_raw.latitude.values
                dec_msla = decompose_msla_field(msla_raw)
                if dec_msla:
                    # Daily submeso gradient
                    dy, dx = np.gradient(dec_msla['submeso'])
                    dy_km = dy / (111.0 * 0.125)
                    dx_km = dx / (111.0 * np.cos(np.deg2rad(ref_lat)) * 0.125)
                    msla_sub_grad = np.sqrt(dy_km**2 + dx_km**2)

                    msla_history['raw'].append(msla_raw.values)
                    msla_history['large_scale'].append(dec_msla['large_scale'])
                    msla_history['meso'].append(dec_msla['meso'])
                    msla_history['submeso'].append(dec_msla['submeso'])
                    msla_history['submeso_grad'].append(msla_sub_grad)
                else:
                    for k in msla_history: msla_history[k].append(None)
            else:
                for k in msla_history: msla_history[k].append(None)

        # 2. Save decomposed fields to NetCDF (time, lat, lon)
        time_arr = pd.to_datetime(analysis_dates)
        
        # Helper to convert list of 2D arrays to a clean 3D numpy array or None if empty
        def _to_3d(history_list, lats, lons):
            # Check if we have at least one valid array to get the shape
            first_valid = next((arr for arr in history_list if arr is not None), None)
            if first_valid is None:
                return None
            ny, nx = first_valid.shape
            clean_list = []
            for arr in history_list:
                if arr is not None:
                    clean_list.append(arr)
                else:
                    clean_list.append(np.full((ny, nx), np.nan))
            return np.stack(clean_list, axis=0)

        # Save SST NetCDF
        sst_3d = {k: _to_3d(sst_history[k], sst_lats, sst_lons) for k in sst_history}
        if sst_3d['raw'] is not None:
            sst_ds = xr.Dataset(
                data_vars={
                    'raw': (['time', 'lat', 'lon'], sst_3d['raw']),
                    'large_scale': (['time', 'lat', 'lon'], sst_3d['large_scale']),
                    'meso': (['time', 'lat', 'lon'], sst_3d['meso']),
                    'submeso': (['time', 'lat', 'lon'], sst_3d['submeso']),
                    'submeso_grad': (['time', 'lat', 'lon'], sst_3d['submeso_grad'])
                },
                coords={
                    'time': time_arr,
                    'lat': sst_lats,
                    'lon': sst_lons
                }
            )
            sst_nc_path = os.path.join(storm_out_dir, f"{storm_name}_SST_decomposed.nc")
            sst_ds.to_netcdf(sst_nc_path)
            print(f"Saved NetCDF: {sst_nc_path}")
            sst_ds.close()
        
        # Save MSLA NetCDF
        msla_3d = {k: _to_3d(msla_history[k], msla_lats, msla_lons) for k in msla_history}
        if msla_3d['raw'] is not None:
            msla_ds = xr.Dataset(
                data_vars={
                    'raw': (['time', 'lat', 'lon'], msla_3d['raw']),
                    'large_scale': (['time', 'lat', 'lon'], msla_3d['large_scale']),
                    'meso': (['time', 'lat', 'lon'], msla_3d['meso']),
                    'submeso': (['time', 'lat', 'lon'], msla_3d['submeso']),
                    'submeso_grad': (['time', 'lat', 'lon'], msla_3d['submeso_grad'])
                },
                coords={
                    'time': time_arr,
                    'lat': msla_lats,
                    'lon': msla_lons
                }
            )
            msla_nc_path = os.path.join(storm_out_dir, f"{storm_name}_MSLA_decomposed.nc")
            msla_ds.to_netcdf(msla_nc_path)
            print(f"Saved NetCDF: {msla_nc_path}")
            msla_ds.close()

        # 3. Generate 1-row evolution plots
        # SST plots
        if sst_3d['raw'] is not None:
            # SST Submeso
            plot_evolution_one_row(
                storm_name=storm_name, window_label="ANALYSIS", dates=analysis_dates,
                lons=sst_lons, lats=sst_lats, field_history=sst_3d['submeso'], bbox=bbox, track=track,
                title=f"{storm_name} -- SST Sub-mesoscale Anomaly (<55 km)", unit="SST Anom (°C)",
                cmap=plt.get_cmap('RdBu_r'), vmin=-0.3, vmax=0.3, diverge=True,
                save_path=os.path.join(storm_out_dir, f"{storm_name}_SST_submeso_evolution.png"), cyc=cyc
            )
            # SST Meso
            plot_evolution_one_row(
                storm_name=storm_name, window_label="ANALYSIS", dates=analysis_dates,
                lons=sst_lons, lats=sst_lats, field_history=sst_3d['meso'], bbox=bbox, track=track,
                title=f"{storm_name} -- SST Mesoscale Anomaly (55-500 km)", unit="SST Anom (°C)",
                cmap=plt.get_cmap('RdBu_r'), vmin=-0.5, vmax=0.5, diverge=True,
                save_path=os.path.join(storm_out_dir, f"{storm_name}_SST_meso_evolution.png"), cyc=cyc
            )
            # SST Submeso Grad
            plot_evolution_one_row(
                storm_name=storm_name, window_label="ANALYSIS", dates=analysis_dates,
                lons=sst_lons, lats=sst_lats, field_history=sst_3d['submeso_grad'], bbox=bbox, track=track,
                title=f"{storm_name} -- SST Sub-mesoscale Spatial Gradient Magnitude", unit="Gradient (°C/km)",
                cmap=plt.get_cmap('viridis'), vmin=0, vmax=0.02, diverge=False,
                save_path=os.path.join(storm_out_dir, f"{storm_name}_SST_submeso_grad_evolution.png"), cyc=cyc
            )

        # MSLA plots
        if msla_3d['raw'] is not None:
            # MSLA Submeso
            plot_evolution_one_row(
                storm_name=storm_name, window_label="ANALYSIS", dates=analysis_dates,
                lons=msla_lons, lats=msla_lats, field_history=msla_3d['submeso'], bbox=bbox, track=track,
                title=f"{storm_name} -- MSLA Sub-mesoscale Anomaly (<69 km)", unit="SLA Anom (m)",
                cmap=plt.get_cmap('RdBu_r'), vmin=-0.04, vmax=0.04, diverge=True,
                save_path=os.path.join(storm_out_dir, f"{storm_name}_MSLA_submeso_evolution.png"), cyc=cyc
            )
            # MSLA Meso
            plot_evolution_one_row(
                storm_name=storm_name, window_label="ANALYSIS", dates=analysis_dates,
                lons=msla_lons, lats=msla_lats, field_history=msla_3d['meso'], bbox=bbox, track=track,
                title=f"{storm_name} -- MSLA Mesoscale Anomaly (69-500 km)", unit="SLA Anom (m)",
                cmap=plt.get_cmap('RdBu_r'), vmin=-0.10, vmax=0.10, diverge=True,
                save_path=os.path.join(storm_out_dir, f"{storm_name}_MSLA_meso_evolution.png"), cyc=cyc
            )
            # MSLA Submeso Grad
            plot_evolution_one_row(
                storm_name=storm_name, window_label="ANALYSIS", dates=analysis_dates,
                lons=msla_lons, lats=msla_lats, field_history=msla_3d['submeso_grad'], bbox=bbox, track=track,
                title=f"{storm_name} -- MSLA Sub-mesoscale Spatial Gradient Magnitude", unit="Gradient (m/km)",
                cmap=plt.get_cmap('viridis'), vmin=0, vmax=0.003, diverge=False,
                save_path=os.path.join(storm_out_dir, f"{storm_name}_MSLA_submeso_grad_evolution.png"), cyc=cyc
            )

        # 4. Compute values for global CSV
        # Parse analysis_dates dictionary to find the baseline (pre) and during windows
        # Note: If no 'pre' is defined in JSON, we fall back to formation to dissipation.
        analysis_dict = cyc.get('analysis_dates', {})
        pre_win = analysis_dict.get('pre')
        during_win = analysis_dict.get('during')
        
        # Default fallback
        if not during_win:
            formation = dates_info.get('formation')
            dissipation = dates_info.get('dissipation')
            during_win = [formation, dissipation]
            
        # Get dates list
        pre_dates = _get_analysis_date_list_from_range(pre_win[0], pre_win[1]) if pre_win else []
        during_dates = _get_analysis_date_list_from_range(during_win[0], during_win[1])

        # Baseline Calculation
        pre_sst_large_mean, pre_sst_large_std = [], []
        pre_sst_meso_std, pre_sst_sub_std = [], []
        pre_msla_large_mean, pre_msla_large_std = [], []
        pre_msla_meso_std, pre_msla_sub_std = [], []

        for d in pre_dates:
            # SST pre
            s = load_sst_for_date(d, bbox)
            if s is not None:
                dec = decompose_sst_field(s)
                if dec:
                    pre_sst_large_mean.append(np.nanmean(dec['large_scale']))
                    pre_sst_large_std.append(np.nanstd(dec['large_scale']))
                    pre_sst_meso_std.append(np.nanstd(dec['meso']))
                    pre_sst_sub_std.append(np.nanstd(dec['submeso']))
            # MSLA pre
            m = load_msla_for_date(d, bbox)
            if m is not None:
                dec = decompose_msla_field(m)
                if dec:
                    pre_msla_large_mean.append(np.nanmean(dec['large_scale']))
                    pre_msla_large_std.append(np.nanstd(dec['large_scale']))
                    pre_msla_meso_std.append(np.nanstd(dec['meso']))
                    pre_msla_sub_std.append(np.nanstd(dec['submeso']))

        sst_base = np.nanmean(pre_sst_large_mean) if pre_sst_large_mean else 0
        msla_base = np.nanmean(pre_msla_large_mean) if pre_msla_large_mean else 0

        sst_large_std_base = np.nanmean(pre_sst_large_std) if pre_sst_large_std else 0
        sst_meso_std_base = np.nanmean(pre_sst_meso_std) if pre_sst_meso_std else 0
        sst_sub_std_base = np.nanmean(pre_sst_sub_std) if pre_sst_sub_std else 0

        msla_large_std_base = np.nanmean(pre_msla_large_std) if pre_msla_large_std else 0
        msla_meso_std_base = np.nanmean(pre_msla_meso_std) if pre_msla_meso_std else 0
        msla_sub_std_base = np.nanmean(pre_msla_sub_std) if pre_msla_sub_std else 0

        # During Window Calculation
        during_sst_large_mean, during_sst_large_std = [], []
        during_sst_meso_mean, during_sst_meso_std = [], []
        during_sst_sub_mean, during_sst_sub_std = [], []
        during_sst_sub_grad_mean = []

        during_msla_large_mean, during_msla_large_std = [], []
        during_msla_meso_mean, during_msla_meso_std = [], []
        during_msla_sub_mean, during_msla_sub_std = [], []
        during_msla_sub_grad_mean = []

        for d in during_dates:
            # SST during
            s = load_sst_for_date(d, bbox)
            if s is not None:
                dec = decompose_sst_field(s)
                if dec:
                    dy, dx = np.gradient(dec['submeso'])
                    dy_km = dy / (111.0 * 0.05)
                    dx_km = dx / (111.0 * np.cos(np.deg2rad(ref_lat)) * 0.05)
                    sst_sub_grad = np.sqrt(dy_km**2 + dx_km**2)

                    during_sst_large_mean.append(np.nanmean(dec['large_scale']))
                    during_sst_large_std.append(np.nanstd(dec['large_scale']))
                    during_sst_meso_mean.append(np.nanmean(dec['meso']))
                    during_sst_meso_std.append(np.nanstd(dec['meso']))
                    during_sst_sub_mean.append(np.nanmean(dec['submeso']))
                    during_sst_sub_std.append(np.nanstd(dec['submeso']))
                    during_sst_sub_grad_mean.append(np.nanmean(sst_sub_grad))

            # MSLA during
            m = load_msla_for_date(d, bbox)
            if m is not None:
                dec = decompose_msla_field(m)
                if dec:
                    dy, dx = np.gradient(dec['submeso'])
                    dy_km = dy / (111.0 * 0.125)
                    dx_km = dx / (111.0 * np.cos(np.deg2rad(ref_lat)) * 0.125)
                    msla_sub_grad = np.sqrt(dy_km**2 + dx_km**2)

                    during_msla_large_mean.append(np.nanmean(dec['large_scale']))
                    during_msla_large_std.append(np.nanstd(dec['large_scale']))
                    during_msla_meso_mean.append(np.nanmean(dec['meso']))
                    during_msla_meso_std.append(np.nanstd(dec['meso']))
                    during_msla_sub_mean.append(np.nanmean(dec['submeso']))
                    during_msla_sub_std.append(np.nanstd(dec['submeso']))
                    during_msla_sub_grad_mean.append(np.nanmean(msla_sub_grad))

        # Compile CSV record
        rec = {
            'Cyclone': storm_name,
            'Peak_Date': str(peak_date_str),
            'SST_Large': np.nanmean(during_sst_large_mean) - sst_base if during_sst_large_mean else np.nan,
            'SST_Meso': np.nanmean(during_sst_meso_mean) if during_sst_meso_mean else np.nan,
            'SST_Submeso': np.nanmean(during_sst_sub_mean) if during_sst_sub_mean else np.nan,
            'MSLA_Large': np.nanmean(during_msla_large_mean) - msla_base if during_msla_large_mean else np.nan,
            'MSLA_Meso': np.nanmean(during_msla_meso_mean) if during_msla_meso_mean else np.nan,
            'MSLA_Submeso': np.nanmean(during_msla_sub_mean) if during_msla_sub_mean else np.nan,
            
            'SST_Large_Std': np.nanmean(during_sst_large_std) - sst_large_std_base if during_sst_large_std else np.nan,
            'SST_Meso_Std': np.nanmean(during_sst_meso_std) - sst_meso_std_base if during_sst_meso_std else np.nan,
            'SST_Submeso_Std': np.nanmean(during_sst_sub_std) - sst_sub_std_base if during_sst_sub_std else np.nan,
            
            'MSLA_Large_Std': np.nanmean(during_msla_large_std) - msla_large_std_base if during_msla_large_std else np.nan,
            'MSLA_Meso_Std': np.nanmean(during_msla_meso_std) - msla_meso_std_base if during_msla_meso_std else np.nan,
            'MSLA_Submeso_Std': np.nanmean(during_msla_sub_std) - msla_sub_std_base if during_msla_sub_std else np.nan,
            
            'SST_Submeso_Grad_Avg': np.nanmean(during_sst_sub_grad_mean) if during_sst_sub_grad_mean else np.nan,
            'MSLA_Submeso_Grad_Avg': np.nanmean(during_msla_sub_grad_mean) if during_msla_sub_grad_mean else np.nan
        }

        print(f"Results computed for {storm_name}:")
        print(f"  SST_Submeso Anomaly: {rec['SST_Submeso']:.5f} | SST_Submeso_Grad: {rec['SST_Submeso_Grad_Avg']:.5f}")
        print(f"  MSLA_Submeso Anomaly: {rec['MSLA_Submeso']:.5f} | MSLA_Submeso_Grad: {rec['MSLA_Submeso_Grad_Avg']:.5f}")
        records.append(rec)
        
        # Cleanup memory
        del sst_history, msla_history, sst_3d, msla_3d
        gc.collect()

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(CSV_FILE_PATH, index=False)
    print(f"\n========================================\nSaved CSV to: {CSV_FILE_PATH}")

    # 5. Global Bar plots
    plot_global_bar_plots(df)

# =============================================================================
# GLOBAL PLOTTING FUNCTIONS
# =============================================================================
def _bar_plot_abs_per_date(df, col, title, y_label, save_name, std_threshold=None):
    if col not in df.columns:
        print(f"No data for {col}")
        return
    std_col = f"{col}_Std"
    has_std = std_col in df.columns
    valid = df.dropna(subset=[col, 'Peak_Date']).copy()
    if valid.empty:
        print(f"No data for {col}")
        return
    valid['Peak_Date'] = pd.to_datetime(valid['Peak_Date'])
    valid = valid.sort_values('Peak_Date')
    colors = ['#d62728' if v >= 0 else '#1f77b4' for v in valid[col]]
    abs_vals = valid[col].abs()
    fig_width = min(20, max(12, len(valid) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(valid['Peak_Date'], abs_vals, color=colors, edgecolor=None, linewidth=0, width=0.7)
    ax.axhline(0, color='gray', linewidth=0.8, zorder=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Peak Date', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    if len(valid) <= 12:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_xlim([pd.to_datetime('2023-12-15'), pd.to_datetime('2025-01-15')])
    fig.autofmt_xdate(rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    ymin, ymax = abs_vals.min(), abs_vals.max()
    offset = 0.02 * (ymax - ymin) if ymax != ymin else 0.01

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Warm (Original > 0)'),
        Patch(facecolor='#1f77b4', label='Cold (Original < 0)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    actual_std_threshold = std_threshold if std_threshold is not None else (valid[std_col].mean() if has_std else 0)

    for i, (bar, name) in enumerate(zip(bars, valid['Cyclone'])):
        height = bar.get_height()
        std_text = ""
        if has_std:
            std_val = valid[std_col].iloc[i]
            if std_val > actual_std_threshold:
                std_text = f"\nΔσ={std_val:.3f}"
        annot_text = f"{name}{std_text}"
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset, annot_text,
                ha='right', va='bottom', fontsize=6, rotation=45, rotation_mode='anchor', color='#333333')

    col_mean = valid[col].mean()
    col_std = valid[col].std()
    abs_mean = abs_vals.mean()
    stats_text = (f"n = {len(valid)}\nOriginal Mean = {col_mean:.3f}\nOriginal Std = {col_std:.3f}\nAbs Mean = {abs_mean:.3f}")
    if has_std:
        stats_text += f"\nΔσ Threshold = {actual_std_threshold:.3f}"

    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    plt.subplots_adjust(bottom=0.25, top=0.9, left=0.08, right=0.98)
    plt.savefig(os.path.join(OUTPUT_ROOT_DIR, f"{col}_Abs_vs_Date.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

def _bar_plot_grad_per_date(df, col, title, y_label, save_name, color='#2ca02c'):
    if col not in df.columns:
        print(f"No data for {col}")
        return
    valid = df.dropna(subset=[col, 'Peak_Date']).copy()
    if valid.empty:
        print(f"No data for {col}")
        return
    valid['Peak_Date'] = pd.to_datetime(valid['Peak_Date'])
    valid = valid.sort_values('Peak_Date')
    fig_width = min(20, max(12, len(valid) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(valid['Peak_Date'], valid[col], color=color, edgecolor=None, linewidth=0, width=0.7)
    ax.axhline(0, color='gray', linewidth=0.8, zorder=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Peak Date', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    if len(valid) <= 12:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_xlim([pd.to_datetime('2023-12-15'), pd.to_datetime('2025-01-15')])
    fig.autofmt_xdate(rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    ymin, ymax = valid[col].min(), valid[col].max()
    offset = 0.02 * (ymax - ymin) if ymax != ymin else 0.01

    for bar, name in zip(bars, valid['Cyclone']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset, name,
                ha='right', va='bottom', fontsize=6, rotation=45, rotation_mode='anchor', color='#333333')

    col_mean = valid[col].mean()
    col_std = valid[col].std()
    stats_text = (f"n = {len(valid)}\nMean = {col_mean:.5f}\nStd Dev = {col_std:.5f}")

    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    plt.subplots_adjust(bottom=0.25, top=0.9, left=0.08, right=0.98)
    plt.savefig(os.path.join(OUTPUT_ROOT_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_global_bar_plots(df):
    print("Generating global bar plots...")
    # 1. SST Submeso Abs
    _bar_plot_abs_per_date(df, 'SST_Submeso', 'Sub-mesoscale SST Absolute Anomaly Magnitude', 'Absolute SST Anomaly (°C)', 'SST_Submeso_Abs_vs_Date.png', std_threshold=0.10)
    # 2. MSLA Submeso Abs
    _bar_plot_abs_per_date(df, 'MSLA_Submeso', 'Sub-mesoscale MSLA Absolute Anomaly Magnitude', 'Absolute MSLA Anomaly (m)', 'MSLA_Submeso_Abs_vs_Date.png', std_threshold=0.05)
    # 3. SST Submeso Gradient
    _bar_plot_grad_per_date(df, 'SST_Submeso_Grad_Avg', 'Sub-mesoscale SST Average Spatial Gradient Magnitude', 'Gradient Magnitude (°C/km)', 'SST_Submeso_Grad_Avg_vs_Date.png', color='#2ca02c')
    # 4. MSLA Submeso Gradient
    _bar_plot_grad_per_date(df, 'MSLA_Submeso_Grad_Avg', 'Sub-mesoscale MSLA Average Spatial Gradient Magnitude', 'Gradient Magnitude (m/km)', 'MSLA_Submeso_Grad_Avg_vs_Date.png', color='#d62728')

if __name__ == '__main__':
    main()
