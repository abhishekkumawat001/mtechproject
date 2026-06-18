#!/usr/bin/env python3
"""Generate the CMEMS Bay of Bengal analysis notebook with cartopy maps."""

import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": ".venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}


def md(cell_id, source):
    notebook["cells"].append({
        "cell_type": "markdown", "id": cell_id,
        "metadata": {}, "source": source
    })


def code(cell_id, source):
    notebook["cells"].append({
        "cell_type": "code", "execution_count": None, "id": cell_id,
        "metadata": {}, "outputs": [], "source": source
    })


# ── Title ────────────────────────────────────────────────────────────
md("title01", [
    "# CMEMS Sea Level Data — Bay of Bengal Analysis 2024\n",
    "\n",
    "Product: **SEALEVEL_GLO_PHY_L4_MY_008_047** (0.125° daily)\n",
    "\n",
    "Features:\n",
    "- Lazy-loaded, chunked datasets via `xr.open_mfdataset`\n",
    "- Generic bounding-box subsetting for any region\n",
    "- Daily maps with cartopy coastlines + geographic axes\n",
    "- Monthly-mean 3×4 panel comparison\n",
    "- Area-averaged time series (latitude-weighted)\n",
    "- Monthly anomaly maps (relative to annual mean)\n",
    "- Switchable variable: `sla`, `adt`, `ugosa`, `vgosa`, …"
])

# ── 1. Imports & config ──────────────────────────────────────────────
md("sec01", ["## 1. Import Libraries & Configuration"])

code("imp01", [
    "import os\n",
    "import glob\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import xarray as xr\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "from matplotlib.colors import Normalize\n",
    "import cartopy.crs as ccrs\n",
    "import cartopy.feature as cfeature\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# ── Configuration ───────────────────────────────────────────\n",
    "ROOT_DIR = os.path.join(\n",
    "    r\"C:\\Users\\abhik\\Downloads\",\n",
    "    \"SEALEVEL_GLO_PHY_L4_MY_008_047 \"\n",
    "    \"cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D 2024 01\",\n",
    ")\n",
    "YEAR = 2024\n",
    "\n",
    "# Bay of Bengal bounding box\n",
    "BBOX_BOB = dict(lat_min=5, lat_max=25, lon_min=80, lon_max=100)\n",
    "\n",
    "plt.rcParams.update({\"figure.dpi\": 110, \"savefig.dpi\": 150})\n",
    "print('All libraries loaded.')"
])

# ── 2. open_year_dataset ─────────────────────────────────────────────
md("sec02", ["## 2. Data Loading — `open_year_dataset()`"])

code("load01", [
    "def open_year_dataset(root_dir, year,\n",
    "                      bbox=None):\n",
    "    \"\"\"\n",
    "    Open every daily NetCDF file under root_dir/01 … root_dir/12,\n",
    "    optionally subset each file to *bbox* before concatenation\n",
    "    to keep memory usage low.\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    root_dir : str         – Parent directory containing 01/ … 12/ folders.\n",
    "    year     : int         – Calendar year (used in messages only).\n",
    "    bbox     : dict|None   – {lat_min, lat_max, lon_min, lon_max}.\n",
    "                             If given, each file is clipped before concat.\n",
    "\n",
    "    Returns\n",
    "    -------\n",
    "    xr.Dataset with a `month` coordinate derived from the time dimension.\n",
    "    \"\"\"\n",
    "    file_list = []\n",
    "    for m in range(1, 13):\n",
    "        pattern = os.path.join(root_dir, f\"{m:02d}\", \"*.nc\")\n",
    "        file_list.extend(sorted(glob.glob(pattern)))\n",
    "\n",
    "    if not file_list:\n",
    "        raise FileNotFoundError(\n",
    "            f\"No .nc files found under {root_dir}/01..12 for year {year}.\"\n",
    "        )\n",
    "\n",
    "    print(f\"Found {len(file_list)} daily files. Loading...\", end=\" \", flush=True)\n",
    "\n",
    "    datasets = []\n",
    "    for fp in file_list:\n",
    "        d = xr.open_dataset(fp, engine=\"netcdf4\")\n",
    "        if bbox is not None:\n",
    "            d = d.sel(\n",
    "                latitude=slice(bbox[\"lat_min\"], bbox[\"lat_max\"]),\n",
    "                longitude=slice(bbox[\"lon_min\"], bbox[\"lon_max\"]),\n",
    "            )\n",
    "        datasets.append(d.load())  # load small subset into RAM\n",
    "        d.close()\n",
    "\n",
    "    ds = xr.concat(datasets, dim=\"time\")\n",
    "    ds = ds.assign_coords(month=(\"time\", ds[\"time\"].dt.month.values))\n",
    "    print(\"done.\")\n",
    "    return ds"
])

# ── 3. subset_bbox ───────────────────────────────────────────────────
md("sec03", ["## 3. Spatial Subsetting — `subset_bbox()`"])

code("bbox01", [
    "def subset_bbox(ds, lat_min, lat_max, lon_min, lon_max):\n",
    "    \"\"\"\n",
    "    Return the dataset clipped to a latitude/longitude bounding box.\n",
    "\n",
    "    Handles both ascending and descending latitude coordinates.\n",
    "    Longitudes are assumed 0–360 (standard for this product).\n",
    "    \"\"\"\n",
    "    lats = ds.latitude.values\n",
    "    if lats[0] > lats[-1]:  # descending\n",
    "        lat_slice = slice(lat_max, lat_min)\n",
    "    else:\n",
    "        lat_slice = slice(lat_min, lat_max)\n",
    "\n",
    "    lons = ds.longitude.values\n",
    "    if lons[0] > lons[-1]:\n",
    "        lon_slice = slice(lon_max, lon_min)\n",
    "    else:\n",
    "        lon_slice = slice(lon_min, lon_max)\n",
    "\n",
    "    return ds.sel(latitude=lat_slice, longitude=lon_slice)"
])

# ── 4. Load & subset ─────────────────────────────────────────────────
md("sec04", ["## 4. Load Dataset & Subset to Bay of Bengal"])

code("run01", [
    "# Subset to Bay of Bengal during loading to save memory\n",
    "ds = open_year_dataset(ROOT_DIR, YEAR, bbox=BBOX_BOB)\n",
    "\n",
    "print(f\"Bay of Bengal subset:  \"\n",
    "      f\"lat {BBOX_BOB['lat_min']}–{BBOX_BOB['lat_max']}°N, \"\n",
    "      f\"lon {BBOX_BOB['lon_min']}–{BBOX_BOB['lon_max']}°E\")\n",
    "print(f\"Time steps : {ds.sizes['time']}\")\n",
    "print(f\"Grid       : {ds.sizes['latitude']} x {ds.sizes['longitude']}\")\n",
    "print(f\"Variables  : {list(ds.data_vars)}\")\n",
    "ds"
])

# ── 5. Generic cartopy map plotter ───────────────────────────────────
md("sec05", ["## 5. Map Plotting Utility"])

code("map01", [
    "def plot_map(ax, da, cmap=\"RdBu_r\", vmin=None, vmax=None, title=None):\n",
    "    \"\"\"\n",
    "    Draw a single pcolormesh map on a cartopy GeoAxes.\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    ax   : GeoAxes        – Created with projection=ccrs.PlateCarree().\n",
    "    da   : xr.DataArray   – 2-D (latitude, longitude) field.\n",
    "    cmap : str             – Colormap name.\n",
    "    vmin, vmax : float     – Colour-bar limits (auto from 2nd/98th pctl if None).\n",
    "    title : str            – Subplot title.\n",
    "\n",
    "    Returns\n",
    "    -------\n",
    "    QuadMesh handle (for colorbar).\n",
    "    \"\"\"\n",
    "    proj = ccrs.PlateCarree()\n",
    "    if vmin is None or vmax is None:\n",
    "        vmin = float(da.quantile(0.02))\n",
    "        vmax = float(da.quantile(0.98))\n",
    "\n",
    "    mesh = ax.pcolormesh(\n",
    "        da.longitude, da.latitude, da.values,\n",
    "        transform=proj, cmap=cmap,\n",
    "        vmin=vmin, vmax=vmax, shading=\"auto\",\n",
    "    )\n",
    "    ax.coastlines(resolution=\"50m\", linewidth=0.5)\n",
    "    ax.add_feature(cfeature.LAND, facecolor=\"lightgray\", zorder=2)\n",
    "    gl = ax.gridlines(\n",
    "        draw_labels=True, linewidth=0.2,\n",
    "        color=\"gray\", alpha=0.5, linestyle=\"--\",\n",
    "    )\n",
    "    gl.top_labels = False\n",
    "    gl.right_labels = False\n",
    "    gl.xlabel_style = {\"fontsize\": 6}\n",
    "    gl.ylabel_style = {\"fontsize\": 6}\n",
    "    if title:\n",
    "        ax.set_title(title, fontsize=9)\n",
    "    return mesh"
])

# ── 6. Daily maps for one month ──────────────────────────────────────
md("sec06", ["## 6. Daily Maps for a Selected Month — `plot_daily_month()`"])

code("daily01", [
    "def plot_daily_month(ds, year, month, var_name=\"sla\",\n",
    "                     lat_min=5, lat_max=25, lon_min=80, lon_max=100,\n",
    "                     ncols=5, figsize=(20, 16), cmap=\"RdBu_r\"):\n",
    "    \"\"\"\n",
    "    Grid of cartopy subplots — one panel per day in the given month.\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    ds       : xr.Dataset  – Must already be subset or will be subset here.\n",
    "    year     : int\n",
    "    month    : int         – 1..12\n",
    "    var_name : str         – 'sla', 'adt', 'ugosa', 'vgosa', etc.\n",
    "    ncols    : int         – Columns in the subplot grid.\n",
    "    \"\"\"\n",
    "    dsm = ds.sel(time=ds[\"time\"].dt.month == month)\n",
    "    dsm = subset_bbox(dsm, lat_min, lat_max, lon_min, lon_max)\n",
    "    da = dsm[var_name]\n",
    "\n",
    "    nt = da.sizes[\"time\"]\n",
    "    nrows = int(np.ceil(nt / ncols))\n",
    "\n",
    "    # Consistent colour limits across all days\n",
    "    vmin = float(da.quantile(0.02))\n",
    "    vmax = float(da.quantile(0.98))\n",
    "\n",
    "    proj = ccrs.PlateCarree()\n",
    "    fig, axes = plt.subplots(\n",
    "        nrows, ncols,\n",
    "        subplot_kw={\"projection\": proj},\n",
    "        figsize=figsize,\n",
    "    )\n",
    "    axes = np.array(axes).flatten()\n",
    "\n",
    "    for i in range(nt):\n",
    "        day_slice = da.isel(time=i)\n",
    "        date_str = pd.Timestamp(day_slice.time.values).strftime(\"%d %b\")\n",
    "        mesh = plot_map(\n",
    "            axes[i], day_slice,\n",
    "            cmap=cmap, vmin=vmin, vmax=vmax,\n",
    "            title=f\"{var_name.upper()} {date_str}\",\n",
    "        )\n",
    "\n",
    "    for j in range(nt, len(axes)):\n",
    "        fig.delaxes(axes[j])\n",
    "\n",
    "    cbar = fig.colorbar(\n",
    "        mesh, ax=axes[:nt].tolist(),\n",
    "        orientation=\"horizontal\", shrink=0.6, pad=0.04,\n",
    "    )\n",
    "    cbar.set_label(f\"{var_name.upper()} (m)\")\n",
    "    fig.suptitle(\n",
    "        f\"{var_name.upper()} Daily Maps — {year}-{month:02d} — Bay of Bengal\",\n",
    "        fontsize=14,\n",
    "    )\n",
    "    plt.tight_layout(rect=[0, 0.04, 1, 0.95])\n",
    "    return fig"
])

code("daily02", [
    "# Plot all days in January\n",
    "fig = plot_daily_month(\n",
    "    ds, YEAR, month=1, var_name=\"sla\",\n",
    "    **BBOX_BOB, ncols=5, figsize=(20, 16),\n",
    ")\n",
    "plt.show()"
])

# ── 7. Monthly means 3x4 ─────────────────────────────────────────────
md("sec07", ["## 7. Monthly Mean Maps (3x4 Panel) — `plot_monthly_means()`"])

code("monthly01", [
    "def plot_monthly_means(ds, year, var_name=\"sla\",\n",
    "                       lat_min=5, lat_max=25, lon_min=80, lon_max=100,\n",
    "                       nrows=3, ncols=4, figsize=(16, 10), cmap=\"RdBu_r\"):\n",
    "    \"\"\"\n",
    "    Compute monthly mean of *var_name* and plot as a 3x4 panel with cartopy.\n",
    "    \"\"\"\n",
    "    dsy = subset_bbox(ds, lat_min, lat_max, lon_min, lon_max)\n",
    "    monthly = dsy[var_name].groupby(\"time.month\").mean(dim=\"time\")\n",
    "\n",
    "    vmin = float(monthly.quantile(0.02))\n",
    "    vmax = float(monthly.quantile(0.98))\n",
    "\n",
    "    month_labels = [\"Jan\", \"Feb\", \"Mar\", \"Apr\", \"May\", \"Jun\",\n",
    "                    \"Jul\", \"Aug\", \"Sep\", \"Oct\", \"Nov\", \"Dec\"]\n",
    "\n",
    "    proj = ccrs.PlateCarree()\n",
    "    fig, axes = plt.subplots(\n",
    "        nrows, ncols,\n",
    "        subplot_kw={\"projection\": proj},\n",
    "        figsize=figsize,\n",
    "    )\n",
    "    axes = axes.flatten()\n",
    "\n",
    "    mesh = None\n",
    "    for i in range(12):\n",
    "        m = i + 1\n",
    "        if m in monthly[\"month\"].values:\n",
    "            mesh = plot_map(\n",
    "                axes[i], monthly.sel(month=m),\n",
    "                cmap=cmap, vmin=vmin, vmax=vmax,\n",
    "                title=month_labels[i],\n",
    "            )\n",
    "        else:\n",
    "            axes[i].set_title(f\"{month_labels[i]} (no data)\", fontsize=9)\n",
    "            axes[i].coastlines()\n",
    "\n",
    "    cbar = fig.colorbar(\n",
    "        mesh, ax=axes.tolist(),\n",
    "        orientation=\"horizontal\", shrink=0.6, pad=0.04,\n",
    "    )\n",
    "    cbar.set_label(f\"{var_name.upper()} (m)\")\n",
    "    fig.suptitle(\n",
    "        f\"Monthly Mean {var_name.upper()} — Bay of Bengal {year}\", fontsize=14,\n",
    "    )\n",
    "    plt.tight_layout(rect=[0, 0.04, 1, 0.95])\n",
    "    return fig"
])

code("monthly02", [
    "fig = plot_monthly_means(ds, YEAR, var_name=\"sla\", **BBOX_BOB)\n",
    "plt.show()"
])

# ── 8. Area-averaged time series ─────────────────────────────────────
md("sec08", ["## 8. Area-Averaged Time Series — `plot_area_timeseries()`"])

code("ts01", [
    "def plot_area_timeseries(ds, var_name=\"sla\",\n",
    "                         lat_min=5, lat_max=25, lon_min=80, lon_max=100,\n",
    "                         figsize=(14, 5)):\n",
    "    \"\"\"\n",
    "    Daily area-averaged time series, weighted by cos(latitude) to\n",
    "    account for converging meridians.\n",
    "    \"\"\"\n",
    "    dsy = subset_bbox(ds, lat_min, lat_max, lon_min, lon_max)\n",
    "    weights = np.cos(np.deg2rad(dsy.latitude))\n",
    "    ts = dsy[var_name].weighted(weights).mean(dim=[\"latitude\", \"longitude\"])\n",
    "    ts_pd = ts.to_series()\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=figsize)\n",
    "    ax.plot(ts_pd.index, ts_pd.values, lw=0.8, color=\"steelblue\")\n",
    "    annual_mean = ts_pd.mean()\n",
    "    ax.axhline(annual_mean, ls=\"--\", color=\"red\", lw=0.7,\n",
    "               label=f\"Annual mean = {annual_mean:.4f} m\")\n",
    "    ax.fill_between(ts_pd.index, ts_pd.values, annual_mean, alpha=0.15)\n",
    "    ax.set_xlabel(\"Date\")\n",
    "    ax.set_ylabel(f\"{var_name.upper()} (m)\")\n",
    "    ax.set_title(\n",
    "        f\"Area-Averaged {var_name.upper()} — Bay of Bengal \"\n",
    "        f\"{int(ds.time.dt.year.values[0])}\"\n",
    "    )\n",
    "    ax.legend()\n",
    "    ax.grid(alpha=0.3)\n",
    "    plt.tight_layout()\n",
    "    return fig"
])

code("ts02", [
    "fig = plot_area_timeseries(ds, var_name=\"sla\", **BBOX_BOB)\n",
    "plt.show()"
])

# ── 9. Anomaly maps ──────────────────────────────────────────────────
md("sec09", ["## 9. Monthly Anomaly Maps — `plot_monthly_anomalies()`"])

code("anom01", [
    "def plot_monthly_anomalies(ds, year, var_name=\"sla\",\n",
    "                           lat_min=5, lat_max=25, lon_min=80, lon_max=100,\n",
    "                           nrows=3, ncols=4, figsize=(16, 10), cmap=\"RdBu_r\"):\n",
    "    \"\"\"\n",
    "    Monthly anomaly = monthly_mean - annual_mean. Plotted as 3x4 panel.\n",
    "    \"\"\"\n",
    "    dsy = subset_bbox(ds, lat_min, lat_max, lon_min, lon_max)\n",
    "    annual_mean = dsy[var_name].mean(dim=\"time\")\n",
    "    monthly = dsy[var_name].groupby(\"time.month\").mean(dim=\"time\")\n",
    "    anomalies = monthly - annual_mean\n",
    "\n",
    "    vabs = float(np.abs(anomalies).quantile(0.98))\n",
    "\n",
    "    month_labels = [\"Jan\", \"Feb\", \"Mar\", \"Apr\", \"May\", \"Jun\",\n",
    "                    \"Jul\", \"Aug\", \"Sep\", \"Oct\", \"Nov\", \"Dec\"]\n",
    "\n",
    "    proj = ccrs.PlateCarree()\n",
    "    fig, axes = plt.subplots(\n",
    "        nrows, ncols,\n",
    "        subplot_kw={\"projection\": proj},\n",
    "        figsize=figsize,\n",
    "    )\n",
    "    axes = axes.flatten()\n",
    "\n",
    "    for i in range(12):\n",
    "        anom = anomalies.sel(month=i + 1)\n",
    "        mesh = plot_map(\n",
    "            axes[i], anom,\n",
    "            cmap=cmap, vmin=-vabs, vmax=vabs,\n",
    "            title=month_labels[i],\n",
    "        )\n",
    "\n",
    "    cbar = fig.colorbar(\n",
    "        mesh, ax=axes.tolist(),\n",
    "        orientation=\"horizontal\", shrink=0.6,\n",
    "        label=f\"{var_name.upper()} anomaly (m)\",\n",
    "    )\n",
    "    fig.suptitle(\n",
    "        f\"{var_name.upper()} Monthly Anomaly (vs Annual Mean) — \"\n",
    "        f\"Bay of Bengal {year}\",\n",
    "        fontsize=14,\n",
    "    )\n",
    "    plt.tight_layout(rect=[0, 0.04, 1, 0.95])\n",
    "    return fig"
])

code("anom02", [
    "fig = plot_monthly_anomalies(ds, YEAR, var_name=\"sla\", **BBOX_BOB)\n",
    "plt.show()"
])

# ── 10. Monthly statistics summary ───────────────────────────────────
md("sec10", ["## 10. Monthly Statistics Summary"])

code("stats01", [
    "month_names = [\"Jan\", \"Feb\", \"Mar\", \"Apr\", \"May\", \"Jun\",\n",
    "               \"Jul\", \"Aug\", \"Sep\", \"Oct\", \"Nov\", \"Dec\"]\n",
    "\n",
    "ds_bob = subset_bbox(ds, **BBOX_BOB)\n",
    "rows = []\n",
    "for m in range(1, 13):\n",
    "    da_m = ds_bob[\"sla\"].sel(time=ds_bob[\"time\"].dt.month == m)\n",
    "    vals = da_m.values.ravel()\n",
    "    valid = vals[np.isfinite(vals)]\n",
    "    rows.append({\n",
    "        \"Month\": month_names[m - 1],\n",
    "        \"Days\": int(da_m.sizes[\"time\"]),\n",
    "        \"Mean (m)\": np.mean(valid),\n",
    "        \"Std (m)\": np.std(valid),\n",
    "        \"Min (m)\": np.min(valid),\n",
    "        \"Max (m)\": np.max(valid),\n",
    "    })\n",
    "\n",
    "df_stats = pd.DataFrame(rows)\n",
    "print(df_stats.to_string(index=False))\n",
    "df_stats.to_csv(\"monthly_sla_statistics_2024.csv\", index=False)\n",
    "print(\"\\nSaved: monthly_sla_statistics_2024.csv\")"
])

# ── 11. Trend figure ─────────────────────────────────────────────────
md("sec11", ["## 11. Trend & Variability Analysis"])

code("trend01", [
    "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n",
    "fig.suptitle(\"SLA Monthly Trends — Bay of Bengal 2024\", fontsize=14, fontweight=\"bold\")\n",
    "\n",
    "x = np.arange(1, 13)\n",
    "\n",
    "# Mean +/- std\n",
    "ax = axes[0, 0]\n",
    "ax.plot(x, df_stats[\"Mean (m)\"], \"o-\", color=\"steelblue\", label=\"Mean\")\n",
    "ax.fill_between(x,\n",
    "                df_stats[\"Mean (m)\"] - df_stats[\"Std (m)\"],\n",
    "                df_stats[\"Mean (m)\"] + df_stats[\"Std (m)\"],\n",
    "                alpha=0.25, color=\"steelblue\")\n",
    "ax.set_ylabel(\"SLA (m)\")\n",
    "ax.set_title(\"Mean +/- 1 Std\")\n",
    "ax.set_xticks(x); ax.set_xticklabels(month_names, rotation=45, ha=\"right\")\n",
    "ax.grid(alpha=0.3)\n",
    "ax.legend()\n",
    "\n",
    "# Min / Max range\n",
    "ax = axes[0, 1]\n",
    "ax.plot(x, df_stats[\"Min (m)\"], \"s-\", color=\"#ff7f0e\", label=\"Min\")\n",
    "ax.plot(x, df_stats[\"Max (m)\"], \"^-\", color=\"#d62728\", label=\"Max\")\n",
    "ax.fill_between(x, df_stats[\"Min (m)\"], df_stats[\"Max (m)\"], alpha=0.15, color=\"gray\")\n",
    "ax.set_ylabel(\"SLA (m)\")\n",
    "ax.set_title(\"Min–Max Range\")\n",
    "ax.set_xticks(x); ax.set_xticklabels(month_names, rotation=45, ha=\"right\")\n",
    "ax.grid(alpha=0.3)\n",
    "ax.legend()\n",
    "\n",
    "# Std bar\n",
    "ax = axes[1, 0]\n",
    "ax.bar(x, df_stats[\"Std (m)\"], color=\"#2ca02c\", alpha=0.7, edgecolor=\"black\")\n",
    "ax.set_ylabel(\"Std Dev (m)\")\n",
    "ax.set_title(\"Monthly Variability\")\n",
    "ax.set_xticks(x); ax.set_xticklabels(month_names, rotation=45, ha=\"right\")\n",
    "ax.grid(alpha=0.3, axis=\"y\")\n",
    "\n",
    "# Days per month\n",
    "ax = axes[1, 1]\n",
    "ax.bar(x, df_stats[\"Days\"], color=\"#9467bd\", alpha=0.7, edgecolor=\"black\")\n",
    "ax.set_ylabel(\"Files\")\n",
    "ax.set_title(\"Data Availability\")\n",
    "ax.set_xticks(x); ax.set_xticklabels(month_names, rotation=45, ha=\"right\")\n",
    "ax.grid(alpha=0.3, axis=\"y\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"sla_trends_2024.png\", bbox_inches=\"tight\")\n",
    "print(\"Saved: sla_trends_2024.png\")\n",
    "plt.show()"
])

# ── 12. Custom region ────────────────────────────────────────────────
md("sec12", [
    "## 12. Custom Region Example\n",
    "\n",
    "Change `BBOX_CUSTOM` to any sub-region and re-use the same functions."
])

code("custom01", [
    "# Central Bay of Bengal eddy hotspot\n",
    "BBOX_CUSTOM = dict(lat_min=10, lat_max=18, lon_min=85, lon_max=95)\n",
    "\n",
    "# Monthly means for ADT in the custom region\n",
    "fig = plot_monthly_means(\n",
    "    ds, YEAR, var_name=\"adt\", **BBOX_CUSTOM,\n",
    ")\n",
    "plt.show()\n",
    "\n",
    "# Time series for SLA in the custom region\n",
    "fig2 = plot_area_timeseries(ds, var_name=\"sla\", **BBOX_CUSTOM)\n",
    "plt.show()"
])

# ── 13. Other variables ──────────────────────────────────────────────
md("sec13", [
    "## 13. Other Variables: UGOSA, VGOSA\n",
    "\n",
    "All plotting functions accept a `var_name` argument."
])

code("other01", [
    "# Geostrophic velocity anomaly — monthly means\n",
    "fig = plot_monthly_means(ds, YEAR, var_name=\"ugosa\", **BBOX_BOB, cmap=\"PiYG\")\n",
    "plt.show()\n",
    "\n",
    "fig = plot_monthly_means(ds, YEAR, var_name=\"vgosa\", **BBOX_BOB, cmap=\"PiYG\")\n",
    "plt.show()"
])

# ── Write notebook ───────────────────────────────────────────────────
output_path = (
    r"c:\Users\abhik\Desktop\project related work"
    r"\CMEMS_Monthly_Visualization_2024.ipynb"
)

with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

file_size = os.path.getsize(output_path)
print(f"Notebook written: {output_path}")
print(f"  Size : {file_size:,} bytes ({file_size / 1024:.1f} KB)")
print(f"  Cells: {len(notebook['cells'])}")
