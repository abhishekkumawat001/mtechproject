"""
SWOT Cross-Swath Bias Correction Module
========================================

This module provides functions to apply cross-swath bias correction to 
SWOT L2 LR SSH Unsmoothed data files.

Based on SWOT Product Description Document (Revision C, Feb 2025):
- Computes SSHA = ssh_karin_2 - mean_sea_surface_cnescls
- Applies height_cor_xover (crossover calibration)
- Removes cross-swath bias using along-track mean subtraction

Usage:
------
    from swot_crossswath_correction import process_file, process_folder
    
    # Process single file
    result = process_file('path/to/file.nc', lat_bounds=(5, 20))
    
    # Process entire folder
    results = process_folder('path/to/folder', lat_bounds=(5, 20))

Author: Generated for SWOT SSH Analysis
Date: January 2026
"""

import numpy as np
import xarray as xr
import os
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def distance_between_points(lon0: np.ndarray, lons: np.ndarray, 
                           lat0: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Calculate distance between points using haversine formula.
    
    Parameters
    ----------
    lon0, lat0 : array-like
        Reference longitude and latitude
    lons, lats : array-like
        Target longitudes and latitudes
        
    Returns
    -------
    np.ndarray
        Distance in meters
    """
    degrees_to_radians = np.pi / 180.0
    phi1 = lat0 * degrees_to_radians
    phi2 = lats * degrees_to_radians
    dphi = phi1 - phi2
    theta1 = lon0 * degrees_to_radians
    theta2 = lons * degrees_to_radians
    dtheta = theta1 - theta2
    
    co = np.sqrt(np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dtheta/2.0)**2)
    arc = 2 * np.arcsin(co)
    dist = arc * 6371.0e3  # Earth radius in meters
    return dist


def calculate_cross_track_distance(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate cross-track distance from the nadir (center) of the swath.
    
    Parameters
    ----------
    lat : np.ndarray
        2D latitude array (num_lines x num_pixels)
    lon : np.ndarray
        2D longitude array (num_lines x num_pixels)
        
    Returns
    -------
    np.ndarray
        Cross-track distance in meters (negative for left side, positive for right)
    """
    # Get nadir position (center of swath)
    nadir_lon = np.nanmean(lon, axis=1, keepdims=True)
    nadir_lat = np.nanmean(lat, axis=1, keepdims=True)
    
    # Calculate distance from nadir
    cross_track_dist = distance_between_points(nadir_lon, lon, nadir_lat, lat)
    
    # Assign sign based on position relative to nadir
    num_pixels = lon.shape[1]
    center_idx = num_pixels // 2
    
    # Pixels to the left of center get negative distance
    cross_track_dist[:, :center_idx] = -cross_track_dist[:, :center_idx]
    
    return cross_track_dist


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_swath(ds_swath: xr.Dataset, 
                  lat_bounds: Optional[Tuple[float, float]] = None,
                  apply_ocean_mask: bool = True,
                  fill_threshold: float = 1e10,
                  ssha_threshold: float = 5.0) -> Dict:
    """
    Process a single swath (left or right) and apply cross-swath bias correction.
    
    Parameters
    ----------
    ds_swath : xr.Dataset
        xarray Dataset for the swath (left or right group)
    lat_bounds : tuple, optional
        (lat_min, lat_max) for latitude-bounded mean calculation
    apply_ocean_mask : bool
        Whether to apply ocean mask (flag == 0)
    fill_threshold : float
        Threshold for fill value detection (default: 1e10)
    ssha_threshold : float
        Maximum valid SSHA magnitude in meters (default: 5.0)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'lat': latitude array
        - 'lon': longitude array
        - 'ssha_original': SSHA before cross-swath correction
        - 'ssha_corrected': SSHA after cross-swath correction
        - 'bias_removed': The bias that was removed
        - 'cross_track_dist': Cross-track distance array
        - 'xover_applied': Whether crossover correction was applied
        - 'valid_points': Number of valid data points
    """
    # Extract coordinates
    lat = ds_swath['latitude'].values
    lon = ds_swath['longitude'].values
    
    # Get SSH and MSS
    ssh_raw = ds_swath['ssh_karin_2'].values.copy()
    mss_raw = ds_swath['mean_sea_surface_cnescls'].values.copy()
    
    # Filter fill values
    ssh = np.where(np.abs(ssh_raw) > fill_threshold, np.nan, ssh_raw)
    mss = np.where(np.abs(mss_raw) > fill_threshold, np.nan, mss_raw)
    
    # Physical bounds for SSH
    ssh = np.where(np.abs(ssh) > 150, np.nan, ssh)
    mss = np.where(np.abs(mss) > 150, np.nan, mss)
    
    # Compute SSHA
    ssha = ssh - mss
    
    # Filter unrealistic SSHA
    ssha = np.where(np.abs(ssha) > ssha_threshold, np.nan, ssha)
    
    # Apply crossover correction if available
    xover_applied = False
    if 'height_cor_xover' in ds_swath:
        xover_raw = ds_swath['height_cor_xover'].values.copy()
        xover = np.where(np.abs(xover_raw) > fill_threshold, np.nan, xover_raw)
        
        valid_xover = np.sum(~np.isnan(xover))
        if valid_xover > 0:
            ssha = ssha + xover  # ADD correction (not subtract!)
            xover_applied = True
    
    # Apply ocean mask if requested
    if apply_ocean_mask and 'ancillary_surface_classification_flag' in ds_swath:
        ocean_mask = ds_swath['ancillary_surface_classification_flag'].values == 0
        ssha = np.where(ocean_mask, ssha, np.nan)
    
    # Calculate cross-track distance
    cross_track_dist = calculate_cross_track_distance(lat, lon)
    
    # Apply cross-swath bias correction (along-track mean subtraction)
    if lat_bounds is not None:
        lat_min, lat_max = lat_bounds
        lat_col = lat[:, lat.shape[1] // 2]
        lat_col = np.where(np.isnan(lat_col), -999, lat_col)
        lat_mask = (lat_col > lat_min) & (lat_col < lat_max)
        
        if np.sum(lat_mask) > 100:
            along_track_mean = np.nanmean(ssha[lat_mask, :], axis=0, keepdims=True)
        else:
            along_track_mean = np.nanmean(ssha, axis=0, keepdims=True)
    else:
        along_track_mean = np.nanmean(ssha, axis=0, keepdims=True)
    
    ssha_corrected = ssha - along_track_mean
    
    return {
        'lat': lat,
        'lon': lon,
        'ssha_original': ssha,
        'ssha_corrected': ssha_corrected,
        'bias_removed': along_track_mean,
        'cross_track_dist': cross_track_dist,
        'xover_applied': xover_applied,
        'valid_points': int(np.sum(~np.isnan(ssha_corrected)))
    }


def process_file(filepath: str,
                 lat_bounds: Optional[Tuple[float, float]] = None,
                 lon_bounds: Optional[Tuple[float, float]] = None,
                 apply_ocean_mask: bool = True,
                 return_datasets: bool = False) -> Dict:
    """
    Process a single SWOT Unsmoothed SSH file and apply cross-swath bias correction.
    
    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    lat_bounds : tuple, optional
        (lat_min, lat_max) for latitude-bounded mean calculation
    lon_bounds : tuple, optional
        (lon_min, lon_max) for regional subsetting (not used in correction, only for info)
    apply_ocean_mask : bool
        Whether to apply ocean mask
    return_datasets : bool
        Whether to return the original xarray datasets
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'filepath': Original file path
        - 'filename': File name
        - 'left': Processed left swath data
        - 'right': Processed right swath data
        - 'metadata': File metadata (cycle, pass, time, etc.)
        - 'ds_left', 'ds_right', 'ds_root': Original datasets (if return_datasets=True)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Open datasets
    ds_root = xr.open_dataset(filepath)
    ds_left = xr.open_dataset(filepath, group='left', engine='netcdf4')
    ds_right = xr.open_dataset(filepath, group='right', engine='netcdf4')
    
    # Extract metadata
    metadata = {
        'cycle_number': ds_root.attrs.get('cycle_number', None),
        'pass_number': ds_root.attrs.get('pass_number', None),
        'time_coverage_start': ds_root.attrs.get('time_coverage_start', None),
        'time_coverage_end': ds_root.attrs.get('time_coverage_end', None),
        'geospatial_lat_min': ds_root.attrs.get('geospatial_lat_min', None),
        'geospatial_lat_max': ds_root.attrs.get('geospatial_lat_max', None),
        'geospatial_lon_min': ds_root.attrs.get('geospatial_lon_min', None),
        'geospatial_lon_max': ds_root.attrs.get('geospatial_lon_max', None),
    }
    
    # Process both swaths
    left_result = process_swath(ds_left, lat_bounds, apply_ocean_mask)
    right_result = process_swath(ds_right, lat_bounds, apply_ocean_mask)
    
    result = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'left': left_result,
        'right': right_result,
        'metadata': metadata
    }
    
    if return_datasets:
        result['ds_left'] = ds_left
        result['ds_right'] = ds_right
        result['ds_root'] = ds_root
    else:
        ds_left.close()
        ds_right.close()
        ds_root.close()
    
    return result


def process_folder(folder_path: str,
                   pattern: str = "*.nc",
                   lat_bounds: Optional[Tuple[float, float]] = None,
                   lon_bounds: Optional[Tuple[float, float]] = None,
                   apply_ocean_mask: bool = True,
                   verbose: bool = True) -> List[Dict]:
    """
    Process all SWOT Unsmoothed SSH files in a folder.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing NetCDF files
    pattern : str
        Glob pattern for file matching (default: "*.nc")
    lat_bounds : tuple, optional
        (lat_min, lat_max) for latitude-bounded mean calculation
    lon_bounds : tuple, optional
        (lon_min, lon_max) for regional info
    apply_ocean_mask : bool
        Whether to apply ocean mask
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    list
        List of result dictionaries, one per file
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")
    
    # Find all matching files
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    
    if len(files) == 0:
        print(f"No files found matching pattern '{pattern}' in {folder_path}")
        return []
    
    if verbose:
        print("=" * 70)
        print("SWOT Cross-Swath Bias Correction - Batch Processing")
        print("=" * 70)
        print(f"Folder: {folder_path}")
        print(f"Files found: {len(files)}")
        if lat_bounds:
            print(f"Latitude bounds: {lat_bounds[0]}° to {lat_bounds[1]}°")
        print("=" * 70)
    
    results = []
    successful = 0
    failed = 0
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        
        try:
            if verbose:
                print(f"\n[{i+1}/{len(files)}] Processing: {filename}")
            
            result = process_file(
                filepath,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
                apply_ocean_mask=apply_ocean_mask
            )
            
            results.append(result)
            successful += 1
            
            if verbose:
                left_pts = result['left']['valid_points']
                right_pts = result['right']['valid_points']
                xover_l = "Yes" if result['left']['xover_applied'] else "No"
                xover_r = "Yes" if result['right']['xover_applied'] else "No"
                print(f"    ✓ Left: {left_pts:,} pts (xover: {xover_l})")
                print(f"    ✓ Right: {right_pts:,} pts (xover: {xover_r})")
                
        except Exception as e:
            failed += 1
            if verbose:
                print(f"    ✗ Error: {str(e)}")
            results.append({
                'filepath': filepath,
                'filename': filename,
                'error': str(e)
            })
    
    if verbose:
        print("\n" + "=" * 70)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"  Successful: {successful}/{len(files)}")
        print(f"  Failed: {failed}/{len(files)}")
        print("=" * 70)
    
    return results


def extract_region(result: Dict,
                   lat_bounds: Tuple[float, float],
                   lon_bounds: Tuple[float, float],
                   swath: str = 'both') -> Dict:
    """
    Extract data for a specific geographic region from processed results.
    
    Parameters
    ----------
    result : dict
        Result from process_file()
    lat_bounds : tuple
        (lat_min, lat_max)
    lon_bounds : tuple
        (lon_min, lon_max)
    swath : str
        'left', 'right', or 'both'
        
    Returns
    -------
    dict
        Extracted data with regional mask applied
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    
    extracted = {'metadata': result['metadata']}
    
    swaths_to_process = ['left', 'right'] if swath == 'both' else [swath]
    
    for s in swaths_to_process:
        data = result[s]
        lat, lon = data['lat'], data['lon']
        
        # Create regional mask
        region_mask = (lat >= lat_min) & (lat <= lat_max) & \
                      (lon >= lon_min) & (lon <= lon_max)
        valid_mask = ~np.isnan(data['ssha_corrected'])
        combined_mask = region_mask & valid_mask
        
        extracted[s] = {
            'lat': lat[combined_mask],
            'lon': lon[combined_mask],
            'ssha_original': data['ssha_original'][combined_mask],
            'ssha_corrected': data['ssha_corrected'][combined_mask],
            'cross_track_dist': data['cross_track_dist'][combined_mask],
            'n_points': int(np.sum(combined_mask))
        }
    
    return extracted


def save_corrected_data(result: Dict, 
                        output_path: str,
                        include_original: bool = True) -> str:
    """
    Save corrected SSHA data to a new NetCDF file.
    
    Parameters
    ----------
    result : dict
        Result from process_file()
    output_path : str
        Path for output file
    include_original : bool
        Whether to include original SSHA in output
        
    Returns
    -------
    str
        Path to saved file
    """
    import xarray as xr
    
    # Create output dataset
    ds_out = xr.Dataset()
    
    for swath_name in ['left', 'right']:
        data = result[swath_name]
        prefix = f"{swath_name}_"
        
        # Add coordinates
        ds_out[f"{prefix}latitude"] = xr.DataArray(
            data['lat'], 
            dims=['num_lines', 'num_pixels'],
            attrs={'units': 'degrees_north', 'long_name': 'Latitude'}
        )
        ds_out[f"{prefix}longitude"] = xr.DataArray(
            data['lon'], 
            dims=['num_lines', 'num_pixels'],
            attrs={'units': 'degrees_east', 'long_name': 'Longitude'}
        )
        
        # Add corrected SSHA
        ds_out[f"{prefix}ssha_corrected"] = xr.DataArray(
            data['ssha_corrected'],
            dims=['num_lines', 'num_pixels'],
            attrs={
                'units': 'm',
                'long_name': 'Sea Surface Height Anomaly (Cross-Swath Bias Corrected)',
                'comment': 'SSHA with cross-swath bias removed using along-track mean subtraction'
            }
        )
        
        if include_original:
            ds_out[f"{prefix}ssha_original"] = xr.DataArray(
                data['ssha_original'],
                dims=['num_lines', 'num_pixels'],
                attrs={
                    'units': 'm',
                    'long_name': 'Sea Surface Height Anomaly (Original)',
                    'comment': 'SSHA with crossover correction applied but before cross-swath bias removal'
                }
            )
        
        # Add cross-track distance
        ds_out[f"{prefix}cross_track_distance"] = xr.DataArray(
            data['cross_track_dist'],
            dims=['num_lines', 'num_pixels'],
            attrs={'units': 'm', 'long_name': 'Cross-track distance from nadir'}
        )
    
    # Add global attributes
    ds_out.attrs = {
        'title': 'SWOT Cross-Swath Bias Corrected SSH',
        'source_file': result['filename'],
        'processing_date': datetime.now().isoformat(),
        'correction_method': 'Along-track mean subtraction',
        **result['metadata']
    }
    
    # Save to file
    ds_out.to_netcdf(output_path)
    
    return output_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_statistics(result: Dict) -> Dict:
    """
    Get summary statistics for processed data.
    
    Parameters
    ----------
    result : dict
        Result from process_file()
        
    Returns
    -------
    dict
        Statistics for each swath
    """
    stats = {'metadata': result['metadata']}
    
    for swath_name in ['left', 'right']:
        data = result[swath_name]
        
        stats[swath_name] = {
            'valid_points': data['valid_points'],
            'xover_applied': data['xover_applied'],
            'ssha_original': {
                'mean': float(np.nanmean(data['ssha_original'])),
                'std': float(np.nanstd(data['ssha_original'])),
                'min': float(np.nanmin(data['ssha_original'])),
                'max': float(np.nanmax(data['ssha_original']))
            },
            'ssha_corrected': {
                'mean': float(np.nanmean(data['ssha_corrected'])),
                'std': float(np.nanstd(data['ssha_corrected'])),
                'min': float(np.nanmin(data['ssha_corrected'])),
                'max': float(np.nanmax(data['ssha_corrected']))
            },
            'bias_removed': {
                'mean': float(np.nanmean(data['bias_removed'])),
                'std': float(np.nanstd(data['bias_removed']))
            }
        }
    
    return stats


# =============================================================================
# MAIN ENTRY POINT (for command-line usage)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Apply cross-swath bias correction to SWOT Unsmoothed SSH files'
    )
    parser.add_argument('input', help='Input file or folder path')
    parser.add_argument('-o', '--output', help='Output folder for corrected files')
    parser.add_argument('--lat-min', type=float, help='Minimum latitude for bounded mean')
    parser.add_argument('--lat-max', type=float, help='Maximum latitude for bounded mean')
    parser.add_argument('--no-ocean-mask', action='store_true', 
                        help='Do not apply ocean mask')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Print detailed progress')
    
    args = parser.parse_args()
    
    # Set up latitude bounds
    lat_bounds = None
    if args.lat_min is not None and args.lat_max is not None:
        lat_bounds = (args.lat_min, args.lat_max)
    
    # Process input
    if os.path.isfile(args.input):
        # Single file
        result = process_file(
            args.input,
            lat_bounds=lat_bounds,
            apply_ocean_mask=not args.no_ocean_mask
        )
        
        stats = get_statistics(result)
        print(f"\nFile: {result['filename']}")
        for swath in ['left', 'right']:
            s = stats[swath]
            print(f"\n{swath.upper()} SWATH:")
            print(f"  Valid points: {s['valid_points']:,}")
            print(f"  Crossover applied: {s['xover_applied']}")
            print(f"  Original SSHA std: {s['ssha_original']['std']*100:.2f} cm")
            print(f"  Corrected SSHA std: {s['ssha_corrected']['std']*100:.2f} cm")
        
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            out_file = os.path.join(args.output, 
                                    result['filename'].replace('.nc', '_corrected.nc'))
            save_corrected_data(result, out_file)
            print(f"\nSaved: {out_file}")
            
    elif os.path.isdir(args.input):
        # Folder
        results = process_folder(
            args.input,
            lat_bounds=lat_bounds,
            apply_ocean_mask=not args.no_ocean_mask,
            verbose=args.verbose or True
        )
        
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            for result in results:
                if 'error' not in result:
                    out_file = os.path.join(args.output,
                                            result['filename'].replace('.nc', '_corrected.nc'))
                    save_corrected_data(result, out_file)
            print(f"\nCorrected files saved to: {args.output}")
    else:
        print(f"Error: {args.input} is not a valid file or folder")
