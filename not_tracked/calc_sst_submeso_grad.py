import pandas as pd
import numpy as np
import json
import os
from not_tracked.cyclone_anomaly_analysis import (
    _get_peak_date, _make_square_bbox, _get_analysis_window_ranges,
    _get_analysis_date_list_from_range, load_sst_for_date, decompose_sst_field
)

def calculate_sst_submeso_gradients(cyclones_dict):
    records = []
    for cyc_name, cyc in cyclones_dict.items():
        print(f"Processing {cyc_name} ...", end=' ', flush=True)

        bbox = cyc.get('bbox')
        track = cyc.get('track')
        if not bbox or not track:
            print("skipped (no bbox/track)")
            continue

        bbox = _make_square_bbox(bbox, track)
        peak_date = _get_peak_date(cyc)
        if peak_date is None:
            print("skipped (no peak date)")
            continue

        windows = _get_analysis_window_ranges(cyc)
        if not windows:
            print("skipped (no analysis window)")
            continue

        # Target Signal (from During/Post window)
        _, start_str, end_str = windows[-1]
        date_list = _get_analysis_date_list_from_range(start_str, end_str)

        sst_sub_grad_vals = []
        ref_lat = (bbox['lat_min'] + bbox['lat_max']) / 2.0
        # SST grid resolution is 0.05 degrees
        dy_km = 111.0 * 0.05
        dx_km = 111.0 * np.cos(np.deg2rad(ref_lat)) * 0.05

        for date in date_list:
            sst_raw = load_sst_for_date(date, bbox)
            if sst_raw is not None:
                dec = decompose_sst_field(sst_raw)
                if dec and 'submeso' in dec:
                    submeso_field = dec['submeso']
                    # Compute gradient (axis 0 = lat/y, axis 1 = lon/x)
                    grad_y, grad_x = np.gradient(submeso_field)
                    
                    # Convert to physical units: degrees Celsius per km
                    grad_y_km = grad_y / dy_km
                    grad_x_km = grad_x / dx_km
                    
                    # Magnitude of the gradient
                    grad_mag = np.sqrt(grad_x_km**2 + grad_y_km**2)
                    
                    # Average over the region
                    mean_grad = np.nanmean(grad_mag)
                    sst_sub_grad_vals.append(mean_grad)

        if sst_sub_grad_vals:
            # Average over the dates
            mean_grad_for_cyc = np.nanmean(sst_sub_grad_vals)
            rec = {
                'Cyclone': cyc_name,
                'SST_Submeso_Grad_Avg': mean_grad_for_cyc
            }
            print(f"Grad={mean_grad_for_cyc:.5f} °C/km")
            records.append(rec)
        else:
            print("No data.")

    return pd.DataFrame(records)

if __name__ == "__main__":
    json_path = 'cyclone_info_2024_padded.json'
    csv_file = 'cyclone_multiscale_anomalies.csv'
    
    if os.path.exists(json_path) and os.path.exists(csv_file):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        CYCLONES = {}
        for basin, b_data in raw_data.items():
            for storm in b_data.get('storms', []):
                name = storm.get('name', '').upper()
                CYCLONES[name] = storm
            
        grad_df = calculate_sst_submeso_gradients(CYCLONES)
        
        # Merge with existing CSV
        df = pd.read_csv(csv_file)
        if 'SST_Submeso_Grad_Avg' in df.columns:
            df = df.drop(columns=['SST_Submeso_Grad_Avg'])
            
        df = pd.merge(df, grad_df, on='Cyclone', how='left')
        df.to_csv(csv_file, index=False)
        print(f"\nAppended SST_Submeso_Grad_Avg to {csv_file}")
    else:
        print("Missing json or csv file.")
