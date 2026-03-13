"""
Visualize CMEMS Sea Level/SSH data with monthwise temperature subplots for 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import xarray for netCDF handling
try:
    import xarray as xr
except ImportError:
    print("xarray not found. Installing...")
    os.system("pip install xarray netCDF4")
    import xarray as xr


def explore_data_structure(data_path):
    """Explore the structure of data in the given path"""
    print(f"Exploring data in: {data_path}")
    print("\nDirectory contents:")
    
    if not os.path.exists(data_path):
        print(f"Path does not exist: {data_path}")
        return None
    
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        
        for file in files[:10]:  # Limit to first 10 files per directory
            print(f'{subindent}{file}')
        
        if len(files) > 10:
            print(f'{subindent}... and {len(files) - 10} more files')
    
    return True


def load_and_process_data(data_path, year=2024):
    """
    Load netCDF files and process monthly data
    """
    print(f"\nLoading data for year {year}...")
    
    monthly_data = {}
    
    # Find all netCDF files
    nc_files = list(Path(data_path).rglob("*.nc"))
    
    if not nc_files:
        # Try without recursion
        nc_files = [f for f in Path(data_path).glob("*.nc")]
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    if not nc_files:
        print("No NetCDF files found. Looking for other formats...")
        return None
    
    # Process files by month
    for nc_file in sorted(nc_files):
        try:
            print(f"\nProcessing: {nc_file.name}")
            
            # Load with xarray
            ds = xr.open_dataset(nc_file)
            print(f"  Variables: {list(ds.data_vars)}")
            print(f"  Dimensions: {dict(ds.dims)}")
            
            # Extract date from filename or dataset
            if 'time' in ds.dims:
                time_var = ds['time']
                print(f"  Time range: {time_var.values[0]} to {time_var.values[-1]}")
                
                # Get month from time
                import pandas as pd
                dates = pd.to_datetime(time_var.values)
                month = dates[0].month
                
                # Look for temperature or SSH variable
                temp_var = None
                for var_name in ['sst', 'temperature', 'temp', 'adt', 'ssh']:
                    if var_name in ds.data_vars:
                        temp_var = var_name
                        break
                
                if temp_var:
                    data = ds[temp_var].values
                    if hasattr(data, 'shape'):
                        print(f"  {temp_var} shape: {data.shape}")
                        monthly_data[month] = {
                            'data': data,
                            'variable': temp_var,
                            'file': nc_file.name,
                            'dataset': ds
                        }
                else:
                    # Use first available variable
                    first_var = list(ds.data_vars)[0]
                    print(f"  Using variable: {first_var}")
                    monthly_data[month] = {
                        'data': ds[first_var].values,
                        'variable': first_var,
                        'file': nc_file.name,
                        'dataset': ds
                    }
            
            ds.close()
            
        except Exception as e:
            print(f"  Error processing file: {e}")
            continue
    
    return monthly_data


def visualize_monthly_data(monthly_data, year=2024):
    """
    Create subplots for each month of available data
    """
    if not monthly_data:
        print("No data available to visualize")
        return
    
    months = sorted(monthly_data.keys())
    n_months = len(months)
    
    # Create subplots
    n_cols = 3
    n_rows = (n_months + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    for idx, month in enumerate(months):
        ax = axes[idx]
        
        month_info = monthly_data[month]
        data = month_info['data']
        var_name = month_info['variable']
        
        # Handle different data shapes
        if len(data.shape) == 3:  # time, lat, lon
            data_2d = data[0, :, :]  # Take first time step
        elif len(data.shape) == 2:  # lat, lon
            data_2d = data
        else:
            # Flatten or average as needed
            data_2d = data.reshape(-1, 1) if len(data.shape) == 1 else data
        
        # Plot
        im = ax.imshow(data_2d, cmap='viridis', aspect='auto')
        ax.set_title(f'{month_names[month]} 2024\n{var_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude Index')
        ax.set_ylabel('Latitude Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(var_name)
        
        # Add statistics
        mean_val = np.nanmean(data_2d)
        min_val = np.nanmin(data_2d)
        max_val = np.nanmax(data_2d)
        
        stats_text = f'Mean: {mean_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(n_months, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Monthly {var_name} Visualization - 2024', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_path = 'monthly_sealevel_visualization_2024.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


def create_statistical_summary(monthly_data, year=2024):
    """
    Create statistical summary of monthly data
    """
    print("\n" + "="*60)
    print(f"STATISTICAL SUMMARY - {year}")
    print("="*60)
    
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    summary_data = []
    
    for month in sorted(monthly_data.keys()):
        month_info = monthly_data[month]
        data = month_info['data']
        var_name = month_info['variable']
        
        # Flatten data for statistics
        flat_data = data.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]
        
        summary_data.append({
            'Month': month_names[month],
            'Variable': var_name,
            'Mean': np.mean(flat_data),
            'Std Dev': np.std(flat_data),
            'Min': np.min(flat_data),
            'Max': np.max(flat_data),
            'Median': np.median(flat_data)
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # Save to CSV
    csv_path = 'monthly_statistics_2024.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"\nStatistics saved to: {csv_path}")
    
    return df_summary


def main():
    """Main execution function"""
    
    # Data path - update this as needed
    data_path = r"C:\Users\abhik\Downloads\SEALEVEL_GLO_PHY_L4_MY_008_047 cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D 2024 01"
    
    print("="*70)
    print("CMEMS Sea Level Data Visualization Tool")
    print("="*70)
    
    # Step 1: Explore data structure
    explore_data_structure(data_path)
    
    # Step 2: Load and process data
    monthly_data = load_and_process_data(data_path, year=2024)
    
    if monthly_data:
        # Step 3: Create visualizations
        visualize_monthly_data(monthly_data, year=2024)
        
        # Step 4: Generate statistical summary
        create_statistical_summary(monthly_data, year=2024)
    else:
        print("\nFailed to load data. Please check the data path and format.")


if __name__ == "__main__":
    main()
