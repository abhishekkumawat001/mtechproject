# Complete Guide to Opening and Viewing NetCDF (.nc) Files

## What are NetCDF Files?

NetCDF (Network Common Data Form) files are a file format designed for storing scientific data, especially multidimensional arrays. SWOT satellite data is commonly distributed in this format.

## Quick Start - 3 Ways to View .nc Files

### Method 1: Using Python (Recommended)

#### Required Libraries
```bash
pip install xarray netcdf4 matplotlib numpy pandas
```

#### Basic Code Example
```python
import xarray as xr

# Open and view the file
ds = xr.open_dataset('your_file.nc')

# Show basic info
print(ds)

# List all variables
print("Variables:", list(ds.data_vars.keys()))

# View specific variable
print(ds['variable_name'])

# Close file
ds.close()
```

### Method 2: Using Our Custom Tools

We've created two Python scripts for you:

1. **`simple_nc_examples.py`** - Basic examples and quick start
2. **`nc_file_viewer_guide.py`** - Advanced interactive viewer

#### To use them:

```bash
# Run basic examples
python simple_nc_examples.py

# Use the advanced viewer
python nc_file_viewer_guide.py
```

### Method 3: Online Viewers

- **NASA Panoply**: Free tool for viewing NetCDF files
- **Ncview**: Simple NetCDF viewer
- **Online NetCDF viewers**: Various web-based tools

## Step-by-Step Instructions

### 1. Install Required Software

#### Option A: Using Anaconda (Recommended)
```bash
# Install Anaconda from https://www.anaconda.com/
# Then install additional packages:
conda install -c conda-forge xarray netcdf4 matplotlib cartopy
```

#### Option B: Using pip
```bash
pip install xarray netcdf4 matplotlib numpy pandas seaborn
```

### 2. Basic File Inspection

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Open file
file_path = 'your_swot_file.nc'
ds = xr.open_dataset(file_path)

# Basic information
print("File dimensions:", dict(ds.dims))
print("Variables:", list(ds.data_vars.keys()))
print("Coordinates:", list(ds.coords.keys()))

# Variable details
for var in ds.data_vars:
    print(f"{var}: {ds[var].shape} - {ds[var].dtype}")
```

### 3. SWOT-Specific Analysis

For SWOT files, look for these common variables:
- `ssh_karin` - Sea Surface Height from KaRIn
- `longitude_nadir` - Longitude coordinates
- `latitude_nadir` - Latitude coordinates
- `time` - Time information

```python
# SWOT SSH analysis example
if 'ssh_karin' in ds.variables:
    ssh = ds['ssh_karin']
    print(f"SSH shape: {ssh.shape}")
    print(f"SSH range: {ssh.min().values:.3f} to {ssh.max().values:.3f} meters")
    
    # Simple plot
    plt.figure(figsize=(10, 6))
    ssh.plot()
    plt.title('SWOT Sea Surface Height')
    plt.show()
```

### 4. Interactive Exploration

Use our interactive tool:

```python
from nc_file_viewer_guide import NetCDFViewer

# Create viewer instance
viewer = NetCDFViewer('your_file.nc')

# Load file
viewer.load_file()

# Start interactive exploration
viewer.interactive_explorer()
```

Available commands in interactive mode:
- `info` - Basic file information
- `structure` - Complete file structure
- `variables` - Detailed variable information
- `view <variable>` - Quick view of data
- `plot <variable>` - Create plots
- `export <variable>` - Export to CSV
- `list` - List all variables
- `quit` - Exit

## Advanced Usage Examples

### 1. Data Extraction and Analysis

```python
import xarray as xr
import pandas as pd

# Open file
ds = xr.open_dataset('swot_file.nc')

# Extract specific variable
ssh_data = ds['ssh_karin']

# Convert to pandas DataFrame
df = ssh_data.to_dataframe()

# Basic statistics
print(df.describe())

# Save to CSV
df.to_csv('extracted_ssh_data.csv')
```

### 2. Creating Visualizations

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# For 2D geographical data
if 'longitude_nadir' in ds and 'latitude_nadir' in ds:
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot SSH data
    scatter = ax.scatter(ds['longitude_nadir'], 
                        ds['latitude_nadir'],
                        c=ds['ssh_karin'], 
                        cmap='viridis',
                        transform=ccrs.PlateCarree())
    
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(scatter, label='SSH (m)')
    plt.title('SWOT Sea Surface Height')
    plt.show()
```

### 3. Time Series Analysis

```python
# If your data has time dimension
if 'time' in ds.dims:
    # Group by time and compute statistics
    daily_mean = ds.groupby('time.day').mean()
    
    # Plot time series
    if 'ssh_karin' in ds:
        ssh_ts = ds['ssh_karin'].mean(dim=['longitude', 'latitude'])
        ssh_ts.plot()
        plt.title('SSH Time Series')
        plt.show()
```

## Troubleshooting Common Issues

### Problem 1: "No module named 'xarray'"
**Solution**: Install required packages
```bash
pip install xarray netcdf4
```

### Problem 2: "File not found" error
**Solution**: Check file path and ensure the file exists
```python
from pathlib import Path
file_path = Path('your_file.nc')
if file_path.exists():
    print("File found!")
else:
    print("File not found. Check the path.")
```

### Problem 3: Memory issues with large files
**Solution**: Use chunking or load specific variables
```python
# Load with chunks
ds = xr.open_dataset('large_file.nc', chunks={'time': 100})

# Load only specific variables
ds = xr.open_dataset('large_file.nc', data_vars=['ssh_karin'])
```

### Problem 4: Plotting issues
**Solution**: Check data dimensions and handle NaN values
```python
# Remove NaN values before plotting
data_clean = data.dropna(dim='longitude')

# For 2D data, use appropriate plot method
if len(data.dims) == 2:
    data.plot.imshow()
elif len(data.dims) == 1:
    data.plot.line()
```

## File Structure Examples

### Typical SWOT File Structure:
```
Dimensions:
  time: 1234
  longitude: 2048
  latitude: 1024

Variables:
  ssh_karin (time, longitude, latitude): Sea Surface Height
  longitude_nadir (time): Longitude coordinates
  latitude_nadir (time): Latitude coordinates
  time (time): Time stamps

Attributes:
  title: "SWOT L2 SSH Product"
  institution: "NASA/JPL"
  ...
```

## Command Line Tools

If you have NetCDF command-line tools installed:

```bash
# Show file structure
ncdump -h filename.nc

# Show coordinate variables
ncdump -c filename.nc

# Extract specific variable
ncdump -v ssh_karin filename.nc

# Convert to text format
ncdump filename.nc > output.txt
```

## Best Practices

1. **Always close files**: Use `with` statements or `.close()`
2. **Check data before plotting**: Look for NaN values and data ranges
3. **Use appropriate visualizations**: Line plots for 1D, heatmaps for 2D
4. **Handle large files carefully**: Use chunking for memory efficiency
5. **Understand your data**: Check units, descriptions, and coordinate systems

## Resources

- **xarray documentation**: http://xarray.pydata.org/
- **NetCDF official site**: https://www.unidata.ucar.edu/software/netcdf/
- **SWOT mission info**: https://swot.jpl.nasa.gov/
- **NASA Panoply**: https://www.giss.nasa.gov/tools/panoply/

## Quick Reference Commands

```python
# Essential imports
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Open file
ds = xr.open_dataset('file.nc')

# Basic info
print(ds.info())

# List variables
print(list(ds.data_vars.keys()))

# Quick plot
ds['variable_name'].plot()
plt.show()

# Export data
ds['variable_name'].to_dataframe().to_csv('output.csv')
```

This guide should help you get started with viewing and analyzing NetCDF files. Start with the simple examples and gradually move to more advanced analysis as needed!
