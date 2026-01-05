# Creating Internal Wave Visualizations from SWOT Data

## Overview
This guide explains how to create publication-quality internal wave visualizations from SWOT L2 SSH data, similar to the images you've seen in oceanographic research papers.

## What You Can Create with Current SWOT Data

### 1. **Swath Plots** (Most Common Internal Wave Visualization)
- **Description**: 2D color plots showing SSH variations across the satellite swath
- **Shows**: Internal wave crests and troughs as alternating color patterns
- **SWOT Variables Needed**: `ssha_karin_2` or `ssh_karin_2`
- **Processing**: High-pass filtering to remove large-scale ocean signals

### 2. **Along-Track Profiles**
- **Description**: Line plots showing wave patterns along satellite track
- **Shows**: Wave amplitude, wavelength, and propagation patterns
- **SWOT Variables Needed**: SSH data along center of swath
- **Processing**: Spectral analysis, autocorrelation for wavelength estimation

### 3. **Geographic Context Maps**
- **Description**: Maps showing where internal waves occur relative to bathymetry
- **Shows**: Spatial distribution and geographic context
- **SWOT Variables Needed**: `latitude`, `longitude`, SSH data
- **Additional Data**: Bathymetry for context

## Advanced Visualizations Requiring Additional Datasets

### 4. **Bathymetric Context Plots**
- **What You Need**: 
  - GEBCO bathymetry data: https://www.gebco.net/data_and_products/gridded_bathymetry_data/
  - ETOPO1 global relief: https://www.ngdc.noaa.gov/mgg/global/
- **Shows**: How internal waves relate to underwater topography
- **Key Features**: Seamounts, ridges, continental shelves that generate waves

### 5. **Multi-Pass Tracking**
- **What You Need**: Multiple SWOT passes over same region
- **Shows**: Wave propagation direction and speed
- **Processing**: Cross-correlation between passes, phase tracking

### 6. **Tidal Separation Plots**
- **What You Need**: 
  - FES2014 tidal model: https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html
  - TPXO tidal predictions: https://www.tpxo.net/
- **Shows**: Pure internal waves separated from tidal signals
- **Processing**: Remove predicted tidal SSH before internal wave analysis

### 7. **Theoretical Wave Analysis**
- **What You Need**:
  - Ocean stratification profiles (Argo floats): https://argo.ucsd.edu/
  - Buoyancy frequency calculations
  - Ocean reanalysis data (HYCOM, GLORYS)
- **Shows**: Comparison with theoretical internal wave modes
- **Processing**: Calculate theoretical phase speeds, mode shapes

## Scripts Provided

### 1. `swot_internal_wave_visualizer.py`
- **Purpose**: Comprehensive internal wave analysis suite
- **Features**: 
  - Multiple visualization types
  - Spectral analysis
  - Wave statistics along track
  - Configurable parameters
- **Use**: For detailed research-quality analysis

### 2. `quick_internal_wave_analysis.py`
- **Purpose**: Rapid visualization for immediate results
- **Features**:
  - Simple but effective plots
  - Minimal dependencies
  - Fast processing
- **Use**: For quick data exploration

## Running the Analysis

### Option 1: Quick Analysis (Recommended to Start)
```python
# Run this first to see immediate results
python quick_internal_wave_analysis.py
```

### Option 2: Comprehensive Analysis
```python
# Run this for full research-quality output
python swot_internal_wave_visualizer.py
```

## Key Processing Steps for Internal Wave Detection

### 1. **Quality Control**
```python
# Use quality flags to filter good data
ssh_clean = ssh.where(quality_flag == 0)
```

### 2. **High-Pass Filtering**
```python
# Remove large-scale signals (>50km wavelength)
# Keep internal wave signals (1-50km wavelength)
from scipy.ndimage import gaussian_filter1d

def high_pass_filter(data, cutoff_wavelength_km=50):
    # Convert wavelength to filter parameter
    sigma = cutoff_wavelength_km / (2.5 * along_track_resolution)
    smoothed = gaussian_filter1d(data, sigma=sigma)
    return data - smoothed  # High-pass filtered result
```

### 3. **Spectral Analysis**
```python
from scipy import signal

# Calculate power spectral density
frequencies, psd = signal.periodogram(ssh_data, fs=sampling_frequency)
wavelengths = 1 / frequencies[1:]  # Convert to wavelengths
```

## Typical Internal Wave Characteristics in SWOT Data

### **Wavelengths**: 1-50 km (internal waves vs. tidal signals >100km)
### **Amplitudes**: 0.01-0.5 m in SSH (can be larger in strong regions)
### **Locations**: 
- Continental shelves and slopes
- Near seamounts and ridges
- Straits and channels
- Areas with strong tidal currents

## Common Visualization Types You'll See in Papers

### Type A: **Swath Intensity Plots**
- Color-coded SSH across satellite swath
- Shows wave crests (red) and troughs (blue)
- Often includes bathymetry contours

### Type B: **Geographic Overview Maps**
- Satellite tracks overlaid on bathymetry
- Color shows internal wave activity
- Includes coastlines and depth contours

### Type C: **Time Series Analysis**
- Multiple passes showing wave evolution
- Phase tracking between passes
- Statistical analysis of wave properties

### Type D: **Spectral Energy Plots**
- Power spectral density vs. wavelength
- Comparison with theoretical spectra
- Energy distribution across wave modes

## Data Sources for Enhanced Analysis

### **Free Datasets:**
1. **GEBCO Bathymetry**: Global seafloor mapping
2. **Argo Float Data**: Ocean temperature/salinity profiles
3. **HYCOM Ocean Model**: 3D ocean state estimates
4. **ERA5 Reanalysis**: Wind and atmospheric forcing

### **Research Datasets:**
1. **FES2014 Tidal Model**: High-accuracy tidal predictions
2. **GLORYS Ocean Reanalysis**: High-resolution ocean state
3. **Regional Bathymetry**: Higher resolution depth data
4. **Ship-based CTD Data**: Local ocean structure

## Tips for Creating Publication-Quality Figures

### 1. **Color Schemes**
- Use `RdBu_r` (red-blue reversed) for SSH anomalies
- Use `viridis` or `plasma` for bathymetry
- Ensure colorblind-friendly palettes

### 2. **Geographic Context**
- Always include lat/lon coordinates
- Add bathymetry contours when possible
- Show coastlines for reference

### 3. **Scale Information**
- Include distance scales
- Show wavelength references
- Add amplitude scales in color bars

### 4. **Multi-Panel Layouts**
- Original SSH + filtered waves
- Geographic context + detailed swath
- Time series + spectral analysis

## Expected Results

After running the scripts, you should get visualizations showing:

1. **Clear wave patterns** in the high-pass filtered data
2. **Geographic distribution** of internal wave activity
3. **Wave statistics** (amplitude, wavelength estimates)
4. **Spectral characteristics** of the internal wave field

The quality and clarity of internal wave signals will depend on:
- **Ocean region** (some areas have stronger internal waves)
- **Tidal conditions** (spring vs. neap tides)
- **Local bathymetry** (wave generation sites)
- **Data quality** (weather conditions during satellite pass)

## Next Steps for Advanced Analysis

1. **Download bathymetry data** for your study region
2. **Collect multiple SWOT passes** for temporal analysis
3. **Add tidal model data** for signal separation
4. **Incorporate ocean model data** for theoretical comparisons
5. **Compare with ship-based measurements** for validation
