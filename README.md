# SWOT Data Analysis Tools

This folder contains several Python scripts for analyzing SWOT Level 2 SSH data with different levels of complexity and functionality.

## Files Overview

### 1. Configuration
- **`swot_config.py`** - Configuration file where you can modify analysis parameters

### 2. Analysis Scripts
- **`swot_simple_analysis.py`** - Basic analysis with good error handling (recommended for beginners)
- **`swot_enhanced_analysis.py`** - Advanced analysis with internal wave detection and comprehensive visualizations
- **`swot_data_analysis.py`** - Original analysis script
- **`SWOT_Data_Analysis.ipynb`** - Interactive Jupyter notebook for step-by-step analysis

### 3. Utility Scripts
- **`run_swot_analysis.py`** - Wrapper script to handle common issues when running analysis

## Quick Start

### Option 1: Simple Analysis (Recommended)
1. Edit `swot_config.py` to set your data directory path
2. Run: `python swot_simple_analysis.py`

### Option 2: Enhanced Analysis
1. Make sure you have all required packages installed:
   ```
   pip install xarray netcdf4 pandas numpy matplotlib seaborn scipy
   ```
2. Run: `python swot_enhanced_analysis.py`

### Option 3: Interactive Notebook
1. Open `SWOT_Data_Analysis.ipynb` in VS Code or Jupyter
2. Run cells step by step

## Configuration

Edit `swot_config.py` to customize:

```python
# Update this to your SWOT data folder
DATA_DIRECTORY = r"C:\path\to\your\SWOT\data"

# Analysis parameters
BOX_HALF_DEG = 0.5  # Analysis box size (degrees)
USE_SSHA_FIRST = True  # Prefer SSHA over SSH
KEEP_QUAL_EQ_ZERO_ONLY = True  # Apply quality control
```

## Expected Input Data

- SWOT Level 2 Low Rate SSH NetCDF files (.nc)
- Files should follow naming convention: `SWOT_L2_LR_SSH_Basic_*_YYYYMMDDTHHMMSS_*.nc`
- Data should contain SSH/SSHA variables and coordinate information

## Outputs

### Simple Analysis:
- Summary statistics CSV file
- 6-panel visualization plot showing:
  - Time series of mean SSH
  - SSH distribution histogram
  - Variability analysis
  - Data coverage assessment
  - Range analysis
  - Mean vs variability scatter plot

### Enhanced Analysis:
- Detailed measurements CSV
- Daily and monthly statistics CSV
- Internal wave analysis results
- 9-panel comprehensive visualization including:
  - All individual measurements
  - Geographic distribution
  - Power spectral density analysis
  - Internal wave detection plots

## Troubleshooting

### Common Issues:

1. **Unicode Encoding Errors**
   - Use `swot_simple_analysis.py` which handles encoding better
   - Or use `run_swot_analysis.py` wrapper

2. **Missing Libraries**
   ```
   pip install xarray netcdf4 pandas numpy matplotlib seaborn scipy
   ```

3. **No Files Found**
   - Check the `DATA_DIRECTORY` path in configuration
   - Ensure .nc files are present and named correctly

4. **No Data Extracted**
   - Files may not contain expected variable names
   - Try increasing `BOX_HALF_DEG` in configuration
   - Check if files contain SSH/SSHA data

### Variable Names Expected:
- SSH: `ssh_karin_2`, `ssh_karin`
- SSHA: `ssha_karin_2`, `ssha_karin`
- Coordinates: `latitude`, `longitude`, `lat`, `lon`
- Quality flags: `*_qual` (corresponding to data variables)

## Features

### Basic Features (All Scripts):
- Chronological file processing
- SSH/SSHA statistical analysis
- Quality control and data validation
- Export capabilities
- Basic visualizations

### Advanced Features (Enhanced Script):
- Internal wave analysis with high-pass filtering
- Power spectral density analysis
- Geographic distribution mapping
- Comprehensive quality assessment
- Advanced statistical aggregations
- Publication-quality visualizations

## Contact

For issues or questions about SWOT data analysis, refer to:
- SWOT Science Team documentation
- PO.DAAC SWOT tutorials
- NASA SWOT mission resources
