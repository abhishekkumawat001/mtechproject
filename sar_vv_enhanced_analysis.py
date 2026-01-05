#!/usr/bin/env python3
"""
Enhanced Sentinel-1 SAR VV Polarization Analysis with Multiple Scaling Methods
==============================================================================

This script provides comprehensive analysis of Sentinel-1 VV polarization data with:
- Linear scale visualization
- Logarithmic scale (dB) visualization  
- Enhanced contrast scaling
- Statistical analysis and comparison

Author: AI Assistant
Date: August 25, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import pandas as pd
import rasterio
from rasterio.plot import show
import seaborn as sns
import warnings
import gc
from datetime import datetime
from skimage import exposure
from scipy import ndimage

warnings.filterwarnings('ignore')
plt.style.use('default')

class SARVVAnalyzerEnhanced:
    """Enhanced SAR VV polarization analyzer with multiple scaling methods."""
    
    def __init__(self, sar_directory: str, output_directory: str = None):
        self.sar_dir = Path(sar_directory)
        self.output_dir = Path(output_directory) if output_directory else Path("sar_analysis_output/vv_polarization_enhanced")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {}
        self.statistics = {}
        
        print(f"Enhanced SAR VV Analyzer initialized")
        print(f"SAR Directory: {self.sar_dir}")
        print(f"Output Directory: {self.output_dir}")
    
    def discover_datasets(self):
        """Discover all Sentinel-1 datasets and their VV files."""
        print("\nDiscovering SAR datasets...")
        
        safe_dirs = [d for d in self.sar_dir.iterdir() if d.is_dir() and d.name.endswith('.SAFE')]
        
        for safe_dir in safe_dirs:
            # Look for VV TIFF files
            vv_files = list(safe_dir.glob('**/s1*-vv-*.tiff'))
            
            if vv_files:
                vv_file = vv_files[0]  # Take the first VV file found
                dataset_info = {
                    'name': safe_dir.name,
                    'path': safe_dir,
                    'vv_file': vv_file,
                    'date': self._extract_date(safe_dir.name)
                }
                
                self.datasets[safe_dir.name] = dataset_info
                print(f"  Found: {safe_dir.name}")
                print(f"    VV file: {vv_file.name}")
        
        print(f"\nTotal datasets found: {len(self.datasets)}")
        return len(self.datasets)
    
    def _extract_date(self, safe_name):
        """Extract acquisition date from SAFE directory name."""
        try:
            # Format: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_...
            parts = safe_name.split('_')
            date_part = parts[4][:8]  # YYYYMMDD
            return pd.to_datetime(date_part, format='%Y%m%d')
        except:
            return None
    
    def load_and_analyze_dataset(self, dataset_name, dataset_info):
        """Load and analyze a single dataset with multiple scaling methods."""
        print(f"  Processing: {dataset_name}")
        
        try:
            # Load VV data
            with rasterio.open(dataset_info['vv_file']) as src:
                vv_data = src.read(1).astype(np.float32)
                profile = src.profile
            
            # Mask zero/invalid values
            vv_data[vv_data == 0] = np.nan
            
            # Calculate statistics for different scales
            stats = self._calculate_multiscale_statistics(vv_data)
            
            # Create downsampled version for visualization
            downsample_factor = max(1, max(vv_data.shape) // 1500)
            if downsample_factor > 1:
                vv_display = vv_data[::downsample_factor, ::downsample_factor]
            else:
                vv_display = vv_data.copy()
            
            # Store processed data
            dataset_info.update({
                'vv_data': vv_display,
                'statistics': stats,
                'profile': profile,
                'downsample_factor': downsample_factor
            })
            
            print(f"    Shape: {vv_data.shape} -> {vv_display.shape} (downsampled)")
            print(f"    Linear range: {stats['linear_min']:.0f} - {stats['linear_max']:.0f}")
            print(f"    dB range: {stats['db_min']:.1f} - {stats['db_max']:.1f}")
            
            # Clean up large array
            del vv_data
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"    Error loading {dataset_name}: {e}")
            return False
    
    def _calculate_multiscale_statistics(self, vv_data):
        """Calculate statistics for multiple scaling methods."""
        valid_data = vv_data[~np.isnan(vv_data)]
        
        if len(valid_data) == 0:
            return {}
        
        # Linear scale statistics
        linear_stats = {
            'linear_min': float(valid_data.min()),
            'linear_max': float(valid_data.max()),
            'linear_mean': float(valid_data.mean()),
            'linear_std': float(valid_data.std()),
            'linear_median': float(np.median(valid_data))
        }
        
        # Logarithmic scale (dB) statistics
        # Convert to dB: dB = 10 * log10(linear)
        vv_db = 10 * np.log10(np.maximum(valid_data, 1e-10))  # Avoid log(0)
        
        db_stats = {
            'db_min': float(vv_db.min()),
            'db_max': float(vv_db.max()),
            'db_mean': float(vv_db.mean()),
            'db_std': float(vv_db.std()),
            'db_median': float(np.median(vv_db))
        }
        
        # Enhanced contrast statistics (histogram equalization)
        vv_normalized = (valid_data - valid_data.min()) / (valid_data.max() - valid_data.min())
        vv_equalized = exposure.equalize_hist(vv_normalized)
        
        enhanced_stats = {
            'enhanced_min': float(vv_equalized.min()),
            'enhanced_max': float(vv_equalized.max()),
            'enhanced_mean': float(vv_equalized.mean()),
            'enhanced_std': float(vv_equalized.std())
        }
        
        # Additional statistics
        percentiles = np.percentile(vv_db, [1, 5, 25, 75, 95, 99])
        percentile_stats = {
            'db_p01': percentiles[0],
            'db_p05': percentiles[1],
            'db_p25': percentiles[2], 
            'db_p75': percentiles[3],
            'db_p95': percentiles[4],
            'db_p99': percentiles[5]
        }
        
        # Combine all statistics
        all_stats = {**linear_stats, **db_stats, **enhanced_stats, **percentile_stats}
        all_stats['data_coverage'] = len(valid_data) / vv_data.size * 100
        
        return all_stats
    
    def create_multiscale_visualization(self, dataset_name, dataset_info, dataset_num):
        """Create comprehensive visualization with linear, log, and enhanced scales."""
        print(f"    Creating multiscale visualization for {dataset_name}")
        
        vv_data = dataset_info['vv_data']
        stats = dataset_info['statistics']
        
        # Create figure with 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Dataset {dataset_num}: VV Polarization Analysis - Multiple Scales\n{dataset_name}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Linear Scale (Original DN Values)
        ax1 = axes[0, 0]
        im1 = ax1.imshow(vv_data, cmap='gray', aspect='auto')
        ax1.set_title('Linear Scale (Digital Numbers)')
        ax1.set_xlabel('Range')
        ax1.set_ylabel('Azimuth')
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Digital Number')
        
        # 2. Logarithmic Scale (dB)
        ax2 = axes[0, 1]
        vv_db = 10 * np.log10(np.maximum(vv_data, 1e-10))
        # Mask extreme values for better visualization
        vv_db_clipped = np.clip(vv_db, stats['db_p05'], stats['db_p95'])
        im2 = ax2.imshow(vv_db_clipped, cmap='viridis', aspect='auto')
        ax2.set_title('Logarithmic Scale (dB)')
        ax2.set_xlabel('Range')
        ax2.set_ylabel('Azimuth')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('VV Backscatter (dB)')
        
        # 3. Enhanced Contrast (Histogram Equalization)
        ax3 = axes[0, 2]
        vv_normalized = (vv_data - np.nanmin(vv_data)) / (np.nanmax(vv_data) - np.nanmin(vv_data))
        vv_enhanced = exposure.equalize_hist(vv_normalized, mask=~np.isnan(vv_normalized))
        im3 = ax3.imshow(vv_enhanced, cmap='plasma', aspect='auto')
        ax3.set_title('Enhanced Contrast (Hist. Equalized)')
        ax3.set_xlabel('Range')
        ax3.set_ylabel('Azimuth')
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.set_label('Enhanced Values')
        
        # 4. Adaptive Histogram Equalization (CLAHE)
        ax4 = axes[1, 0]
        vv_adaptive = exposure.equalize_adapthist(vv_normalized, clip_limit=0.03)
        im4 = ax4.imshow(vv_adaptive, cmap='inferno', aspect='auto')
        ax4.set_title('Adaptive Enhanced (CLAHE)')
        ax4.set_xlabel('Range')
        ax4.set_ylabel('Azimuth')
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
        cbar4.set_label('CLAHE Values')
        
        # 5. Histogram Comparison
        ax5 = axes[1, 1]
        valid_linear = vv_data[~np.isnan(vv_data)]
        valid_db = vv_db[~np.isnan(vv_db)]
        
        if len(valid_db) > 0:
            ax5.hist(valid_db, bins=50, alpha=0.7, density=True, color='blue', label='dB scale')
            ax5.axvline(stats['db_mean'], color='red', linestyle='--', label=f'Mean: {stats["db_mean"]:.1f} dB')
            ax5.axvline(stats['db_median'], color='orange', linestyle='--', label=f'Median: {stats["db_median"]:.1f} dB')
        
        ax5.set_xlabel('VV Backscatter (dB)')
        ax5.set_ylabel('Density')
        ax5.set_title('VV Distribution (dB)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""
DATASET STATISTICS

Shape: {vv_data.shape}
Coverage: {stats['data_coverage']:.1f}%

LINEAR SCALE:
Range: {stats['linear_min']:.0f} - {stats['linear_max']:.0f}
Mean: {stats['linear_mean']:.0f}
Std: {stats['linear_std']:.0f}

LOGARITHMIC SCALE (dB):
Range: {stats['db_min']:.1f} - {stats['db_max']:.1f}
Mean: {stats['db_mean']:.1f}
Std: {stats['db_std']:.1f}
Median: {stats['db_median']:.1f}

PERCENTILES (dB):
5th: {stats['db_p05']:.1f}
25th: {stats['db_p25']:.1f}
75th: {stats['db_p75']:.1f}
95th: {stats['db_p95']:.1f}

ENHANCED CONTRAST:
Mean: {stats['enhanced_mean']:.3f}
Std: {stats['enhanced_std']:.3f}
"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save plot
        safe_name = dataset_name.replace('.SAFE', '').replace('S1A_IW_GRDH_1SDV_', '')
        output_file = self.output_dir / f"dataset_{dataset_num:02d}_multiscale_{safe_name}.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"      Saved: {output_file.name}")
        
        # Clean up memory
        del vv_data, vv_db, vv_enhanced, vv_adaptive
        gc.collect()
    
    def _create_short_name(self, dataset_name):
        """Create a shorter, more readable dataset name."""
        # Remove common prefixes and suffixes
        short_name = dataset_name.replace('.SAFE', '')
        short_name = short_name.replace('S1A_IW_GRDH_1SDV_', 'S1A_')
        short_name = short_name.replace('S1B_IW_GRDH_1SDV_', 'S1B_')
        
        # Extract date and orbit information
        if '_' in short_name:
            parts = short_name.split('_')
            if len(parts) >= 5:
                # Try to extract date (YYYYMMDDTHHMMSS format)
                for part in parts:
                    if len(part) >= 8 and part[:8].isdigit():
                        date_part = part[:8]  # YYYYMMDD
                        formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                        
                        # Try to find orbit number (usually starts with numbers)
                        orbit_part = ""
                        for p in parts:
                            if p.isdigit() and len(p) <= 6:
                                orbit_part = f"_{p}"
                                break
                        
                        return f"S1_{formatted_date}{orbit_part}"
        
        # Fallback: use first 20 characters if no pattern found
        return short_name[:20] if len(short_name) > 20 else short_name

    def create_comparison_plots(self):
        """Create comparison plots across all datasets."""
        print("\n  Creating comparison plots...")
        
        # Prepare data for comparison
        comparison_data = []
        for dataset_name, dataset_info in self.datasets.items():
            if 'statistics' in dataset_info:
                stats = dataset_info['statistics']
                # Create shorter dataset name
                short_name = self._create_short_name(dataset_name)
                stats['dataset'] = short_name
                stats['full_name'] = dataset_name
                comparison_data.append(stats)
        
        if not comparison_data:
            print("No data available for comparison!")
            return
        
        df = pd.DataFrame(comparison_data)
        df.set_index('dataset', inplace=True)
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('SAR VV Polarization - Multi-Dataset Comparison', fontsize=16, fontweight='bold')
        
        datasets = df.index
        
        # Row 1: Linear scale comparisons
        ax1 = axes[0, 0]
        bars1 = ax1.bar(datasets, df['linear_mean'], alpha=0.7, color='blue')
        ax1.set_ylabel('Mean (Linear)')
        ax1.set_title('Linear Scale Mean')
        ax1.tick_params(axis='x', rotation=0)  # No rotation for short names
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        bars2 = ax2.bar(datasets, df['linear_std'], alpha=0.7, color='cyan')
        ax2.set_ylabel('Std (Linear)')
        ax2.set_title('Linear Scale Std Dev')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[0, 2]
        linear_range = df['linear_max'] - df['linear_min']
        bars3 = ax3.bar(datasets, linear_range, alpha=0.7, color='lightblue')
        ax3.set_ylabel('Range (Linear)')
        ax3.set_title('Linear Scale Range')
        ax3.tick_params(axis='x', rotation=0)
        ax3.grid(True, alpha=0.3)
        
        # Row 2: Logarithmic scale (dB) comparisons
        ax4 = axes[1, 0]
        bars4 = ax4.bar(datasets, df['db_mean'], alpha=0.7, color='green')
        ax4.set_ylabel('Mean (dB)')
        ax4.set_title('dB Scale Mean')
        ax4.tick_params(axis='x', rotation=0)
        ax4.grid(True, alpha=0.3)
        
        ax5 = axes[1, 1]
        bars5 = ax5.bar(datasets, df['db_std'], alpha=0.7, color='lime')
        ax5.set_ylabel('Std (dB)')
        ax5.set_title('dB Scale Std Dev')
        ax5.tick_params(axis='x', rotation=0)
        ax5.grid(True, alpha=0.3)
        
        ax6 = axes[1, 2]
        db_range = df['db_max'] - df['db_min']
        bars6 = ax6.bar(datasets, db_range, alpha=0.7, color='darkgreen')
        ax6.set_ylabel('Range (dB)')
        ax6.set_title('dB Scale Range')
        ax6.tick_params(axis='x', rotation=0)
        ax6.grid(True, alpha=0.3)
        
        # Row 3: Enhanced contrast and coverage
        ax7 = axes[2, 0]
        bars7 = ax7.bar(datasets, df['enhanced_mean'], alpha=0.7, color='purple')
        ax7.set_ylabel('Enhanced Mean')
        ax7.set_title('Enhanced Contrast Mean')
        ax7.tick_params(axis='x', rotation=0)
        ax7.grid(True, alpha=0.3)
        
        ax8 = axes[2, 1]
        bars8 = ax8.bar(datasets, df['data_coverage'], alpha=0.7, color='orange')
        ax8.set_ylabel('Coverage (%)')
        ax8.set_title('Data Coverage')
        ax8.tick_params(axis='x', rotation=0)
        ax8.grid(True, alpha=0.3)
        
        # Percentiles comparison
        ax9 = axes[2, 2]
        width = 0.15
        x = np.arange(len(datasets))
        ax9.bar(x - 2*width, df['db_p05'], width, label='5th %ile', alpha=0.7)
        ax9.bar(x - width, df['db_p25'], width, label='25th %ile', alpha=0.7)
        ax9.bar(x, df['db_median'], width, label='Median', alpha=0.7)
        ax9.bar(x + width, df['db_p75'], width, label='75th %ile', alpha=0.7)
        ax9.bar(x + 2*width, df['db_p95'], width, label='95th %ile', alpha=0.7)
        ax9.set_ylabel('VV (dB)')
        ax9.set_title('dB Percentiles Comparison')
        ax9.set_xticks(x)
        ax9.set_xticklabels(datasets, rotation=0)
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        comparison_file = self.output_dir / "multiscale_comparison_analysis.png"
        plt.savefig(comparison_file, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    Comparison plot saved: {comparison_file.name}")
        
        # Save comparison statistics
        stats_file = self.output_dir / "multiscale_statistics_comparison.csv"
        df.to_csv(stats_file)
        print(f"    Statistics saved: {stats_file.name}")
        
        return df
    
    def run_complete_analysis(self):
        """Run complete VV polarization analysis with multiple scaling methods."""
        print("="*60)
        print("ENHANCED SAR VV POLARIZATION ANALYSIS")
        print("="*60)
        print("Features: Linear Scale | Log Scale | Enhanced Contrast")
        print("="*60)
        
        # Discover datasets
        num_datasets = self.discover_datasets()
        if num_datasets == 0:
            print("No SAR datasets found!")
            return
        
        # Process each dataset
        print(f"\nProcessing {num_datasets} datasets...")
        successful_datasets = 0
        
        for i, (dataset_name, dataset_info) in enumerate(self.datasets.items()):
            print(f"\nDataset {i+1}/{num_datasets}:")
            
            success = self.load_and_analyze_dataset(dataset_name, dataset_info)
            if success:
                self.create_multiscale_visualization(dataset_name, dataset_info, i+1)
                successful_datasets += 1
        
        # Create comparison plots
        if successful_datasets > 1:
            print(f"\nCreating comparison analysis...")
            self.create_comparison_plots()
        
        # Generate summary report
        self.generate_summary_report(successful_datasets)
        
        print("="*60)
        print("ENHANCED ANALYSIS COMPLETE!")
        print(f"Successfully processed: {successful_datasets}/{num_datasets} datasets")
        print(f"Results saved in: {self.output_dir}")
        print("="*60)
    
    def generate_summary_report(self, successful_datasets):
        """Generate comprehensive summary report."""
        print(f"\n  Generating summary report...")
        
        report_content = f"""
# Enhanced SAR VV Polarization Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** Multi-Scale VV Polarization Analysis
**Datasets Processed:** {successful_datasets}

## Scaling Methods Applied

### 1. Linear Scale
- **Description:** Original digital number values from SAR sensor
- **Use Case:** Raw data analysis, calibration verification
- **Visualization:** Grayscale intensity map

### 2. Logarithmic Scale (dB)
- **Description:** 10 * log10(linear) transformation
- **Use Case:** Standard SAR backscatter analysis, feature detection
- **Visualization:** Color-mapped dB values with percentile clipping

### 3. Enhanced Contrast
- **Description:** Histogram equalization for improved visual contrast
- **Use Case:** Feature enhancement, visual interpretation
- **Visualization:** Plasma colormap with enhanced dynamic range

### 4. Adaptive Enhanced (CLAHE)
- **Description:** Contrast Limited Adaptive Histogram Equalization
- **Use Case:** Local contrast enhancement, detailed feature analysis
- **Visualization:** Inferno colormap with adaptive enhancement

## Files Generated

### Individual Dataset Analysis
"""
        
        for i, (dataset_name, dataset_info) in enumerate(self.datasets.items()):
            if 'statistics' in dataset_info:
                safe_name = dataset_name.replace('.SAFE', '').replace('S1A_IW_GRDH_1SDV_', '')
                report_content += f"- `dataset_{i+1:02d}_multiscale_{safe_name}.png`\n"
        
        report_content += f"""
### Comparison Analysis
- `multiscale_comparison_analysis.png` - Cross-dataset comparison
- `multiscale_statistics_comparison.csv` - Numerical statistics

## Statistical Summary

The analysis provides comprehensive statistics for each scaling method:
- **Linear Scale:** Raw sensor values and their distribution
- **dB Scale:** Logarithmic backscatter coefficients (standard in SAR)
- **Enhanced Scale:** Contrast-enhanced values for visualization

## Recommendations

1. **Scientific Analysis:** Use dB scale values for quantitative analysis
2. **Visual Interpretation:** Use enhanced contrast for feature identification
3. **Calibration:** Use linear scale for sensor calibration and validation
4. **Comparison:** Use percentile statistics for cross-dataset comparison

---
*Analysis performed using Enhanced SAR VV Analyzer*
"""
        
        report_file = self.output_dir / "ENHANCED_ANALYSIS_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"    Report saved: {report_file.name}")


def main():
    """Main function to run enhanced SAR VV analysis."""
    # Configuration
    SAR_DIRECTORY = r"C:/Users/abhik/Desktop/project related work/sar_extracted"
    OUTPUT_DIRECTORY = r"C:/Users/abhik/Desktop/project related work/sar_analysis_output/vv_polarization_enhanced"
    
    print("ENHANCED SENTINEL-1 SAR VV POLARIZATION ANALYSIS")
    print("="*50)
    print("Features: Linear | Logarithmic | Enhanced Contrast")
    print("="*50)
    print(f"SAR Data Directory: {SAR_DIRECTORY}")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    print()
    
    try:
        # Initialize analyzer
        analyzer = SARVVAnalyzerEnhanced(SAR_DIRECTORY, OUTPUT_DIRECTORY)
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
