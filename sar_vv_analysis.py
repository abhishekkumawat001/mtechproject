#!/usr/bin/env python3
"""
Comprehensive VV Polarization Analysis for Sentinel-1 SAR Data
============================================================

This script analyzes VV polarization data from all five Sentinel-1 datasets
for the Andaman and Nicobar region. It provides statistical analysis,
visualization, and comparative assessment.

Author: Automated Analysis System
Date: August 24, 2025
Data: Sentinel-1 SAR GRDH products from March 25, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from pathlib import Path
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import warnings
from datetime import datetime
import os
from matplotlib.colors import LogNorm
from scipy import ndimage, stats
import cv2

# Suppress warnings
warnings.filterwarnings('ignore')

class SARVVAnalyzer:
    """Comprehensive VV polarization analyzer for Sentinel-1 SAR data."""
    
    def __init__(self, sar_data_dir: str, output_dir: str = None):
        """
        Initialize the SAR VV analyzer.
        
        Parameters:
        -----------
        sar_data_dir : str
            Path to the directory containing SAR .SAFE folders
        output_dir : str
            Output directory for results
        """
        self.sar_data_dir = Path(sar_data_dir)
        
        if output_dir is None:
            self.output_dir = Path("sar_analysis_output") / "vv_polarization_analysis"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results storage
        self.datasets_info = []
        self.vv_data = {}
        self.analysis_results = {}
        
        print("="*60)
        print("SENTINEL-1 VV POLARIZATION ANALYSIS")
        print("="*60)
        print(f"SAR Data Directory: {self.sar_data_dir}")
        print(f"Output Directory: {self.output_dir}")
        print()
    
    def discover_sar_datasets(self):
        """Discover all SAR datasets and their VV files."""
        print("Discovering SAR datasets...")
        
        safe_folders = list(self.sar_data_dir.glob("*.SAFE"))
        
        for safe_folder in safe_folders:
            measurement_dir = safe_folder / "measurement"
            
            if measurement_dir.exists():
                # Find VV polarization file
                vv_files = list(measurement_dir.glob("*-vv-*.tiff"))
                
                if vv_files:
                    vv_file = vv_files[0]  # Take the first VV file
                    
                    # Extract metadata from filename
                    filename_parts = safe_folder.name.split('_')
                    
                    dataset_info = {
                        'safe_folder': safe_folder.name,
                        'vv_file': vv_file,
                        'start_time': filename_parts[4],
                        'end_time': filename_parts[5],
                        'orbit': filename_parts[6],
                        'mission_id': filename_parts[7],
                        'product_id': filename_parts[8]
                    }
                    
                    self.datasets_info.append(dataset_info)
                    print(f"  Found: {safe_folder.name}")
                    print(f"    VV file: {vv_file.name}")
        
        print(f"\nTotal datasets found: {len(self.datasets_info)}")
        return self.datasets_info
    
    def load_vv_data(self):
        """Load VV polarization data from all datasets."""
        print("\nLoading VV polarization data...")
        
        for i, dataset_info in enumerate(self.datasets_info):
            print(f"Loading dataset {i+1}/{len(self.datasets_info)}: {dataset_info['safe_folder']}")
            
            try:
                # Load VV data
                with rasterio.open(dataset_info['vv_file']) as src:
                    vv_array = src.read(1)  # Read first band
                    transform = src.transform
                    crs = src.crs
                    nodata = src.nodata
                    
                    # Get spatial information
                    bounds = src.bounds
                    
                    # Store data
                    dataset_key = f"dataset_{i+1}"
                    self.vv_data[dataset_key] = {
                        'data': vv_array,
                        'transform': transform,
                        'crs': crs,
                        'bounds': bounds,
                        'nodata': nodata,
                        'info': dataset_info,
                        'shape': vv_array.shape
                    }
                    
                    print(f"  Shape: {vv_array.shape}")
                    print(f"  Data type: {vv_array.dtype}")
                    print(f"  Value range: {vv_array.min():.2f} to {vv_array.max():.2f}")
                    
            except Exception as e:
                print(f"  Error loading {dataset_info['safe_folder']}: {e}")
        
        print(f"Successfully loaded {len(self.vv_data)} datasets")
    
    def perform_statistical_analysis(self):
        """Perform comprehensive statistical analysis on VV data."""
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS")
        print("="*50)
        
        # Initialize results storage
        stats_summary = []
        
        for dataset_key, data_info in self.vv_data.items():
            vv_array = data_info['data']
            
            # Mask nodata values
            if data_info['nodata'] is not None:
                valid_data = vv_array[vv_array != data_info['nodata']]
            else:
                valid_data = vv_array[~np.isnan(vv_array)]
            
            if len(valid_data) == 0:
                print(f"No valid data in {dataset_key}")
                continue
            
            # Convert to dB if data is in linear scale (typical for Sentinel-1)
            # Check if data needs dB conversion (linear values are typically > 1)
            if valid_data.max() > 1:
                vv_db = 10 * np.log10(np.maximum(valid_data, 1e-10))  # Avoid log(0)
                is_linear = True
            else:
                vv_db = valid_data  # Already in dB
                is_linear = False
            
            # Calculate comprehensive statistics
            stats = {
                'dataset': dataset_key,
                'safe_folder': data_info['info']['safe_folder'],
                'start_time': data_info['info']['start_time'],
                'total_pixels': vv_array.size,
                'valid_pixels': len(valid_data),
                'data_coverage_percent': (len(valid_data) / vv_array.size) * 100,
                'is_linear_scale': is_linear,
                'raw_min': float(valid_data.min()),
                'raw_max': float(valid_data.max()),
                'raw_mean': float(valid_data.mean()),
                'raw_std': float(valid_data.std()),
                'db_min': float(vv_db.min()),
                'db_max': float(vv_db.max()),
                'db_mean': float(vv_db.mean()),
                'db_std': float(vv_db.std()),
                'db_median': float(np.median(vv_db)),
                'db_p25': float(np.percentile(vv_db, 25)),
                'db_p75': float(np.percentile(vv_db, 75)),
                'db_p95': float(np.percentile(vv_db, 95)),
                'db_p99': float(np.percentile(vv_db, 99))
            }
            
            stats_summary.append(stats)
            
            # Store dB data for visualization
            data_info['vv_db'] = vv_db
            data_info['valid_mask'] = valid_data if data_info['nodata'] is not None else ~np.isnan(vv_array)
            
            # Print statistics
            print(f"\n{dataset_key.upper()} ({data_info['info']['safe_folder']}):")
            print(f"  Data coverage: {stats['data_coverage_percent']:.1f}%")
            print(f"  Scale: {'Linear' if is_linear else 'dB'}")
            print(f"  VV (dB): {stats['db_min']:.2f} to {stats['db_max']:.2f}")
            print(f"  Mean: {stats['db_mean']:.2f} dB")
            print(f"  Std: {stats['db_std']:.2f} dB")
        
        # Convert to DataFrame and save
        self.stats_df = pd.DataFrame(stats_summary)
        self.stats_df.to_csv(self.output_dir / "vv_statistical_summary.csv", index=False)
        
        print(f"\nStatistical summary saved: {self.output_dir / 'vv_statistical_summary.csv'}")
        
        return self.stats_df
    
    def create_individual_visualizations(self):
        """Create individual visualizations for each dataset."""
        print("\nCreating individual dataset visualizations...")
        
        for dataset_key, data_info in self.vv_data.items():
            print(f"  Creating plots for {dataset_key}...")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Sentinel-1 VV Polarization Analysis - {dataset_key.upper()}\n{data_info["info"]["safe_folder"]}', 
                        fontsize=14)
            
            vv_array = data_info['data']
            vv_db = data_info['vv_db']
            
            # 1. Raw SAR image
            ax1 = axes[0, 0]
            # Apply log normalization for better visualization
            im1 = ax1.imshow(vv_array, cmap='gray', norm=LogNorm(vmin=max(vv_array.min(), 1e-6), vmax=vv_array.max()))
            ax1.set_title('VV Polarization (Raw Scale)')
            ax1.set_xlabel('Range (pixels)')
            ax1.set_ylabel('Azimuth (pixels)')
            plt.colorbar(im1, ax=ax1, label='Backscatter')
            
            # 2. dB scale image (if converted)
            ax2 = axes[0, 1]
            if data_info.get('vv_db') is not None and len(vv_db.shape) > 0:
                # Create dB image
                vv_array_db = np.full_like(vv_array, np.nan, dtype=float)
                valid_mask = data_info['valid_mask']
                if data_info['nodata'] is not None:
                    valid_mask = vv_array != data_info['nodata']
                else:
                    valid_mask = ~np.isnan(vv_array)
                
                if valid_mask.sum() > 0:
                    vv_array_db[valid_mask] = 10 * np.log10(np.maximum(vv_array[valid_mask], 1e-10))
                    im2 = ax2.imshow(vv_array_db, cmap='viridis', vmin=np.nanpercentile(vv_array_db, 2), 
                                   vmax=np.nanpercentile(vv_array_db, 98))
                    plt.colorbar(im2, ax=ax2, label='Backscatter (dB)')
                else:
                    ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'dB conversion failed', ha='center', va='center', transform=ax2.transAxes)
            
            ax2.set_title('VV Polarization (dB Scale)')
            ax2.set_xlabel('Range (pixels)')
            ax2.set_ylabel('Azimuth (pixels)')
            
            # 3. Histogram
            ax3 = axes[1, 0]
            if len(vv_db) > 0:
                ax3.hist(vv_db, bins=100, alpha=0.7, density=True, color='blue', edgecolor='black')
                ax3.axvline(np.mean(vv_db), color='red', linestyle='--', label=f'Mean: {np.mean(vv_db):.2f} dB')
                ax3.axvline(np.median(vv_db), color='orange', linestyle='--', label=f'Median: {np.median(vv_db):.2f} dB')
                ax3.set_xlabel('VV Backscatter (dB)')
                ax3.set_ylabel('Density')
                ax3.set_title('VV Backscatter Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. Statistics summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Get stats for this dataset
            dataset_stats = self.stats_df[self.stats_df['dataset'] == dataset_key].iloc[0]
            
            stats_text = f"""
DATASET STATISTICS

Shape: {vv_array.shape[0]} × {vv_array.shape[1]} pixels
Valid Data: {dataset_stats['data_coverage_percent']:.1f}%
Time: {dataset_stats['start_time']}

VV BACKSCATTER (dB)
Min: {dataset_stats['db_min']:.2f}
Max: {dataset_stats['db_max']:.2f}
Mean: {dataset_stats['db_mean']:.2f}
Std: {dataset_stats['db_std']:.2f}
Median: {dataset_stats['db_median']:.2f}

PERCENTILES
25th: {dataset_stats['db_p25']:.2f}
75th: {dataset_stats['db_p75']:.2f}
95th: {dataset_stats['db_p95']:.2f}
99th: {dataset_stats['db_p99']:.2f}
            """
            
            ax4.text(0.05, 0.95, stats_text.strip(), transform=ax4.transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save individual plot
            output_file = self.output_dir / f"{dataset_key}_vv_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {output_file}")
    
    def create_comparative_analysis(self):
        """Create comparative analysis across all datasets."""
        print("\nCreating comparative analysis...")
        
        # 1. Multi-dataset comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Sentinel-1 VV Polarization - Multi-Dataset Comparative Analysis', fontsize=16)
        
        # Extract all dB data for comparison
        all_vv_db = []
        dataset_labels = []
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.vv_data)))
        
        for i, (dataset_key, data_info) in enumerate(self.vv_data.items()):
            vv_db = data_info['vv_db']
            all_vv_db.extend(vv_db)
            dataset_labels.extend([dataset_key] * len(vv_db))
        
        # 1. Combined histogram
        ax1 = axes[0, 0]
        for i, (dataset_key, data_info) in enumerate(self.vv_data.items()):
            vv_db = data_info['vv_db']
            ax1.hist(vv_db, bins=50, alpha=0.6, label=dataset_key, color=colors[i], density=True)
        
        ax1.set_xlabel('VV Backscatter (dB)')
        ax1.set_ylabel('Density')
        ax1.set_title('VV Backscatter Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        ax2 = axes[0, 1]
        vv_data_list = [data_info['vv_db'] for data_info in self.vv_data.values()]
        dataset_names = list(self.vv_data.keys())
        
        box_plot = ax2.boxplot(vv_data_list, labels=dataset_names, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('VV Backscatter (dB)')
        ax2.set_title('VV Backscatter Range Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Mean vs Std scatter plot
        ax3 = axes[0, 2]
        means = [self.stats_df[self.stats_df['dataset'] == key]['db_mean'].iloc[0] 
                for key in self.vv_data.keys()]
        stds = [self.stats_df[self.stats_df['dataset'] == key]['db_std'].iloc[0] 
               for key in self.vv_data.keys()]
        
        scatter = ax3.scatter(means, stds, c=range(len(means)), cmap='viridis', s=100, alpha=0.7)
        
        for i, dataset_key in enumerate(self.vv_data.keys()):
            ax3.annotate(dataset_key, (means[i], stds[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Mean VV Backscatter (dB)')
        ax3.set_ylabel('Standard Deviation (dB)')
        ax3.set_title('Mean vs Variability')
        ax3.grid(True, alpha=0.3)
        
        # 4. Time series of statistics
        ax4 = axes[1, 0]
        times = [info['start_time'] for info in self.datasets_info]
        times_clean = [t.replace('T', ' ') for t in times]
        
        ax4.plot(range(len(means)), means, 'o-', label='Mean', linewidth=2, markersize=8)
        ax4.plot(range(len(stds)), stds, 's-', label='Std Dev', linewidth=2, markersize=8)
        
        ax4.set_xticks(range(len(times_clean)))
        ax4.set_xticklabels([t[:13] for t in times_clean], rotation=45)
        ax4.set_ylabel('VV Backscatter (dB)')
        ax4.set_title('Temporal Variation of VV Statistics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Data coverage comparison
        ax5 = axes[1, 1]
        coverage = [self.stats_df[self.stats_df['dataset'] == key]['data_coverage_percent'].iloc[0] 
                   for key in self.vv_data.keys()]
        
        bars = ax5.bar(range(len(coverage)), coverage, color=colors, alpha=0.7)
        ax5.set_xticks(range(len(dataset_names)))
        ax5.set_xticklabels(dataset_names, rotation=45)
        ax5.set_ylabel('Data Coverage (%)')
        ax5.set_title('Valid Data Coverage by Dataset')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, coverage):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        # 6. Correlation matrix (if applicable)
        ax6 = axes[1, 2]
        
        # Create summary statistics table
        summary_stats = self.stats_df[['dataset', 'db_mean', 'db_std', 'db_median', 'data_coverage_percent']]
        
        # Display as table
        ax6.axis('tight')
        ax6.axis('off')
        
        table_data = []
        for _, row in summary_stats.iterrows():
            table_data.append([
                row['dataset'],
                f"{row['db_mean']:.2f}",
                f"{row['db_std']:.2f}",
                f"{row['db_median']:.2f}",
                f"{row['data_coverage_percent']:.1f}%"
            ])
        
        table = ax6.table(cellText=table_data,
                         colLabels=['Dataset', 'Mean (dB)', 'Std (dB)', 'Median (dB)', 'Coverage'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax6.set_title('Summary Statistics Table')
        
        plt.tight_layout()
        
        # Save comparative plot
        comparative_output = self.output_dir / "vv_comparative_analysis.png"
        plt.savefig(comparative_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparative analysis saved: {comparative_output}")
    
    def create_mosaic_visualization(self):
        """Create a mosaic view of all VV datasets."""
        print("\nCreating mosaic visualization...")
        
        n_datasets = len(self.vv_data)
        cols = 3
        rows = (n_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        fig.suptitle('Sentinel-1 VV Polarization - All Datasets Mosaic View', fontsize=16)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (dataset_key, data_info) in enumerate(self.vv_data.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            vv_array = data_info['data']
            
            # Apply log normalization for visualization
            im = ax.imshow(vv_array, cmap='gray', 
                          norm=LogNorm(vmin=max(vv_array.min(), 1e-6), vmax=vv_array.max()))
            
            ax.set_title(f'{dataset_key}\n{data_info["info"]["start_time"][:13]}')
            ax.set_xlabel('Range')
            ax.set_ylabel('Azimuth')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for i in range(n_datasets, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save mosaic plot
        mosaic_output = self.output_dir / "vv_mosaic_view.png"
        plt.savefig(mosaic_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mosaic visualization saved: {mosaic_output}")
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report."""
        print("\nGenerating analysis report...")
        
        report_content = f"""
# Sentinel-1 VV Polarization Analysis Report

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Location:** {self.sar_data_dir}
**Output Directory:** {self.output_dir}

## Dataset Overview

**Total Datasets Analyzed:** {len(self.vv_data)}
**Acquisition Date:** March 25, 2025
**Sensor:** Sentinel-1A
**Product Type:** GRDH (Ground Range Detected High Resolution)
**Polarization:** VV (Vertical transmit, Vertical receive)

## Datasets Processed

"""
        
        for i, info in enumerate(self.datasets_info, 1):
            report_content += f"**Dataset {i}:** {info['safe_folder']}\n"
            report_content += f"- Start Time: {info['start_time']}\n"
            report_content += f"- End Time: {info['end_time']}\n"
            report_content += f"- Orbit: {info['orbit']}\n\n"
        
        report_content += f"""
## Statistical Summary

"""
        
        # Add statistical summary
        if hasattr(self, 'stats_df'):
            overall_stats = {
                'mean_db_range': f"{self.stats_df['db_mean'].min():.2f} to {self.stats_df['db_mean'].max():.2f} dB",
                'std_db_range': f"{self.stats_df['db_std'].min():.2f} to {self.stats_df['db_std'].max():.2f} dB",
                'coverage_range': f"{self.stats_df['data_coverage_percent'].min():.1f}% to {self.stats_df['data_coverage_percent'].max():.1f}%"
            }
            
            report_content += f"""
**VV Backscatter Range (Mean):** {overall_stats['mean_db_range']}
**Standard Deviation Range:** {overall_stats['std_db_range']}
**Data Coverage Range:** {overall_stats['coverage_range']}

### Detailed Statistics by Dataset

| Dataset | Mean (dB) | Std (dB) | Min (dB) | Max (dB) | Coverage (%) |
|---------|-----------|----------|----------|----------|--------------|
"""
            
            for _, row in self.stats_df.iterrows():
                report_content += f"| {row['dataset']} | {row['db_mean']:.2f} | {row['db_std']:.2f} | {row['db_min']:.2f} | {row['db_max']:.2f} | {row['data_coverage_percent']:.1f}% |\n"
        
        report_content += f"""

## Analysis Results

### Key Findings

1. **Data Quality:** All {len(self.vv_data)} datasets were successfully processed
2. **Temporal Consistency:** Data acquired within a short time window (same day)
3. **Spatial Coverage:** High-resolution SAR data covering the target region

### Generated Outputs

1. **Individual Analysis Plots:** One detailed plot per dataset
2. **Comparative Analysis:** Multi-dataset comparison plots
3. **Mosaic View:** All datasets displayed together
4. **Statistical Summary:** Comprehensive statistics in CSV format

### Files Generated

- `vv_statistical_summary.csv` - Detailed statistics for all datasets
- `dataset_X_vv_analysis.png` - Individual analysis plots
- `vv_comparative_analysis.png` - Multi-dataset comparison
- `vv_mosaic_view.png` - All datasets mosaic view
- `vv_analysis_report.md` - This analysis report

## Methodology

**Data Processing:**
1. Automatic discovery of SAR datasets
2. VV polarization data extraction
3. Linear to dB scale conversion (if needed)
4. Statistical analysis and visualization
5. Multi-dataset comparison

**Visualization Techniques:**
- Log-normalized intensity display
- Statistical distribution analysis
- Multi-dataset comparative plots
- Mosaic view for spatial context

## Quality Assessment

The analysis provides comprehensive insights into:
- Backscatter intensity variations
- Data quality and coverage
- Temporal consistency
- Spatial patterns

**Recommendation:** The processed VV polarization data is suitable for:
- Ocean surface roughness analysis
- Wind speed estimation
- Ship detection applications
- Marine environmental monitoring

---

*Report generated automatically by SAR VV Polarization Analyzer*
"""
        
        # Save report
        report_file = self.output_dir / "vv_analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Analysis report saved: {report_file}")
    
    def run_complete_analysis(self):
        """Run the complete VV polarization analysis pipeline."""
        print("Starting complete VV polarization analysis...")
        
        try:
            # Discover datasets
            self.discover_sar_datasets()
            
            if not self.datasets_info:
                print("No SAR datasets found!")
                return
            
            # Load VV data
            self.load_vv_data()
            
            if not self.vv_data:
                print("No VV data loaded!")
                return
            
            # Perform statistical analysis
            self.perform_statistical_analysis()
            
            # Create visualizations
            self.create_individual_visualizations()
            self.create_comparative_analysis()
            self.create_mosaic_visualization()
            
            # Generate report
            self.generate_analysis_report()
            
            print("\n" + "="*60)
            print("VV POLARIZATION ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Results saved in: {self.output_dir}")
            print(f"Datasets analyzed: {len(self.vv_data)}")
            print(f"Total outputs generated: {len(list(self.output_dir.glob('*')))}")
            print("="*60)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run SAR VV polarization analysis."""
    
    # Configuration
    SAR_DATA_DIR = r"C:\Users\abhik\Desktop\project related work\sar_extracted"
    OUTPUT_DIR = r"C:\Users\abhik\Desktop\project related work\sar_analysis_output\vv_polarization"
    
    print("SENTINEL-1 SAR VV POLARIZATION ANALYSIS")
    print("=" * 50)
    print(f"SAR Data Directory: {SAR_DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()
    
    # Check if SAR data directory exists
    if not Path(SAR_DATA_DIR).exists():
        print(f"Error: SAR data directory not found: {SAR_DATA_DIR}")
        return
    
    try:
        # Initialize analyzer
        analyzer = SARVVAnalyzer(SAR_DATA_DIR, OUTPUT_DIR)
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"Error initializing analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
