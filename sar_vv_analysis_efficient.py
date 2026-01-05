#!/usr/bin/env python3
"""
Memory-Efficient SAR VV Polarization Analysis
=============================================

This script provides memory-optimized analysis of Sentinel-1 SAR VV polarization data.
It handles large datasets by using intelligent downsampling and memory management.

Author: Automated Analysis System
Date: August 25, 2025
Target: Sentinel-1 SAR data in sar_extracted folder
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import rasterio
from pathlib import Path
import pandas as pd
import warnings
from datetime import datetime
import gc  # Garbage collection

warnings.filterwarnings('ignore')

class MemoryEfficientSARAnalyzer:
    """Memory-efficient SAR VV polarization analyzer."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("sar_analysis_output/vv_polarization_efficient")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory management settings
        self.max_plot_size = 2000  # Maximum dimension for plotting
        self.max_histogram_samples = 500000  # Maximum samples for histogram
        
        print("Memory-Efficient SAR VV Analyzer initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def find_sar_datasets(self):
        """Find all SAR datasets and their VV files."""
        datasets = []
        
        for safe_dir in self.data_dir.glob("S1*.SAFE"):
            if safe_dir.is_dir():
                # Look for VV polarization files
                measurement_dir = safe_dir / "measurement"
                if measurement_dir.exists():
                    vv_files = list(measurement_dir.glob("*vv*.tiff"))
                    if vv_files:
                        datasets.append({
                            'safe_dir': safe_dir,
                            'vv_file': vv_files[0],  # Take first VV file
                            'name': safe_dir.name
                        })
        
        print(f"Found {len(datasets)} SAR datasets with VV polarization")
        return datasets
    
    def analyze_single_dataset(self, dataset_info, dataset_id):
        """Analyze a single SAR dataset with memory optimization."""
        print(f"\nAnalyzing {dataset_info['name']}...")
        
        vv_file = dataset_info['vv_file']
        
        # Read metadata first
        with rasterio.open(vv_file) as src:
            meta = src.meta
            shape = (src.height, src.width)
            print(f"  Shape: {shape}")
            print(f"  Data type: {meta['dtype']}")
            
            # Calculate downsampling for visualization if needed
            if max(shape) > self.max_plot_size:
                factor = max(shape) / self.max_plot_size
                plot_shape = (int(shape[0] / factor), int(shape[1] / factor))
                print(f"  Downsampling to {plot_shape} for visualization")
            else:
                plot_shape = shape
                factor = 1
            
            # Read downsampled data for plotting
            if factor > 1:
                # Read with decimation
                decimation = int(factor)
                plot_data = src.read(1, 
                                   out_shape=plot_shape,
                                   resampling=rasterio.enums.Resampling.average)
            else:
                plot_data = src.read(1)
            
            # Convert to dB (handle zeros)
            plot_data_db = np.where(plot_data > 0, 
                                  10 * np.log10(plot_data.astype(np.float64)), 
                                  -100)
            
            # Read statistical sample for histogram (memory efficient)
            sample_size = min(self.max_histogram_samples, shape[0] * shape[1])
            
            if sample_size < shape[0] * shape[1]:
                # Random sampling
                total_pixels = shape[0] * shape[1]
                sample_indices = np.random.choice(total_pixels, sample_size, replace=False)
                
                # Read samples
                hist_samples = []
                block_size = 1000000  # Read in blocks
                
                for i in range(0, len(sample_indices), block_size):
                    batch_indices = sample_indices[i:i+block_size]
                    # Convert 1D indices to 2D
                    rows = batch_indices // shape[1]
                    cols = batch_indices % shape[1]
                    
                    # Read specific pixels (simplified - read small windows)
                    for r, c in zip(rows, cols):
                        try:
                            window = rasterio.windows.Window(c, r, 1, 1)
                            pixel_val = src.read(1, window=window)[0, 0]
                            if pixel_val > 0:
                                hist_samples.append(10 * np.log10(pixel_val))
                            else:
                                hist_samples.append(-100)
                        except:
                            continue
                
                hist_data = np.array(hist_samples)
            else:
                # Small enough to read all
                full_data = src.read(1)
                hist_data = np.where(full_data > 0,
                                   10 * np.log10(full_data.astype(np.float64)),
                                   -100).flatten()
        
        # Calculate statistics
        valid_hist = hist_data[hist_data > -100]
        if len(valid_hist) > 0:
            stats = {
                'mean': np.mean(valid_hist),
                'std': np.std(valid_hist),
                'min': np.min(valid_hist),
                'max': np.max(valid_hist),
                'median': np.median(valid_hist),
                'p25': np.percentile(valid_hist, 25),
                'p75': np.percentile(valid_hist, 75),
                'p95': np.percentile(valid_hist, 95),
                'coverage': len(valid_hist) / len(hist_data) * 100
            }
        else:
            stats = {k: 0 for k in ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75', 'p95', 'coverage']}
        
        print(f"  Statistics: Mean={stats['mean']:.2f}dB, Std={stats['std']:.2f}dB")
        
        # Create visualization
        self.create_dataset_plot(plot_data_db, hist_data, stats, dataset_info, dataset_id)
        
        # Clean up memory
        del plot_data, plot_data_db, hist_data, valid_hist
        gc.collect()
        
        return stats
    
    def create_dataset_plot(self, plot_data_db, hist_data, stats, dataset_info, dataset_id):
        """Create visualization for a single dataset."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'VV Polarization Analysis - Dataset {dataset_id}\n{dataset_info["name"]}', fontsize=14)
        
        # 1. VV intensity image
        ax1 = axes[0, 0]
        im1 = ax1.imshow(plot_data_db, cmap='gray', vmin=-30, vmax=5)
        ax1.set_title('VV Intensity (dB)')
        ax1.set_xlabel('Range')
        ax1.set_ylabel('Azimuth')
        plt.colorbar(im1, ax=ax1, label='VV (dB)')
        
        # 2. Histogram
        ax2 = axes[0, 1]
        valid_hist = hist_data[hist_data > -100]
        if len(valid_hist) > 0:
            ax2.hist(valid_hist, bins=100, alpha=0.7, density=True, color='blue')
        ax2.set_xlabel('VV (dB)')
        ax2.set_ylabel('Density')
        ax2.set_title('VV Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Enhanced contrast
        ax3 = axes[1, 0]
        if len(valid_hist) > 0:
            p2, p98 = np.percentile(valid_hist, [2, 98])
            im3 = ax3.imshow(plot_data_db, cmap='viridis', vmin=p2, vmax=p98)
            plt.colorbar(im3, ax=ax3, label='VV (dB)')
        else:
            im3 = ax3.imshow(plot_data_db, cmap='viridis')
        ax3.set_title('Enhanced Contrast')
        ax3.set_xlabel('Range')
        ax3.set_ylabel('Azimuth')
        
        # 4. Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
VV BACKSCATTER STATISTICS

Data Coverage: {stats['coverage']:.1f}%
Shape: {plot_data_db.shape}

INTENSITY (dB):
Mean: {stats['mean']:.2f}
Std: {stats['std']:.2f}
Min: {stats['min']:.2f}
Max: {stats['max']:.2f}
Median: {stats['median']:.2f}

PERCENTILES:
25th: {stats['p25']:.2f}
75th: {stats['p75']:.2f}
95th: {stats['p95']:.2f}
        """
        
        ax4.text(0.1, 0.9, stats_text.strip(), transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save with moderate DPI
        output_file = self.output_dir / f"vv_analysis_dataset_{dataset_id}.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")
    
    def create_summary_comparison(self, all_stats):
        """Create summary comparison of all datasets."""
        print("\nCreating summary comparison...")
        
        # Create comparison DataFrame
        df = pd.DataFrame(all_stats).T
        df.index.name = 'Dataset'
        
        # Save statistics
        csv_file = self.output_dir / "vv_comparison_summary.csv"
        df.to_csv(csv_file)
        print(f"Statistics saved: {csv_file}")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('VV Polarization - Multi-Dataset Comparison', fontsize=16)
        
        datasets = df.index
        
        # 1. Mean comparison
        ax1 = axes[0, 0]
        ax1.bar(datasets, df['mean'], alpha=0.7, color='blue')
        ax1.set_ylabel('Mean VV (dB)')
        ax1.set_title('Mean VV Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Standard deviation comparison
        ax2 = axes[0, 1]
        ax2.bar(datasets, df['std'], alpha=0.7, color='red')
        ax2.set_ylabel('Std VV (dB)')
        ax2.set_title('VV Variability Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Range comparison
        ax3 = axes[1, 0]
        vv_range = df['max'] - df['min']
        ax3.bar(datasets, vv_range, alpha=0.7, color='green')
        ax3.set_ylabel('VV Range (dB)')
        ax3.set_title('VV Dynamic Range')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Data coverage
        ax4 = axes[1, 1]
        ax4.bar(datasets, df['coverage'], alpha=0.7, color='orange')
        ax4.set_ylabel('Data Coverage (%)')
        ax4.set_title('Valid Data Coverage')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comparison_file = self.output_dir / "vv_datasets_comparison.png"
        plt.savefig(comparison_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved: {comparison_file}")
        
        return df
    
    def run_analysis(self):
        """Run complete memory-efficient VV analysis."""
        print("="*60)
        print("MEMORY-EFFICIENT SAR VV POLARIZATION ANALYSIS")
        print("="*60)
        
        # Find datasets
        datasets = self.find_sar_datasets()
        if not datasets:
            print("No SAR datasets found!")
            return
        
        # Analyze each dataset
        all_stats = {}
        
        for i, dataset_info in enumerate(datasets, 1):
            try:
                stats = self.analyze_single_dataset(dataset_info, i)
                all_stats[f"Dataset_{i}"] = stats
            except Exception as e:
                print(f"Error analyzing {dataset_info['name']}: {e}")
                continue
        
        if not all_stats:
            print("No datasets were successfully analyzed!")
            return
        
        # Create comparison
        comparison_df = self.create_summary_comparison(all_stats)
        
        # Create final summary
        self.create_final_summary(comparison_df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Results saved in: {self.output_dir}")
        print("="*60)
    
    def create_final_summary(self, df):
        """Create final analysis summary."""
        summary = {
            'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Total Datasets': len(df),
            'Output Directory': str(self.output_dir),
            'Overall Statistics': {
                'Mean VV Range': f"{df['mean'].min():.2f} to {df['mean'].max():.2f} dB",
                'Average Std Dev': f"{df['std'].mean():.2f} dB",
                'Best Coverage': f"{df['coverage'].max():.1f}%",
                'Worst Coverage': f"{df['coverage'].min():.1f}%"
            }
        }
        
        # Save summary
        summary_text = "SENTINEL-1 VV POLARIZATION ANALYSIS SUMMARY\n"
        summary_text += "=" * 50 + "\n\n"
        
        for key, value in summary.items():
            if isinstance(value, dict):
                summary_text += f"{key}:\n"
                for k, v in value.items():
                    summary_text += f"  {k}: {v}\n"
            else:
                summary_text += f"{key}: {value}\n"
        
        summary_text += "\nDATASET DETAILS:\n"
        summary_text += "-" * 20 + "\n"
        summary_text += df.round(2).to_string()
        
        summary_file = self.output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        print(f"\nSummary report saved: {summary_file}")


def main():
    """Main function to run memory-efficient SAR analysis."""
    
    # Configuration
    DATA_DIR = r"C:/Users/abhik/Desktop/project related work/sar_extracted"
    
    print("Starting memory-efficient SAR VV analysis...")
    
    try:
        # Initialize analyzer
        analyzer = MemoryEfficientSARAnalyzer(DATA_DIR)
        
        # Run analysis
        analyzer.run_analysis()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
