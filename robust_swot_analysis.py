#!/usr/bin/env python3
"""
Robust SWOT L2 SSH Analysis
===========================

A simplified but robust analysis script that properly handles SWOT L2 data structure.

Author: AI Assistant
Date: August 25, 2025
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scientific analysis libraries
from scipy import signal
from scipy.stats import zscore

class RobustSWOTAnalyzer:
    def __init__(self, data_dir, output_dir=None):
        """Initialize the robust SWOT analyzer."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'robust_swot_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.stats_dir = self.output_dir / 'statistics'
        self.spatial_dir = self.output_dir / 'spatial_analysis'
        self.internal_waves_dir = self.output_dir / 'internal_waves'
        
        for dir_path in [self.stats_dir, self.spatial_dir, self.internal_waves_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.all_data = []
        print(f"Robust SWOT Analyzer initialized")
        print(f"Output directory: {self.output_dir}")

    def load_and_process_data(self):
        """Load and process SWOT data with proper error handling."""
        print("\n=== LOADING SWOT DATA ===")
        
        nc_files = list(self.data_dir.glob("*.nc"))
        basic_files = [f for f in nc_files if 'Basic' in f.name]
        
        print(f"Found {len(basic_files)} Basic SSH files")
        
        successful_files = 0
        
        for i, file_path in enumerate(basic_files[:15]):  # Process first 15 files
            try:
                print(f"  Processing {i+1}/{min(15, len(basic_files))}: {file_path.name}")
                
                with xr.open_dataset(file_path) as ds:
                    # Get basic info
                    cycle = ds.attrs.get('cycle_number', i)
                    pass_number = ds.attrs.get('pass_number', 0)
                    
                    # Extract SSH data (use ssh_karin which is the main SSH variable)
                    ssh_karin = ds['ssh_karin'].values
                    lat = ds['latitude'].values
                    lon = ds['longitude'].values
                    time = ds['time'].values
                    
                    # Quality flag
                    quality_flag = ds['ssh_karin_qual'].values if 'ssh_karin_qual' in ds else None
                    
                    # Handle different dimensions properly
                    if ssh_karin.ndim > 1:
                        # Flatten multidimensional arrays
                        ssh_flat = ssh_karin.flatten()
                        lat_flat = lat.flatten() if lat.ndim > 1 else np.repeat(lat, len(ssh_flat))
                        lon_flat = lon.flatten() if lon.ndim > 1 else np.repeat(lon, len(ssh_flat))
                        time_flat = time.flatten() if time.ndim > 1 else np.repeat(time, len(ssh_flat))
                        quality_flat = quality_flag.flatten() if quality_flag is not None and quality_flag.ndim > 1 else None
                    else:
                        ssh_flat = ssh_karin
                        lat_flat = lat
                        lon_flat = lon
                        time_flat = time
                        quality_flat = quality_flag
                    
                    # Filter valid data
                    valid_mask = ~np.isnan(ssh_flat)
                    if quality_flat is not None:
                        valid_mask = valid_mask & (quality_flat == 0)
                    
                    if np.sum(valid_mask) > 0:
                        valid_ssh = ssh_flat[valid_mask]
                        valid_lat = lat_flat[valid_mask]
                        valid_lon = lon_flat[valid_mask]
                        valid_time = time_flat[valid_mask]
                        
                        # Store data
                        for j in range(len(valid_ssh)):
                            if j % 100 == 0:  # Subsample for memory efficiency
                                self.all_data.append({
                                    'file': file_path.name,
                                    'cycle': cycle,
                                    'pass': pass_number,
                                    'ssh': valid_ssh[j],
                                    'latitude': valid_lat[j],
                                    'longitude': valid_lon[j],
                                    'time': pd.to_datetime(valid_time[j]),
                                    'date': pd.to_datetime(valid_time[j]).date()
                                })
                        
                        successful_files += 1
                        print(f"    ✓ Extracted {np.sum(valid_mask)} valid points (subsampled: {len(valid_ssh)//100})")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
        
        print(f"\nData loading completed!")
        print(f"Successfully processed: {successful_files} files")
        print(f"Total data points: {len(self.all_data)}")

    def basic_statistics_analysis(self):
        """Perform comprehensive statistical analysis."""
        print("\n=== BASIC QUALITY & STATISTICS ANALYSIS ===")
        
        if not self.all_data:
            print("No data available for analysis!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_data)
        
        print(f"Analyzing {len(df)} data points...")
        
        # 1. Overall Statistics
        print("\n1. Overall SSH Statistics:")
        overall_stats = df['ssh'].describe()
        print(overall_stats)
        
        # 2. Daily Statistics
        daily_stats = df.groupby('date').agg({
            'ssh': ['count', 'mean', 'std', 'min', 'max']
        }).round(4)
        daily_stats.columns = ['count', 'mean', 'std', 'min', 'max']
        daily_stats['amplitude'] = daily_stats['max'] - daily_stats['min']
        
        # Save daily statistics
        daily_stats.to_csv(self.stats_dir / 'daily_statistics.csv')
        print(f"Daily statistics saved to: {self.stats_dir / 'daily_statistics.csv'}")
        
        # 3. Anomaly Detection
        df['ssh_zscore'] = zscore(df['ssh'])
        anomalies = df[np.abs(df['ssh_zscore']) > 3]
        
        print(f"\nAnomalies detected: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")
        
        # 4. Create visualization
        self._create_comprehensive_plots(df, daily_stats, anomalies)
        
        return df, daily_stats, anomalies

    def _create_comprehensive_plots(self, df, daily_stats, anomalies):
        """Create comprehensive analysis plots."""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('SWOT L2 SSH Comprehensive Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Time series
        ax1 = axes[0, 0]
        ax1.scatter(df['time'], df['ssh'], alpha=0.3, s=1, c='blue', label='SSH')
        ax1.scatter(anomalies['time'], anomalies['ssh'], c='red', s=5, alpha=0.8, label=f'Anomalies ({len(anomalies)})')
        ax1.set_ylabel('SSH (m)')
        ax1.set_title('SSH Time Series with Anomaly Detection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. SSH Distribution
        ax2 = axes[0, 1]
        ax2.hist(df['ssh'], bins=50, alpha=0.7, color='blue', density=True)
        ax2.axvline(df['ssh'].mean(), color='red', linestyle='--', label=f'Mean: {df["ssh"].mean():.2f}m')
        ax2.set_xlabel('SSH (m)')
        ax2.set_ylabel('Density')
        ax2.set_title('SSH Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Daily mean
        ax3 = axes[1, 0]
        daily_stats_reset = daily_stats.reset_index()
        ax3.plot(daily_stats_reset['date'], daily_stats_reset['mean'], 'b-', linewidth=2, label='Daily Mean')
        ax3.fill_between(daily_stats_reset['date'], 
                        daily_stats_reset['mean'] - daily_stats_reset['std'],
                        daily_stats_reset['mean'] + daily_stats_reset['std'],
                        alpha=0.3, color='blue', label='±1σ')
        ax3.set_ylabel('SSH (m)')
        ax3.set_title('Daily SSH Statistics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Daily amplitude
        ax4 = axes[1, 1]
        ax4.plot(daily_stats_reset['date'], daily_stats_reset['amplitude'], 'r-', linewidth=2)
        ax4.set_ylabel('Amplitude (m)')
        ax4.set_title('Daily SSH Amplitude (Max - Min)')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Spatial distribution
        ax5 = axes[2, 0]
        scatter = ax5.scatter(df['longitude'], df['latitude'], c=df['ssh'], 
                             cmap='RdYlBu_r', s=1, alpha=0.6)
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')
        ax5.set_title('SSH Spatial Distribution')
        plt.colorbar(scatter, ax=ax5, label='SSH (m)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Pass statistics
        ax6 = axes[2, 1]
        pass_stats = df.groupby('pass')['ssh'].agg(['count', 'mean', 'std'])
        pass_stats = pass_stats.sort_index()
        ax6.bar(pass_stats.index, pass_stats['count'], alpha=0.7, color='purple')
        ax6.set_xlabel('Pass Number')
        ax6.set_ylabel('Data Points')
        ax6.set_title('Data Coverage by Pass')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.stats_dir / 'comprehensive_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive analysis plot saved: {output_file}")

    def spatial_pattern_analysis(self, df):
        """Perform spatial pattern analysis."""
        print("\n=== SPATIAL PATTERN ANALYSIS ===")
        
        # Create spatial maps
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SWOT SSH Spatial Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall spatial map
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(df['longitude'], df['latitude'], c=df['ssh'], 
                              cmap='RdYlBu_r', s=2, alpha=0.6)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('SSH Spatial Distribution')
        plt.colorbar(scatter1, ax=ax1, label='SSH (m)')
        ax1.grid(True, alpha=0.3)
        
        # 2. SSH anomalies spatial
        ssh_mean = df['ssh'].mean()
        df['ssh_anomaly'] = df['ssh'] - ssh_mean
        
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(df['longitude'], df['latitude'], c=df['ssh_anomaly'], 
                              cmap='RdBu_r', s=2, alpha=0.6)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('SSH Anomalies (SSH - Mean)')
        plt.colorbar(scatter2, ax=ax2, label='SSH Anomaly (m)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Pass tracks
        ax3 = axes[1, 0]
        unique_passes = df['pass'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_passes)))
        
        for i, pass_num in enumerate(sorted(unique_passes)[:10]):  # Show first 10 passes
            pass_data = df[df['pass'] == pass_num]
            ax3.plot(pass_data['longitude'], pass_data['latitude'], 
                    color=colors[i], alpha=0.7, linewidth=1, 
                    label=f'Pass {pass_num}')
        
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_title('SWOT Pass Tracks')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Mesoscale structures (using SSH gradient)
        ax4 = axes[1, 1]
        
        # Simple gradient detection
        high_gradient = df[np.abs(df['ssh_anomaly']) > df['ssh_anomaly'].std()]
        
        ax4.scatter(df['longitude'], df['latitude'], c='lightblue', s=0.5, alpha=0.3, label='Background')
        ax4.scatter(high_gradient['longitude'], high_gradient['latitude'], 
                   c=high_gradient['ssh_anomaly'], cmap='RdBu_r', s=5, alpha=0.8)
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('Potential Mesoscale Features\n(High SSH Gradient Regions)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.spatial_dir / 'spatial_pattern_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Spatial pattern analysis saved: {output_file}")

    def internal_wave_analysis(self, df):
        """Perform internal wave analysis using high-pass filtering."""
        print("\n=== INTERNAL WAVE ANALYSIS ===")
        
        # Group by pass for along-track analysis
        passes = df['pass'].unique()
        internal_wave_results = []
        
        print(f"Analyzing internal waves in {len(passes)} passes...")
        
        for pass_num in sorted(passes)[:5]:  # Analyze first 5 passes
            pass_data = df[df['pass'] == pass_num].sort_values('time')
            
            if len(pass_data) < 50:
                continue
                
            print(f"  Analyzing Pass {pass_num} ({len(pass_data)} points)")
            
            # Calculate along-track distance (approximate)
            lats = pass_data['latitude'].values
            lons = pass_data['longitude'].values
            
            # Simple distance calculation
            distances = np.zeros(len(lats))
            for i in range(1, len(lats)):
                dlat = lats[i] - lats[i-1]
                dlon = lons[i] - lons[i-1]
                distances[i] = distances[i-1] + np.sqrt((dlat * 111000)**2 + 
                                                       (dlon * 111000 * np.cos(np.radians(lats[i])))**2)
            
            ssh_values = pass_data['ssh'].values
            
            # High-pass filter to extract internal waves
            if len(ssh_values) > 20:
                # Remove large-scale trend
                window_size = min(len(ssh_values) // 10, 20)
                if window_size >= 5:
                    try:
                        from scipy.signal import savgol_filter
                        ssh_smooth = savgol_filter(ssh_values, window_size, 3)
                        internal_wave_signal = ssh_values - ssh_smooth
                        
                        # Find wavelengths
                        peaks, _ = signal.find_peaks(internal_wave_signal, height=np.std(internal_wave_signal)/3)
                        
                        wavelengths = []
                        if len(peaks) > 1:
                            for i in range(len(peaks)-1):
                                wavelength = distances[peaks[i+1]] - distances[peaks[i]]
                                if 1000 < wavelength < 50000:  # 1-50 km range
                                    wavelengths.append(wavelength)
                        
                        result = {
                            'pass': pass_num,
                            'n_points': len(pass_data),
                            'signal_std': np.std(internal_wave_signal),
                            'n_wavelengths': len(wavelengths),
                            'mean_wavelength': np.mean(wavelengths) if wavelengths else np.nan,
                            'max_amplitude': np.max(np.abs(internal_wave_signal))
                        }
                        
                        internal_wave_results.append(result)
                        
                        # Create individual pass visualization
                        self._plot_internal_waves(distances/1000, ssh_values, internal_wave_signal, 
                                                pass_num, wavelengths)
                        
                    except Exception as e:
                        print(f"    Error in internal wave analysis: {e}")
        
        # Create summary
        if internal_wave_results:
            self._create_internal_wave_summary(internal_wave_results)
        
        return internal_wave_results

    def _plot_internal_waves(self, distances_km, ssh_original, wave_signal, pass_num, wavelengths):
        """Plot internal wave analysis for a single pass."""
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Original SSH
        axes[0].plot(distances_km, ssh_original, 'b-', linewidth=1, label='Original SSH')
        axes[0].set_ylabel('SSH (m)')
        axes[0].set_title(f'Pass {pass_num}: SSH and Internal Wave Analysis')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Internal wave signal
        axes[1].plot(distances_km, wave_signal, 'r-', linewidth=1, label='Internal Wave Signal')
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Along-track Distance (km)')
        axes[1].set_ylabel('Wave Signal (m)')
        axes[1].set_title('High-frequency Internal Wave Signal')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Add statistics
        stats_text = f"""
        Statistics:
        • Signal std: {np.std(wave_signal):.3f} m
        • Wavelengths detected: {len(wavelengths)}
        • Mean wavelength: {np.mean(wavelengths)/1000:.1f} km (if detected)
        • Max amplitude: {np.max(np.abs(wave_signal)):.3f} m
        """
        
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        output_file = self.internal_waves_dir / f'internal_waves_pass_{pass_num}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_internal_wave_summary(self, results):
        """Create summary of internal wave analysis."""
        
        df_results = pd.DataFrame(results)
        
        # Save results
        df_results.to_csv(self.internal_waves_dir / 'internal_wave_results.csv', index=False)
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Internal Wave Analysis Summary', fontsize=14, fontweight='bold')
        
        # Signal strength
        axes[0, 0].bar(df_results['pass'], df_results['signal_std'], alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Pass Number')
        axes[0, 0].set_ylabel('Signal Std (m)')
        axes[0, 0].set_title('Internal Wave Signal Strength')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Wavelength distribution
        valid_wavelengths = df_results[df_results['mean_wavelength'].notna()]['mean_wavelength'] / 1000
        if len(valid_wavelengths) > 0:
            axes[0, 1].hist(valid_wavelengths, bins=10, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Mean Wavelength (km)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Wavelength Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Detection success
        detection_rate = (df_results['n_wavelengths'] > 0).sum() / len(df_results) * 100
        axes[1, 0].bar(['Detected', 'Not Detected'], [detection_rate, 100-detection_rate],
                      color=['green', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_title(f'Detection Rate: {detection_rate:.1f}%')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Amplitude vs signal strength
        axes[1, 1].scatter(df_results['signal_std'], df_results['max_amplitude'], alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Signal Std (m)')
        axes[1, 1].set_ylabel('Max Amplitude (m)')
        axes[1, 1].set_title('Signal Strength vs Amplitude')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.internal_waves_dir / 'internal_wave_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Internal wave summary saved: {output_file}")

    def run_analysis(self):
        """Run the complete analysis workflow."""
        
        print("🌊 STARTING ROBUST SWOT ANALYSIS 🌊")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # 1. Load data
            self.load_and_process_data()
            
            if not self.all_data:
                print("❌ No valid data found!")
                return
            
            # 2. Basic statistics
            df, daily_stats, anomalies = self.basic_statistics_analysis()
            
            # 3. Spatial analysis
            self.spatial_pattern_analysis(df)
            
            # 4. Internal wave analysis
            internal_results = self.internal_wave_analysis(df)
            
            # 5. Final summary
            self.create_final_summary(df, daily_stats, internal_results)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 50)
            print("🎉 ROBUST ANALYSIS COMPLETED! 🎉")
            print(f"Duration: {duration}")
            print(f"Output: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            raise

    def create_final_summary(self, df, daily_stats, internal_results):
        """Create final summary report."""
        
        print("\n=== CREATING FINAL SUMMARY ===")
        
        # Summary statistics
        summary = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_points': len(df),
            'time_span': (df['time'].max() - df['time'].min()).days,
            'ssh_mean': df['ssh'].mean(),
            'ssh_std': df['ssh'].std(),
            'ssh_range': df['ssh'].max() - df['ssh'].min(),
            'lat_range': [df['latitude'].min(), df['latitude'].max()],
            'lon_range': [df['longitude'].min(), df['longitude'].max()],
            'n_passes': df['pass'].nunique(),
            'internal_wave_detections': len(internal_results) if internal_results else 0
        }
        
        # Create summary figure
        fig = plt.figure(figsize=(16, 12))
        
        # Title and summary text
        fig.suptitle('SWOT L2 SSH Analysis Summary Report', fontsize=16, fontweight='bold')
        
        # Create text summary
        summary_text = f"""
        SWOT L2 SSH ANALYSIS SUMMARY
        Analysis Date: {summary['analysis_date']}
        
        DATA OVERVIEW:
        • Total data points: {summary['total_points']:,}
        • Time span: {summary['time_span']} days
        • Number of passes: {summary['n_passes']}
        • Latitude range: {summary['lat_range'][0]:.2f}° to {summary['lat_range'][1]:.2f}°
        • Longitude range: {summary['lon_range'][0]:.2f}° to {summary['lon_range'][1]:.2f}°
        
        SSH STATISTICS:
        • Mean SSH: {summary['ssh_mean']:.3f} m
        • SSH Std: {summary['ssh_std']:.3f} m
        • SSH Range: {summary['ssh_range']:.3f} m
        
        ANALYSIS RESULTS:
        ✓ Basic Quality & Statistics
        ✓ Spatial Pattern Analysis
        ✓ Internal Wave Detection ({summary['internal_wave_detections']} passes analyzed)
        ✓ Mesoscale Structure Analysis
        
        OUTPUT DIRECTORIES:
        • Statistics: {self.stats_dir.name}/
        • Spatial Analysis: {self.spatial_dir.name}/
        • Internal Waves: {self.internal_waves_dir.name}/
        """
        
        # Add summary text to figure
        ax_text = fig.add_subplot(111)
        ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        ax_text.axis('off')
        
        # Save summary
        output_file = self.output_dir / 'analysis_summary_report.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary as JSON
        import json
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Final summary saved: {output_file}")
        print(f"Summary statistics: {self.output_dir / 'summary_statistics.json'}")


def main():
    """Main execution."""
    data_dir = r"C:\Users\abhik\Desktop\project related work\SWOT_L2_LR_SSH_2.0_2.0-20250822_162623"
    
    analyzer = RobustSWOTAnalyzer(data_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
