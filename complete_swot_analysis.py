#!/usr/bin/env python3
"""
Final SWOT Analysis for Andaman and Nicobar Region
=================================================
Complete analysis with proper date handling and comprehensive outputs.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def final_analysis():
    """Complete SWOT analysis for Andaman and Nicobar region."""
    
    # Configuration
    data_dir = Path(r"C:/Users/abhik/Desktop/project related work/SWOT_L2_LR_SSH_2.0_2.0-20250822_162623")
    output_dir = Path("swot_analysis_output/figures_andaman_nicobar")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target region
    region = {
        'lat_min': 6.0, 'lat_max': 15.0,
        'lon_min': 92.0, 'lon_max': 95.0
    }
    
    print("="*60)
    print("FINAL SWOT ANALYSIS - ANDAMAN AND NICOBAR REGION")
    print("="*60)
    print(f"Target region: {region['lat_min']}°N-{region['lat_max']}°N, {region['lon_min']}°E-{region['lon_max']}°E")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find Basic SSH files
    files = list(data_dir.glob("*Basic*.nc"))
    print(f"Found {len(files)} Basic SSH files")
    print(f"Processing first 20 files for analysis...")
    print()
    
    # Process files (limit to first 20 for manageable analysis)
    all_data = []
    file_info = []
    max_files = 20
    
    for i, file_path in enumerate(files[:max_files]):
        print(f"Processing file {i+1}/{max_files}: {file_path.name}")
        
        try:
            # Load dataset
            ds = xr.open_dataset(file_path)
            
            # Filter for region
            lat_mask = (ds.latitude >= region['lat_min']) & (ds.latitude <= region['lat_max'])
            lon_mask = (ds.longitude >= region['lon_min']) & (ds.longitude <= region['lon_max'])
            region_mask = lat_mask & lon_mask
            
            # Check if we have data in region
            n_points = int(region_mask.sum())
            if n_points == 0:
                print(f"  No data in region")
                ds.close()
                continue
            
            print(f"  Found {n_points:,} data points in region")
            
            # Extract data
            lats = ds.latitude.where(region_mask).values.flatten()
            lons = ds.longitude.where(region_mask).values.flatten()
            
            # Get SSH data
            ssh_data = {}
            for var in ['ssh_karin', 'ssh_karin_2']:
                if var in ds.data_vars:
                    ssh_data[var] = ds[var].where(region_mask).values.flatten()
            
            # Extract time from filename properly
            filename_parts = file_path.name.split('_')
            # Format: SWOT_L2_LR_SSH_Basic_CCC_PPP_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_PGC0_01.nc
            # Index:  0    1  2  3    4     5   6   7               8               9    10
            cycle = filename_parts[4]  # Cycle number (001, 002, etc.)
            pass_num = filename_parts[5]  # Pass number (230, 258, etc.)
            start_time = filename_parts[6]  # Start time YYYYMMDDTHHMMSS
            
            date_str = start_time[:8]  # YYYYMMDD
            datetime_obj = pd.to_datetime(start_time, format='%Y%m%dT%H%M%S')
            
            # Store file info
            file_info.append({
                'file': file_path.name,
                'cycle': cycle,
                'pass': pass_num,
                'date': date_str,
                'datetime': datetime_obj,
                'n_points': n_points
            })
            
            # Sample data for large datasets to keep manageable
            sample_size = min(n_points, 5000)  # Max 5000 points per file
            valid_indices = []
            
            for j in range(len(lats)):
                if not np.isnan(lats[j]) and not np.isnan(lons[j]):
                    valid_indices.append(j)
                    
            # Sample if too many points
            if len(valid_indices) > sample_size:
                import random
                valid_indices = random.sample(valid_indices, sample_size)
            
            # Store sampled data points
            for j in valid_indices:
                data_point = {
                    'cycle': cycle,
                    'pass': pass_num,
                    'date': date_str,
                    'datetime': datetime_obj,
                    'latitude': lats[j],
                    'longitude': lons[j],
                    'file': file_path.name
                }
                
                # Add SSH values
                for var, values in ssh_data.items():
                    data_point[var] = values[j] if j < len(values) else np.nan
                
                all_data.append(data_point)
            
            ds.close()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not all_data:
        print("No valid data found!")
        return
    
    # Create DataFrames
    df = pd.DataFrame(all_data)
    df = df.dropna(subset=['latitude', 'longitude'])
    
    file_df = pd.DataFrame(file_info)
    
    print(f"\nDATA SUMMARY:")
    print(f"Total data points: {len(df):,}")
    print(f"Files processed: {len(file_df)}")
    print(f"Unique dates: {len(df['date'].unique())}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Latitude range: {df['latitude'].min():.3f}° to {df['latitude'].max():.3f}°")
    print(f"Longitude range: {df['longitude'].min():.3f}° to {df['longitude'].max():.3f}°")
    
    # SSH statistics
    print(f"\nSSH STATISTICS:")
    for var in ['ssh_karin', 'ssh_karin_2']:
        if var in df.columns:
            ssh_data = df[var].dropna()
            if len(ssh_data) > 0:
                print(f"\n{var}:")
                print(f"  Count: {len(ssh_data):,}")
                print(f"  Mean: {ssh_data.mean():.3f} m")
                print(f"  Std: {ssh_data.std():.3f} m")
                print(f"  Min: {ssh_data.min():.3f} m")
                print(f"  Max: {ssh_data.max():.3f} m")
    
    # Save data
    df.to_csv(output_dir / "final_ssh_analysis.csv", index=False)
    file_df.to_csv(output_dir / "file_summary.csv", index=False)
    
    # Create plots
    create_final_plots(df, output_dir)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"Results saved in: {output_dir}")
    print(f"Files created:")
    print(f"  - final_ssh_analysis.csv")
    print(f"  - file_summary.csv")
    print(f"  - spatial_temporal_analysis.png")
    print(f"{'='*60}")

def create_final_plots(df, output_dir):
    """Create final comprehensive plots."""
    print(f"\nCreating final plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SWOT SSH Analysis - Andaman and Nicobar Region\nSpatial and Temporal Analysis', fontsize=16)
    
    # 1. SSH Karin spatial distribution
    ax1 = axes[0, 0]
    ssh1_data = df[df['ssh_karin'].notna()]
    if len(ssh1_data) > 0:
        scatter = ax1.scatter(ssh1_data['longitude'], ssh1_data['latitude'], 
                            c=ssh1_data['ssh_karin'], cmap='RdYlBu_r', s=1, alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='SSH (m)')
        ax1.set_title(f'SSH Karin Distribution\n({len(ssh1_data):,} points)')
        ax1.set_xlabel('Longitude (°E)')
        ax1.set_ylabel('Latitude (°N)')
        ax1.grid(True, alpha=0.3)
    
    # 2. SSH Karin 2 spatial distribution
    ax2 = axes[0, 1]
    ssh2_data = df[df['ssh_karin_2'].notna()]
    if len(ssh2_data) > 0:
        scatter = ax2.scatter(ssh2_data['longitude'], ssh2_data['latitude'], 
                            c=ssh2_data['ssh_karin_2'], cmap='RdYlBu_r', s=1, alpha=0.6)
        plt.colorbar(scatter, ax=ax2, label='SSH (m)')
        ax2.set_title(f'SSH Karin 2 Distribution\n({len(ssh2_data):,} points)')
        ax2.set_xlabel('Longitude (°E)')
        ax2.set_ylabel('Latitude (°N)')
        ax2.grid(True, alpha=0.3)
    
    # 3. SSH histograms
    ax3 = axes[0, 2]
    if 'ssh_karin' in df.columns:
        ssh1 = df['ssh_karin'].dropna()
        if len(ssh1) > 0:
            ax3.hist(ssh1, bins=50, alpha=0.7, label=f'SSH Karin (n={len(ssh1):,})', density=True)
    if 'ssh_karin_2' in df.columns:
        ssh2 = df['ssh_karin_2'].dropna()
        if len(ssh2) > 0:
            ax3.hist(ssh2, bins=50, alpha=0.7, label=f'SSH Karin 2 (n={len(ssh2):,})', density=True)
    ax3.set_xlabel('SSH (m)')
    ax3.set_ylabel('Density')
    ax3.set_title('SSH Value Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Temporal analysis - daily means
    ax4 = axes[1, 0]
    if len(df['datetime'].unique()) > 1:
        daily_stats = df.groupby('datetime').agg({
            'ssh_karin': 'mean',
            'ssh_karin_2': 'mean'
        })
        if 'ssh_karin' in daily_stats.columns:
            ax4.plot(daily_stats.index, daily_stats['ssh_karin'], 'o-', label='SSH Karin', linewidth=2)
        if 'ssh_karin_2' in daily_stats.columns:
            ax4.plot(daily_stats.index, daily_stats['ssh_karin_2'], 's-', label='SSH Karin 2', linewidth=2)
        ax4.set_ylabel('Mean SSH (m)')
        ax4.set_title('Daily Mean SSH')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Single day data\nNo temporal variation', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Temporal Analysis')
    
    # 5. Data distribution by cycle/pass
    ax5 = axes[1, 1]
    cycle_counts = df.groupby(['cycle', 'pass']).size().reset_index()
    cycle_counts['label'] = cycle_counts['cycle'] + '_' + cycle_counts['pass']
    ax5.bar(range(len(cycle_counts)), cycle_counts[0], alpha=0.7)
    ax5.set_xlabel('Cycle_Pass')
    ax5.set_ylabel('Data Points')
    ax5.set_title('Data Distribution by Cycle/Pass')
    ax5.set_xticks(range(len(cycle_counts)))
    ax5.set_xticklabels(cycle_counts['label'], rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. SSH correlation and statistics
    ax6 = axes[1, 2]
    common_data = df[['ssh_karin', 'ssh_karin_2']].dropna()
    if len(common_data) > 10:
        # Sample for plotting if too many points
        if len(common_data) > 5000:
            plot_data = common_data.sample(n=5000)
        else:
            plot_data = common_data
            
        ax6.scatter(plot_data['ssh_karin'], plot_data['ssh_karin_2'], 
                   alpha=0.5, s=0.5)
        ax6.set_xlabel('SSH Karin (m)')
        ax6.set_ylabel('SSH Karin 2 (m)')
        ax6.set_title('SSH Karin vs SSH Karin 2')
        
        # Add correlation and statistics
        correlation = common_data['ssh_karin'].corr(common_data['ssh_karin_2'])
        rmse = np.sqrt(((common_data['ssh_karin'] - common_data['ssh_karin_2'])**2).mean())
        bias = (common_data['ssh_karin'] - common_data['ssh_karin_2']).mean()
        
        stats_text = f'Correlation: {correlation:.4f}\nRMSE: {rmse:.4f} m\nBias: {bias:.4f} m'
        ax6.text(0.05, 0.95, stats_text, 
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add 1:1 line
        min_val = min(plot_data['ssh_karin'].min(), plot_data['ssh_karin_2'].min())
        max_val = max(plot_data['ssh_karin'].max(), plot_data['ssh_karin_2'].max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='1:1 line')
        ax6.legend()
    
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_temporal_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Final plot saved: {output_dir / 'spatial_temporal_analysis.png'}")

if __name__ == "__main__":
    final_analysis()
