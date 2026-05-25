import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

def plot_gradient_timeline(df, col, title, y_label, save_name):
    valid = df.dropna(subset=[col, 'Peak_Date']).copy()
    if valid.empty:
        print(f"No data for {col}")
        return
        
    valid['Peak_Date'] = pd.to_datetime(valid['Peak_Date'])
    valid = valid.sort_values('Peak_Date')

    # Figure size
    fig_width = min(20, max(12, len(valid) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # All values are magnitudes (>=0), using a distinct color
    color = '#2ca02c' # Green for spatial gradient

    bars = ax.bar(
        valid['Peak_Date'],
        valid[col],
        color=color,
        edgecolor=None,
        linewidth=0,
        width=0.7
    )

    ax.axhline(0, color='gray', linewidth=0.8, zorder=2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Peak Date', fontsize=12, fontweight='bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    if len(valid) <= 12:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    ax.set_xlim([
        pd.to_datetime('2023-12-15'),
        pd.to_datetime('2025-01-15')
    ])

    fig.autofmt_xdate(rotation=45)
    ax.grid(axis='y', alpha=0.3)

    ymin, ymax = valid[col].min(), valid[col].max()
    offset = 0.02 * (ymax - ymin) if ymax != ymin else 0.01

    for bar, name in zip(bars, valid['Cyclone']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            name,
            ha='right',
            va='bottom',
            fontsize=6,
            rotation=45,
            rotation_mode='anchor',
            color='#333333'
        )

    # Statistics box
    col_mean = valid[col].mean()
    col_std = valid[col].std()

    stats_text = (
        f"n = {len(valid)}\n"
        f"Mean = {col_mean:.5f}\n"
        f"Std Dev = {col_std:.5f}"
    )

    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            alpha=0.8,
            edgecolor='#cccccc'
        )
    )

    plt.subplots_adjust(
        bottom=0.25,
        top=0.9,
        left=0.08,
        right=0.98
    )

    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_name}")
    plt.close(fig)

if __name__ == '__main__':
    csv_file = 'cyclone_multiscale_anomalies.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
    else:
        df = pd.read_csv(csv_file)
        plot_gradient_timeline(
            df,
            col='SST_Submeso_Grad_Avg',
            title='Sub-mesoscale SST Average Spatial Gradient Magnitude',
            y_label='Gradient Magnitude (°C/km)',
            save_name='SST_Submeso_Grad_Avg_vs_Date.png'
        )
