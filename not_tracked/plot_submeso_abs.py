import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

def _bar_plot_abs_per_date(df, col, title, y_label, save_name, save=True, std_threshold=None):
    """
    Plots the absolute magnitude of the anomaly for each cyclone across the timeline.
    Colors bars red for original warm (positive) anomalies and blue for cold (negative).
    """
    if col not in df.columns:
        print(f"No data for {col}")
        return

    # Check if the corresponding std column exists
    std_col = f"{col}_Std"
    has_std = std_col in df.columns

    # Keep only rows with valid anomaly/std/date values
    if has_std:
        valid = df.dropna(subset=[col, std_col, 'Peak_Date']).copy()
    else:
        valid = df.dropna(subset=[col, 'Peak_Date']).copy()

    if valid.empty:
        print(f"No data for {col}")
        return

    # Ensure dates are datetime
    valid['Peak_Date'] = pd.to_datetime(valid['Peak_Date'])

    # Sort chronologically
    valid = valid.sort_values('Peak_Date')

    # Color bars
    colors = ['#d62728' if v >= 0 else '#1f77b4' for v in valid[col]]

    # Absolute values for plotting
    abs_vals = valid[col].abs()

    # Figure size
    fig_width = min(20, max(12, len(valid) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Plot bars
    bars = ax.bar(
        valid['Peak_Date'],
        abs_vals,
        color=colors,
        edgecolor=None,
        linewidth=0,
        width=0.7
    )

    # Zero line
    ax.axhline(0, color='gray', linewidth=0.8, zorder=2)

    # Titles and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Peak Date', fontsize=12, fontweight='bold')

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    if len(valid) <= 12:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Restrict timeline
    ax.set_xlim([
        pd.to_datetime('2023-12-15'),
        pd.to_datetime('2025-01-15')
    ])

    fig.autofmt_xdate(rotation=45)

    ax.grid(axis='y', alpha=0.3)

    # Offset for labels
    ymin, ymax = abs_vals.min(), abs_vals.max()
    offset = 0.02 * (ymax - ymin) if ymax != ymin else 0.01

    # Legend for Warm/Cold
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Warm (Original > 0)'),
        Patch(facecolor='#1f77b4', label='Cold (Original < 0)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # ============================================================
    # STD THRESHOLD
    # ============================================================
    actual_std_threshold = (
        std_threshold
        if std_threshold is not None
        else valid[std_col].mean() if has_std else 0
    )

    # ============================================================
    # BAR ANNOTATIONS
    # ============================================================
    for i, (bar, name) in enumerate(zip(bars, valid['Cyclone'])):
        height = bar.get_height()

        # Standard deviation anomaly text
        std_text = ""
        if has_std:
            std_val = valid[std_col].iloc[i]
            if std_val > actual_std_threshold:
                std_text = f"\nΔσ={std_val:.3f}"

        annot_text = f"{name}{std_text}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            annot_text,
            ha='right',
            va='bottom',
            fontsize=6,
            rotation=45,
            rotation_mode='anchor',
            color='#333333'
        )

    # ============================================================
    # STATISTICS BOX
    # ============================================================
    col_mean = valid[col].mean()
    col_std = valid[col].std()
    
    # We show the original data stats (mean and std) but can add abs mean
    abs_mean = abs_vals.mean()

    stats_text = (
        f"n = {len(valid)}\n"
        f"Original Mean = {col_mean:.3f}\n"
        f"Original Std = {col_std:.3f}\n"
        f"Abs Mean = {abs_mean:.3f}"
    )

    if has_std:
        stats_text += f"\nΔσ Threshold = {actual_std_threshold:.3f}"

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

    # Layout adjustment
    plt.subplots_adjust(
        bottom=0.25,
        top=0.9,
        left=0.08,
        right=0.98
    )

    # Save figure
    if save:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_name}")

    plt.close(fig)

def plot_submeso_abs(df):
    scales = [
        ('SST_Submeso', 'Sub-mesoscale SST Absolute Anomaly Magnitude', 'Absolute SST Anomaly (°C)', 0.10),
        ('MSLA_Submeso', 'Sub-mesoscale MSLA Absolute Anomaly Magnitude', 'Absolute MSLA Anomaly (m)', 0.05)
    ]
    for col, title, y_label, std_thresh in scales:
        _bar_plot_abs_per_date(df, col, title, y_label, f"{col}_Abs_vs_Date.png", save=True, std_threshold=std_thresh)

if __name__ == '__main__':
    csv_file = 'cyclone_multiscale_anomalies.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
    else:
        df = pd.read_csv(csv_file)
        plot_submeso_abs(df)
