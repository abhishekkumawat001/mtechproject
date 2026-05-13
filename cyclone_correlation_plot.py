import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_anomaly_correlation(df, save=True):
    """
    Scatter of SST vs MSLA mean anomaly across all cyclones,
    with regression line and Pearson r / p-value annotation.
    """
    # Note: Using column names from user's snippet. Ensure these exist in your DataFrame.
    # If using the multiscale CSV, you might need to rename columns or map them here.
    col_x = 'SST_Anomaly'
    col_y = 'MSLA_Anomaly'
    
    if col_x not in df.columns or col_y not in df.columns:
        # Fallback to Meso-scale if user's names aren't found
        col_x, col_y = 'SST_Meso', 'MSLA_Meso'

    valid = df.dropna(subset=[col_x, col_y]).copy()
    if valid.empty:
        print("No paired data."); return

    sst  = valid[col_x].values
    msla = valid[col_y].values
    corr, pval = pearsonr(sst, msla)

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', 'H',
               'X', 'd', 'P', '8', '*']
    
    # Modern colormap lookup
    cmap = plt.get_cmap('tab20', len(valid))

    fig, ax = plt.subplots(figsize=(9, 7))

    # Calculate midpoints for intelligent annotation placement
    x_mid = np.mean(sst)
    y_mid = np.mean(msla)

    for i, (_, row) in enumerate(valid.iterrows()):
        x, y = row[col_x], row[col_y]
        ax.scatter(x, y,
                   marker=markers[i % len(markers)],
                   color=cmap(i), s=90, zorder=4,
                   edgecolors='white', linewidths=0.5,
                   label=row['Cyclone'])
        
        # --- START MODIFIED ANNOTATION LOGIC ---
        # Adjust text position based on location relative to height/width center
        # This prevents labels from going out of plot bounds.
        x_off = 5 if x < x_mid else -5
        y_off = 5 if y < y_mid else -5
        ha = 'left' if x_off > 0 else 'right'
        va = 'bottom' if y_off > 0 else 'top'

        ax.annotate(row['Cyclone'],
                    xy=(x, y),
                    xytext=(x_off, y_off), textcoords='offset points',
                    ha=ha, va=va,
                    fontsize=6.5, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.5))
        # --- END MODIFIED ANNOTATION LOGIC ---

    # regression
    m, b   = np.polyfit(sst, msla, 1)
    xs     = np.linspace(sst.min(), sst.max(), 200)
    ax.plot(xs, m * xs + b, color='crimson', linestyle='--',
            linewidth=1.8, label=f'Linear fit  r={corr:.2f},  p={pval:.3f}')

    ax.axhline(0, color='gray', linewidth=0.6)
    ax.axvline(0, color='gray', linewidth=0.6)

    ax.set_xlabel('Mean SST Anomaly (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean MSLA Anomaly (m)',  fontsize=12, fontweight='bold')
    ax.set_title('SST ↔ MSLA Correlation — All Cyclones',
                 fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.25)

    # legend outside to the right in two columns
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=6.5, framealpha=0.9, ncol=2, 
              title="Cyclones", title_fontsize=8,
              labelspacing=0.3, borderpad=0.5)

    ax.text(0.02, 0.97,
            f'Pearson r = {corr:.3f}\np-value   = {pval:.4f}\nn = {len(valid)}',
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    # Manual adjustment to prevent "Tight layout" warnings and accommodate the 2-column legend
    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.15)
    if save:
        plt.savefig('SST_MSLA_Correlation.png', dpi=150, bbox_inches='tight')
        print("Saved: SST_MSLA_Correlation.png")
    plt.show()

if __name__ == "__main__":
    try:
        df = pd.read_csv('cyclone_multiscale_anomalies.csv')
        plot_anomaly_correlation(df)
    except Exception as e:
        print(f"Error: {e}")
