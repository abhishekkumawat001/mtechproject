import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import json

# =============================================================================
# STEP 0 — LOAD INTENSITY DATA FROM JSON
# =============================================================================

def load_intensity_data(json_path='cyclone_info_2024_padded.json'):
    """Extract max_wind_mph and min_pressure_mb for every storm."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    records = []
    for basin_key, basin in data.items():
        if not isinstance(basin, dict) or 'storms' not in basin:
            continue
        for s in basin['storms']:
            records.append({
                'Cyclone': s['name'],
                'Category': s.get('category', ''),
                'Max_Wind_mph': s.get('max_wind_mph', np.nan),
                'Min_Pressure_mb': s.get('min_pressure_mb', np.nan),
            })
    return pd.DataFrame(records)


# =============================================================================
# STEP 1 — MERGE ANOMALIES + INTENSITY
# =============================================================================

def build_coci_dataframe(anomaly_csv='cyclone_multiscale_anomalies.csv',
                         json_path='cyclone_info_2024_padded.json'):
    """Merge the pre-computed SST/MSLA anomalies with intensity metadata."""
    df_anom = pd.read_csv(anomaly_csv)
    df_int  = load_intensity_data(json_path)

    # normalise names to uppercase so CSV and JSON always match
    df_anom['Cyclone'] = df_anom['Cyclone'].str.upper()
    df_int['Cyclone']  = df_int['Cyclone'].str.upper()

    # merge on cyclone name
    df = df_anom.merge(df_int, on='Cyclone', how='left')

    # use meso-scale as primary anomaly (most relevant for cyclone coupling)
    df = df.rename(columns={'SST_Meso': 'SST_Anom', 'MSLA_Meso': 'MSLA_Anom'})

    # drop rows missing critical columns
    df = df.dropna(subset=['SST_Anom', 'MSLA_Anom', 'Max_Wind_mph'])

    # z-score normalisation
    for col, zcol in [('SST_Anom', 'Z_SST'),
                      ('MSLA_Anom', 'Z_MSLA'),
                      ('Max_Wind_mph', 'Z_V')]:
        mu, sigma = df[col].mean(), df[col].std()
        df[zcol] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    return df


# =============================================================================
# STEP 2 — COMPUTE ALL INDEX VARIANTS
# =============================================================================

def compute_all_indices(df):
    """Add columns for every COCI formulation."""

    # --- Option A: Equal-weight linear ---
    df['COCI_A'] = (df['Z_SST'] + df['Z_MSLA'] + df['Z_V']) / 3.0

    # --- Option B: Ocean forcing × intensity (multiplicative) ---
    df['COCI_B'] = (df['Z_SST'] + df['Z_MSLA']) * df['Z_V']

    # --- Option C: Correlation-weighted (data-driven) ---
    if len(df) >= 2:
        r_sst, _  = pearsonr(df['SST_Anom'], df['Max_Wind_mph'])
        r_msla, _ = pearsonr(df['MSLA_Anom'], df['Max_Wind_mph'])
    else:
        r_sst, r_msla = 0.5, 0.5
    w_sst  = abs(r_sst)
    w_msla = abs(r_msla)
    w_sum  = w_sst + w_msla
    if w_sum > 0:
        df['COCI_C'] = (w_sst * df['Z_SST'] + w_msla * df['Z_MSLA']) / w_sum
    else:
        df['COCI_C'] = (df['Z_SST'] + df['Z_MSLA']) / 2.0

    # --- Option D: Simple ocean-only (no intensity) ---
    df['COCI_D'] = (df['Z_SST'] + df['Z_MSLA']) / 2.0

    # store the correlation weights for reporting
    df.attrs['r_sst']  = r_sst
    df.attrs['r_msla'] = r_msla

    return df


# =============================================================================
# STEP 3 — CLASSIFY EACH CYCLONE
# =============================================================================

def classify_coci(value):
    if value > 2:
        return 'Extreme'
    elif value > 1:
        return 'Strong'
    elif value > 0:
        return 'Moderate'
    elif value > -1:
        return 'Weak'
    else:
        return 'Suppressed'


def add_classifications(df):
    for idx_col in ['COCI_A', 'COCI_B', 'COCI_C', 'COCI_D']:
        df[f'{idx_col}_Class'] = df[idx_col].apply(classify_coci)
    return df


# =============================================================================
# STEP 4 — VALIDATE: correlate each index with Vmax
# =============================================================================

def validate_indices(df):
    """Return a summary DataFrame of validation metrics."""
    results = []
    for idx_col, label in [
        ('COCI_A', 'A: Equal-Weight Linear'),
        ('COCI_B', 'B: Ocean × Intensity'),
        ('COCI_C', 'C: Correlation-Weighted'),
        ('COCI_D', 'D: Ocean-Only'),
    ]:
        if len(df) >= 2:
            r, p = pearsonr(df[idx_col], df['Max_Wind_mph'])
        else:
            r, p = 0.0, 1.0
        results.append({
            'Index': label,
            'Column': idx_col,
            'Pearson_r': r,
            'p_value': p,
            'Mean': df[idx_col].mean(),
            'Std': df[idx_col].std(),
        })
    return pd.DataFrame(results)


# =============================================================================
# STEP 5 — VALIDATE: mean COCI by cyclone category
# =============================================================================

def category_validation(df):
    """Mean of each index grouped by cyclone category."""
    cat_order = ['TS', '1', '2', '3', '4', '5']
    idx_cols = ['COCI_A', 'COCI_B', 'COCI_C', 'COCI_D']
    grp = df.groupby('Category')[idx_cols].mean()
    grp = grp.reindex([c for c in cat_order if c in grp.index])
    grp['Count'] = df.groupby('Category')['Cyclone'].count()
    return grp


# =============================================================================
# PLOTS
# =============================================================================

def plot_index_comparison(df, save=True):
    """Bar chart comparing all 4 COCI variants for every cyclone."""
    idx_cols = ['COCI_A', 'COCI_B', 'COCI_C', 'COCI_D']
    labels   = ['A: Equal-Wt', 'B: Ocean×V', 'C: Corr-Wt', 'D: Ocean-Only']
    colors   = ['#5C6BC0', '#26A69A', '#FFA726', '#EF5350']

    df_sorted = df.sort_values('Max_Wind_mph')
    x = np.arange(len(df_sorted))
    width = 0.2

    # Cap the width to prevent extremely wide plots when many cyclones are present
    fig_width = min(18, max(14, len(df_sorted) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    for i, (col, lab, clr) in enumerate(zip(idx_cols, labels, colors)):
        ax.bar(x + i * width, df_sorted[col].values, width,
               label=lab, color=clr, edgecolor='white', zorder=3)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(df_sorted['Cyclone'], rotation=60, fontsize=7, ha='right')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_ylabel('COCI Value', fontsize=12, fontweight='bold')
    ax.set_title('COCI Comparison — All Variants (sorted by wind speed)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    plt.tight_layout()
    if save:
        plt.savefig('COCI_Comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: COCI_Comparison.png")
    plt.show()


def plot_best_index_scatter(df, best_col, best_label, save=True):
    """Scatter plot: best COCI vs Max Wind with regression."""
    r, p = pearsonr(df[best_col], df['Max_Wind_mph'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df[best_col], df['Max_Wind_mph'],
                    c=df['Max_Wind_mph'], cmap='YlOrRd',
                    s=80, edgecolors='white', linewidths=0.5, zorder=4)
    # annotate
    for _, row in df.iterrows():
        ax.annotate(row['Cyclone'],
                    xy=(row[best_col], row['Max_Wind_mph']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=5.5, color='#444444')
    # regression
    m, b = np.polyfit(df[best_col].values, df['Max_Wind_mph'].values, 1)
    xs = np.linspace(df[best_col].min(), df[best_col].max(), 200)
    ax.plot(xs, m * xs + b, 'k--', lw=1.5,
            label=f'Fit  r={r:.3f},  p={p:.4f}')

    ax.set_xlabel(f'{best_label}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Wind (mph)', fontsize=12, fontweight='bold')
    ax.set_title(f'Best Index Validation: {best_label} vs Intensity',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.colorbar(sc, ax=ax, label='Max Wind (mph)')
    plt.tight_layout()
    if save:
        plt.savefig('COCI_Best_Scatter.png', dpi=150, bbox_inches='tight')
        print("Saved: COCI_Best_Scatter.png")
    plt.show()


def plot_category_means(cat_df, save=True):
    """Grouped bar chart of mean COCI by cyclone category."""
    idx_cols = ['COCI_A', 'COCI_B', 'COCI_C', 'COCI_D']
    labels   = ['A: Equal-Wt', 'B: Ocean×V', 'C: Corr-Wt', 'D: Ocean-Only']
    colors   = ['#5C6BC0', '#26A69A', '#FFA726', '#EF5350']

    x = np.arange(len(cat_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, lab, clr) in enumerate(zip(idx_cols, labels, colors)):
        vals = cat_df[col].fillna(0).values
        ax.bar(x + i * width, vals, width,
               label=lab, color=clr, edgecolor='white', zorder=3)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"Cat {c}\n(n={int(cat_df.loc[c, 'Count'])})"
                        for c in cat_df.index], fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_ylabel('Mean COCI', fontsize=12, fontweight='bold')
    ax.set_title('Mean COCI by Cyclone Category',
                 fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    plt.tight_layout()
    if save:
        plt.savefig('COCI_Category_Means.png', dpi=150, bbox_inches='tight')
        print("Saved: COCI_Category_Means.png")
    plt.show()


# =============================================================================
# MASTER RUNNER
# =============================================================================

def run_coci_analysis(anomaly_csv='cyclone_multiscale_anomalies.csv',
                      json_path='cyclone_info_2024_padded.json',
                      save=True):
    """End-to-end COCI analysis. Returns (df, validation, category, best_col)."""

    print("=" * 60)
    print("  CYCLONE OCEAN COUPLING INDEX (COCI) ANALYSIS")
    print("=" * 60)

    # 1. Build merged dataframe
    df = build_coci_dataframe(anomaly_csv, json_path)
    print(f"\nMerged {len(df)} cyclones with SST/MSLA anomalies + intensity.\n")

    # 2. Compute all indices
    df = compute_all_indices(df)
    df = add_classifications(df)

    # 3. Validate
    val_df = validate_indices(df)
    print("─── Validation: Pearson r (Index vs Max Wind) ───")
    print(val_df[['Index', 'Pearson_r', 'p_value']].to_string(index=False))

    # 4. Identify best
    best_row = val_df.loc[val_df['Pearson_r'].abs().idxmax()]
    best_col = best_row['Column']
    best_label = best_row['Index']
    print(f"\n★ BEST INDEX: {best_label}  (r = {best_row['Pearson_r']:.4f})")

    # 5. Category validation
    cat_df = category_validation(df)
    print("\n─── Mean COCI by Category ───")
    print(cat_df.to_string())

    # 6. Correlation weights used in Option C
    print(f"\nCorrelation weights (Option C):")
    print(f"  corr(SST, Vmax)  = {df.attrs.get('r_sst', 0):.4f}")
    print(f"  corr(MSLA, Vmax) = {df.attrs.get('r_msla', 0):.4f}")

    # 7. Plots
    plot_index_comparison(df, save)
    plot_best_index_scatter(df, best_col, best_label, save)
    plot_category_means(cat_df, save)

    # 8. Save full table
    out_csv = 'coci_results.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nFull results → {out_csv}")

    val_csv = 'coci_validation.csv'
    val_df.to_csv(val_csv, index=False)
    print(f"Validation   → {val_csv}")

    return df, val_df, cat_df, best_col


# =============================================================================
# RUN  (uncomment to execute)
# =============================================================================
# df_coci, val, cat, best = run_coci_analysis()
