"""
Validate similarity analysis predictions against actual AUROC performance.

This script loads similarity metrics and actual anomaly detection results
to demonstrate the predictive power of separability metrics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import stats


def load_results(results_dir="../results"):
    """Load both similarity metrics and anomaly detection results."""
    results_dir = Path(results_dir)

    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    data = []

    for category in categories:
        # Load similarity metrics
        sim_file = results_dir / category / "similarity_analysis" / "similarity_metrics.json"
        auroc_file = results_dir / category / "metrics.json"

        if sim_file.exists() and auroc_file.exists():
            with open(sim_file, 'r') as f:
                sim_data = json.load(f)
            with open(auroc_file, 'r') as f:
                auroc_data = json.load(f)

            data.append({
                'category': category,
                'separability': sim_data['separability'],
                'cohens_d': sim_data['cohens_d'],
                'auroc': auroc_data['auroc'],
                'avg_precision': auroc_data['average_precision'],
            })
        else:
            print(f"Warning: Missing data for {category}")

    return pd.DataFrame(data)


def create_validation_plots(df, save_dir="../results/validation"):
    """Create plots showing prediction vs actual performance."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Separability vs AUROC
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    scatter = ax.scatter(df['separability'], df['auroc'],
                        s=200, alpha=0.6, edgecolors='black', linewidth=2)

    # Add labels
    for _, row in df.iterrows():
        ax.annotate(row['category'],
                   (row['separability'], row['auroc']),
                   fontsize=9, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')

    # Fit regression line
    z = np.polyfit(df['separability'], df['auroc'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['separability'].min(), df['separability'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8,
            label=f'Linear fit: AUROC = {z[0]:.3f}×Sep + {z[1]:.3f}')

    # Calculate correlation
    correlation, p_value = stats.pearsonr(df['separability'], df['auroc'])

    ax.set_xlabel('Separability Score (Predicted Difficulty)', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUROC (Actual Performance)', fontsize=14, fontweight='bold')
    ax.set_title(f'Validation: Separability Predicts Zero-Shot Performance\n' +
                 f'Pearson r = {correlation:.3f}, p < {p_value:.2e}',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (>0.8)')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add text box with statistics
    textstr = f'Correlation: r = {correlation:.3f}\np-value: {p_value:.2e}\nR² = {correlation**2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_dir / 'separability_vs_auroc.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'separability_vs_auroc.png'}")
    plt.close()

    # 2. Cohen's d vs AUROC
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(df['cohens_d'], df['auroc'],
                        s=200, alpha=0.6, edgecolors='black', linewidth=2,
                        c=df['separability'], cmap='RdYlGn')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Separability', fontsize=12, fontweight='bold')

    for _, row in df.iterrows():
        ax.annotate(row['category'],
                   (row['cohens_d'], row['auroc']),
                   fontsize=9, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')

    # Fit regression
    z = np.polyfit(df['cohens_d'], df['auroc'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['cohens_d'].min(), df['cohens_d'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8,
            label=f'Linear fit: AUROC = {z[0]:.3f}×d + {z[1]:.3f}')

    correlation, p_value = stats.pearsonr(df['cohens_d'], df['auroc'])

    ax.set_xlabel('Cohen\'s d (Effect Size)', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUROC (Actual Performance)', fontsize=14, fontweight='bold')
    ax.set_title(f'Effect Size vs Performance\n' +
                 f'Pearson r = {correlation:.3f}, p < {p_value:.2e}',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'cohens_d_vs_auroc.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'cohens_d_vs_auroc.png'}")
    plt.close()

    # 3. Performance ranking comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Sort by separability
    df_sep = df.sort_values('separability', ascending=False)
    colors_sep = ['green' if s > 1.0 else 'orange' if s > 0.5 else 'red'
                  for s in df_sep['separability']]

    ax1.barh(df_sep['category'], df_sep['separability'],
             color=colors_sep, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Separability (Predicted)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Category', fontsize=12, fontweight='bold')
    ax1.set_title('Predicted Difficulty Ranking', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Sort by AUROC
    df_auroc = df.sort_values('auroc', ascending=False)
    colors_auroc = ['green' if a > 0.9 else 'orange' if a > 0.8 else 'red'
                    for a in df_auroc['auroc']]

    ax2.barh(df_auroc['category'], df_auroc['auroc'],
             color=colors_auroc, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('AUROC (Actual)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Category', fontsize=12, fontweight='bold')
    ax2.set_title('Actual Performance Ranking', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim([0.8, 1.0])

    plt.tight_layout()
    plt.savefig(save_dir / 'predicted_vs_actual_ranking.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'predicted_vs_actual_ranking.png'}")
    plt.close()


def print_validation_summary(df):
    """Print validation statistics."""
    print("\n" + "=" * 80)
    print("VALIDATION: SIMILARITY ANALYSIS vs ACTUAL AUROC")
    print("=" * 80)

    # Correlations
    sep_corr, sep_p = stats.pearsonr(df['separability'], df['auroc'])
    cohens_corr, cohens_p = stats.pearsonr(df['cohens_d'], df['auroc'])

    print(f"\n📊 CORRELATION ANALYSIS:")
    print(f"  Separability vs AUROC:")
    print(f"    Pearson r = {sep_corr:.4f}")
    print(f"    p-value = {sep_p:.2e}")
    print(f"    R² = {sep_corr**2:.4f}")
    print(f"    {'✓ Statistically significant' if sep_p < 0.05 else '✗ Not significant'}")

    print(f"\n  Cohen's d vs AUROC:")
    print(f"    Pearson r = {cohens_corr:.4f}")
    print(f"    p-value = {cohens_p:.2e}")
    print(f"    R² = {cohens_corr**2:.4f}")
    print(f"    {'✓ Statistically significant' if cohens_p < 0.05 else '✗ Not significant'}")

    # Prediction accuracy
    print(f"\n🎯 PREDICTION ACCURACY:")

    # Categories with separability > 1.0 should have AUROC > 0.9
    easy_cats = df[df['separability'] > 1.0]
    easy_correct = (easy_cats['auroc'] > 0.9).sum()
    print(f"  High separability (>1.0) → High AUROC (>0.9):")
    print(f"    {easy_correct}/{len(easy_cats)} correct ({100*easy_correct/len(easy_cats):.1f}%)")

    # Categories with separability < 0.5 should have AUROC < 0.95
    hard_cats = df[df['separability'] < 0.5]
    hard_correct = (hard_cats['auroc'] < 0.95).sum()
    print(f"  Low separability (<0.5) → Moderate AUROC (<0.95):")
    print(f"    {hard_correct}/{len(hard_cats)} correct ({100*hard_correct/len(hard_cats):.1f}%)")

    # Best and worst predictions
    print(f"\n🏆 BEST PREDICTIONS:")
    df['prediction_error'] = abs(df['auroc'] - (0.6 + 0.2 * df['separability']))
    best = df.nsmallest(3, 'prediction_error')
    for _, row in best.iterrows():
        print(f"  {row['category']}: Predicted (sep={row['separability']:.2f}) " +
              f"→ Actual AUROC={row['auroc']:.4f}")

    print(f"\n⚠️  LARGEST DEVIATIONS:")
    worst = df.nlargest(3, 'prediction_error')
    for _, row in worst.iterrows():
        print(f"  {row['category']}: Separability={row['separability']:.2f}, " +
              f"AUROC={row['auroc']:.4f} (diff={row['prediction_error']:.4f})")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Category':<15} {'Separability':<15} {'AUROC':<10} {'Match':<10}")
    print("-" * 80)

    df_sorted = df.sort_values('separability', ascending=False)
    for _, row in df_sorted.iterrows():
        match = "✓" if (row['separability'] > 1.0 and row['auroc'] > 0.9) or \
                      (row['separability'] < 0.5 and row['auroc'] < 0.95) else "~"
        print(f"{row['category']:<15} {row['separability']:<15.4f} {row['auroc']:<10.4f} {match:<10}")

    print("=" * 80 + "\n")


def main():
    print("Loading results...")
    df = load_results()

    print(f"Found {len(df)} categories with complete data\n")

    # Print validation summary
    print_validation_summary(df)

    # Create plots
    print("Generating validation plots...")
    create_validation_plots(df)

    # Save to CSV
    output_file = Path("../results/validation_comparison.csv")
    df.to_csv(output_file, index=False)
    print(f"\n📁 Saved validation data to: {output_file}")

    print("\n✅ Validation complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
