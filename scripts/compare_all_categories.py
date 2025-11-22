"""
Compare similarity analysis results across all MVTec AD categories.

This script aggregates results from individual category analyses and creates
comparative visualizations and rankings.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns


def load_all_results(results_dir="../results"):
    """Load similarity metrics for all categories."""
    results_dir = Path(results_dir)

    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    all_results = {}

    for category in categories:
        metrics_file = results_dir / category / "similarity_analysis" / "similarity_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_results[category] = json.load(f)
        else:
            print(f"Warning: No results found for {category}")

    return all_results


def create_comparison_dataframe(all_results):
    """Create pandas DataFrame from all results."""
    data = []

    for category, results in all_results.items():
        row = {
            'Category': category,
            'Normal-Normal Mean': results['normal_to_normal']['mean'],
            'Normal-Normal Std': results['normal_to_normal']['std'],
            'Defect-Defect Mean': results['defect_to_defect']['mean'],
            'Defect-Defect Std': results['defect_to_defect']['std'],
            'Normal-Defect Mean': results['normal_to_defect']['mean'],
            'Normal-Defect Std': results['normal_to_defect']['std'],
            'Cohen\'s d': results['cohens_d'],
            'Separability': results['separability'],
            'T-test p-value': results['ttest_normal_vs_defect']['p_value'],
        }
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('Separability', ascending=False)

    return df


def visualize_category_comparison(df, save_dir):
    """Create comprehensive comparison visualizations."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Separability ranking
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['green' if s > 1.0 else 'orange' if s > 0.5 else 'red'
              for s in df['Separability']]

    bars = ax.barh(df['Category'], df['Separability'], color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Excellent (>1.0)')
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Good (>0.5)')
    ax.set_xlabel('Separability Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Category', fontsize=12, fontweight='bold')
    ax.set_title('Category Difficulty Ranking by Separability\n(Higher = Easier for Zero-Shot Detection)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_dir / 'category_separability_ranking.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'category_separability_ranking.png'}")
    plt.close()

    # 2. Cohen's d comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    df_sorted = df.sort_values('Cohen\'s d', ascending=False)
    colors = ['green' if d > 1.2 else 'orange' if d > 0.8 else 'yellow' if d > 0.5 else 'red'
              for d in df_sorted['Cohen\'s d']]

    ax.barh(df_sorted['Category'], df_sorted['Cohen\'s d'], color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=1.2, color='green', linestyle='--', linewidth=2, label='Very Large (>1.2)')
    ax.axvline(x=0.8, color='orange', linestyle='--', linewidth=2, label='Large (>0.8)')
    ax.axvline(x=0.5, color='yellow', linestyle='--', linewidth=2, label='Medium (>0.5)')
    ax.set_xlabel('Cohen\'s d (Effect Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Category', fontsize=12, fontweight='bold')
    ax.set_title('Effect Size Comparison Across Categories\n(Higher = Stronger Separation)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_dir / 'category_cohens_d_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'category_cohens_d_comparison.png'}")
    plt.close()

    # 3. Similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 12))

    heatmap_data = df[['Category', 'Normal-Normal Mean', 'Defect-Defect Mean',
                       'Normal-Defect Mean']].set_index('Category')

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.6, vmax=1.0, ax=ax, cbar_kws={'label': 'Similarity Score'})
    ax.set_title('Similarity Scores Heatmap Across Categories',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Category', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / 'category_similarity_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'category_similarity_heatmap.png'}")
    plt.close()

    # 4. Scatter plot: Separability vs Cohen's d
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(df['Cohen\'s d'], df['Separability'],
                        s=200, alpha=0.6, edgecolors='black', linewidth=2)

    # Add category labels
    for idx, row in df.iterrows():
        ax.annotate(row['Category'],
                   (row['Cohen\'s d'], row['Separability']),
                   fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('Cohen\'s d (Effect Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Separability Score', fontsize=12, fontweight='bold')
    ax.set_title('Relationship: Effect Size vs Separability',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add quadrant lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / 'separability_vs_effect_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'separability_vs_effect_size.png'}")
    plt.close()

    # 5. Intra-class consistency comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df['Normal-Normal Mean'], width,
           label='Normal-to-Normal', color='green', alpha=0.7, yerr=df['Normal-Normal Std'])
    ax.bar(x + width/2, df['Defect-Defect Mean'], width,
           label='Defect-to-Defect', color='red', alpha=0.7, yerr=df['Defect-Defect Std'])

    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('Intra-Class Similarity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Category'], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.6, 1.0])

    plt.tight_layout()
    plt.savefig(save_dir / 'intra_class_similarity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'intra_class_similarity_comparison.png'}")
    plt.close()


def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SIMILARITY ANALYSIS - ALL CATEGORIES")
    print("=" * 80)

    print("\n📊 CATEGORY RANKING BY DIFFICULTY (Easiest → Hardest)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Category':<15} {'Separability':<15} {'Cohen\'s d':<15} {'Difficulty':<15}")
    print("-" * 80)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        sep = row['Separability']
        cohens = row['Cohen\'s d']

        if sep > 1.0 and cohens > 1.2:
            difficulty = "Very Easy"
        elif sep > 0.5 and cohens > 0.8:
            difficulty = "Easy"
        elif sep > 0.3:
            difficulty = "Moderate"
        else:
            difficulty = "Hard"

        print(f"{idx:<6} {row['Category']:<15} {sep:<15.4f} {cohens:<15.4f} {difficulty:<15}")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nSeparability Scores:")
    print(f"  Mean: {df['Separability'].mean():.4f}")
    print(f"  Std:  {df['Separability'].std():.4f}")
    print(f"  Min:  {df['Separability'].min():.4f} ({df.loc[df['Separability'].idxmin(), 'Category']})")
    print(f"  Max:  {df['Separability'].max():.4f} ({df.loc[df['Separability'].idxmax(), 'Category']})")

    print(f"\nCohen's d:")
    print(f"  Mean: {df['Cohen\'s d'].mean():.4f}")
    print(f"  Std:  {df['Cohen\'s d'].std():.4f}")
    print(f"  Min:  {df['Cohen\'s d'].min():.4f} ({df.loc[df['Cohen\'s d'].idxmin(), 'Category']})")
    print(f"  Max:  {df['Cohen\'s d'].max():.4f} ({df.loc[df['Cohen\'s d'].idxmax(), 'Category']})")

    print("\n" + "=" * 80)
    print("TOP 5 EASIEST CATEGORIES (Best for Zero-Shot)")
    print("=" * 80)
    for idx, (_, row) in enumerate(df.head(5).iterrows(), 1):
        print(f"{idx}. {row['Category']:<15} - Separability: {row['Separability']:.4f}, Cohen's d: {row['Cohen\'s d']:.4f}")

    print("\n" + "=" * 80)
    print("TOP 5 HARDEST CATEGORIES (May Need Few-Shot)")
    print("=" * 80)
    for idx, (_, row) in enumerate(df.tail(5).iterrows(), 1):
        print(f"{idx}. {row['Category']:<15} - Separability: {row['Separability']:.4f}, Cohen's d: {row['Cohen\'s d']:.4f}")

    print("\n" + "=" * 80)


def main():
    print("Loading results from all categories...")
    all_results = load_all_results("../results")

    print(f"Found results for {len(all_results)} categories\n")

    # Create comparison dataframe
    df = create_comparison_dataframe(all_results)

    # Print statistics
    print_summary_statistics(df)

    # Save to CSV
    output_file = Path("../results/all_categories_comparison.csv")
    df.to_csv(output_file, index=False)
    print(f"\n📁 Saved comparison table to: {output_file}")

    # Create visualizations
    print("\n🎨 Generating comparison visualizations...")
    visualize_category_comparison(df, "../results/category_comparison")

    print("\n✅ Analysis complete!")
    print(f"📁 All results saved to: ../results/category_comparison/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
