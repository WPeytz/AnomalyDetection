"""
Run "Prompt (full)" experiments across all categories.

This script trains visual prompts using ALL available defect samples
(rather than just 5-shot) to replicate the "Prompt (full)" results.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json


# Categories from the paper's table
CATEGORIES = ['bottle', 'carpet', 'cable', 'screw', 'hazelnut']


EXPECTED_RESULTS = {
    'bottle': 1.000,
    'carpet': 0.996,
    'cable': 0.936,
    'screw': 0.984,
    'hazelnut': 0.901,
}


def count_defect_samples(root_dir: str, category: str) -> int:
    """Count total defect samples available for a category."""
    test_path = Path(root_dir) / category / 'test'
    if not test_path.exists():
        
        return 0

    count = 0
    for defect_folder in test_path.iterdir():
        if defect_folder.is_dir() and defect_folder.name != 'good':
            count += len(list(defect_folder.glob('*.png')))
    return count


def run_training(category: str, num_defect_samples: int, args):
    """Run training for a single category."""
    cmd = [
        sys.executable,
        'prompt_based_feature_adaption/train_prompts.py',
        '--category', category,
        '--root-dir', args.root_dir,
        '--num-defect-samples', str(num_defect_samples),
        '--num-epochs', str(args.num_epochs),
        '--modulation-type', args.modulation_type,
        '--scaling-factor', str(args.scaling_factor),
        '--output-dir', args.output_dir,
    ]

    print(f"\n{'='*70}")
    print(f"Training: {category} (using {num_defect_samples} defect samples)")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_evaluation(category: str, args):
    """Run evaluation for a single category."""
    checkpoint_path = Path(args.output_dir) / f"{category}_prompts.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    cmd = [
        sys.executable,
        'prompt_based_feature_adaption/evaluate_prompts.py',
        '--category', category,
        '--checkpoint', str(checkpoint_path),
        '--root-dir', args.root_dir,
        '--output-dir', args.results_dir,
    ]

    print(f"\n{'='*70}")
    print(f"Evaluating: {category}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        results_path = Path(args.results_dir) / f"{category}_comparison.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run Prompt (full) experiments")
    parser.add_argument("--root-dir", type=str, default="mvtec_ad",
                       help="Root directory of MVTec AD dataset")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--modulation-type", type=str, default="per_patch",
                       choices=['per_patch', 'global', 'learned_transform'],
                       help="Type of prompt modulation")
    parser.add_argument("--scaling-factor", type=float, default=0.1,
                       help="Scaling factor for prompt influence")
    parser.add_argument("--output-dir", type=str,
                       default="prompt_based_feature_adaption/checkpoints/prompt_full",
                       help="Output directory for checkpoints")
    parser.add_argument("--results-dir", type=str,
                       default="prompt_based_feature_adaption/results/prompt_full",
                       help="Output directory for results")
    parser.add_argument("--categories", type=str, nargs='+', default=CATEGORIES,
                       help="Categories to run experiments on")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation (skip training)")

    args = parser.parse_args()

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PROMPT (FULL) EXPERIMENTS")
    print("="*70)
    print(f"\nCategories: {args.categories}")
    print(f"Modulation type: {args.modulation_type}")
    print(f"Scaling factor: {args.scaling_factor}")
    print(f"Epochs: {args.num_epochs}")

    # Count defect samples for each category
    print("\nDefect samples per category:")
    defect_counts = {}
    for cat in args.categories:
        count = count_defect_samples(args.root_dir, cat)
        defect_counts[cat] = count
        print(f"  {cat}: {count}")

    results = {}

    # Training phase
    if not args.eval_only:
        print("\n" + "="*70)
        print("TRAINING PHASE")
        print("="*70)

        for cat in args.categories:
            num_samples = defect_counts[cat]
            if num_samples == 0:
                print(f"Skipping {cat}: no defect samples found")
                continue

            success = run_training(cat, num_samples, args)
            if not success:
                print(f"Training failed for {cat}")

    # Evaluation phase
    print("\n" + "="*70)
    print("EVALUATION PHASE")
    print("="*70)

    for cat in args.categories:
        result = run_evaluation(cat, args)
        if result:
            results[cat] = result

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: PROMPT (FULL) RESULTS")
    print("="*70)
    print(f"\n{'Category':<12} {'Our AUROC':<12} {'Expected':<12} {'Diff':<12}")
    print("-"*48)

    for cat in args.categories:
        if cat in results:
            our_auroc = results[cat]['prompt_tuned']['auroc']
            expected = EXPECTED_RESULTS.get(cat, 'N/A')
            if isinstance(expected, float):
                diff = our_auroc - expected
                print(f"{cat:<12} {our_auroc:<12.4f} {expected:<12.3f} {diff:+.4f}")
            else:
                print(f"{cat:<12} {our_auroc:<12.4f} {expected:<12}")
        else:
            print(f"{cat:<12} {'N/A':<12} {EXPECTED_RESULTS.get(cat, 'N/A')}")

    # Save combined results
    combined_results_path = Path(args.results_dir) / "prompt_full_summary.json"
    with open(combined_results_path, 'w') as f:
        json.dump({
            'results': results,
            'expected': EXPECTED_RESULTS,
            'config': {
                'modulation_type': args.modulation_type,
                'scaling_factor': args.scaling_factor,
                'num_epochs': args.num_epochs,
            }
        }, f, indent=2)

    print(f"\nResults saved to {combined_results_path}")


if __name__ == "__main__":
    main()
