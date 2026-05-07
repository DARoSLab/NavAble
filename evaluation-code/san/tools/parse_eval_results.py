#!/usr/bin/env python3
"""
Parse eval metrics.json files and summarize Stage-1 results.
Generates markdown tables for real-data-results.md.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re


def parse_metrics_file(path: Path) -> Dict:
    """Load and parse a metrics.json file."""
    with open(path) as f:
        return json.load(f)


def extract_model_domain_from_dir(work_dir_path: str) -> Tuple[str, str]:
    """
    Extract model and domain from work_dirs subdirectory structure.

    Examples:
    - eval_split/mask2former_real_v2_only_*  → ('mask2former', 'real_v2')
    - eval_split/segformer_opensrc_only_*    → ('segformer', 'opensrc')
    - eval_combined_test/mask2former_combined_test_* → ('mask2former', 'combined_test')
    """
    dirname = Path(work_dir_path).name

    # Try split-domain pattern: {model}_{domain}_only_*
    match = re.search(r'(mask2former|segformer|san)_(real_v2|opensrc)_only', dirname)
    if match:
        return match.group(1), match.group(2)

    # Try combined-test pattern: {model}_combined_test_*
    match = re.search(r'(mask2former|segformer|san)_combined_test', dirname)
    if match:
        return match.group(1), 'combined_test'

    # Try old combined pattern: {model}_{ts}
    if 'eval_combined_test' in work_dir_path:
        for model in ['mask2former', 'segformer', 'san']:
            if model in dirname:
                return model, 'combined_test'

    return None, None


def print_split_domain_results(eval_dir: Path):
    """Find and print split-domain eval results."""
    split_dirs = sorted(eval_dir.glob('*_only_*'))

    if not split_dirs:
        print("No split-domain results found.")
        return

    print("\n## Split-Domain Test Results (with turnstile excluded)")
    print("\n### Overall Metrics")
    print("| Domain | SegFormer mIoU | Mask2Former mIoU | SAN mIoU |")
    print("|---|---|---|---|")

    domain_metrics = {}

    for split_dir in split_dirs:
        metrics_file = split_dir / 'metrics.json'
        if not metrics_file.exists():
            continue

        metrics = parse_metrics_file(metrics_file)
        model, domain = extract_model_domain_from_dir(str(split_dir))

        if not model or domain not in ['real_v2', 'opensrc']:
            continue

        if domain not in domain_metrics:
            domain_metrics[domain] = {}

        # Extract mIoU (should already have turnstile excluded via config)
        miou = metrics.get('mIoU', None)
        domain_metrics[domain][model] = miou

    for domain in ['real_v2', 'opensrc']:
        if domain in domain_metrics:
            seg = domain_metrics[domain].get('segformer', 'N/A')
            m2f = domain_metrics[domain].get('mask2former', 'N/A')
            san = domain_metrics[domain].get('san', 'N/A')

            if isinstance(seg, (int, float)):
                seg = f"{seg:.2f}"
            if isinstance(m2f, (int, float)):
                m2f = f"{m2f:.2f}"
            if isinstance(san, (int, float)):
                san = f"{san:.2f}"

            print(f"| **{domain}** | {seg} | {m2f} | {san} |")


def print_combined_test_results(eval_dir: Path):
    """Find and print combined-test results."""
    combined_dirs = sorted(eval_dir.glob('*combined_test*'))

    if not combined_dirs:
        print("No combined-test results found.")
        return

    print("\n## Combined Test Results (real_v2/test + opensrc/test, turnstile excluded)")
    print("\n| Metric | SegFormer | Mask2Former | SAN |")
    print("|---|---|---|---|")

    combined_metrics = {}

    for combined_dir in combined_dirs:
        metrics_file = combined_dir / 'metrics.json'
        if not metrics_file.exists():
            continue

        metrics = parse_metrics_file(metrics_file)
        model, domain = extract_model_domain_from_dir(str(combined_dir))

        if not model or domain != 'combined_test':
            continue

        combined_metrics[model] = metrics

    # Extract main metrics
    metrics_to_show = ['mIoU', 'mAP50-95', 'Precision', 'Recall']

    for metric_name in metrics_to_show:
        seg = combined_metrics.get('segformer', {}).get(metric_name, 'N/A')
        m2f = combined_metrics.get('mask2former', {}).get(metric_name, 'N/A')
        san = combined_metrics.get('san', {}).get(metric_name, 'N/A')

        if isinstance(seg, (int, float)):
            seg = f"{seg:.2f}"
        if isinstance(m2f, (int, float)):
            m2f = f"{m2f:.2f}"
        if isinstance(san, (int, float)):
            san = f"{san:.2f}"

        print(f"| {metric_name} | {seg} | {m2f} | {san} |")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    eval_dir = project_root / 'work_dirs'

    print("=" * 80)
    print("Stage-1 Evaluation Results Summary")
    print("=" * 80)

    print_split_domain_results(eval_dir)
    print_combined_test_results(eval_dir)

    print("\n" + "=" * 80)
    print("Done. Copy and paste the above into real-data-results.md")
    print("=" * 80)


if __name__ == '__main__':
    main()
