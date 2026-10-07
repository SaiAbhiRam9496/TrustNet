"""
TrustNet Benchmark Evaluation Suite.

Evaluates TrustNet's fact-checking pipeline on AVeriTeC benchmark dataset samples
and calculates Precision, Recall, Per-Class F1, and Overall Macro-F1 metrics.
"""

import sys
import os
import argparse
import logging
import time
from typing import List, Dict, Any, Tuple
import httpx

# Ensure backend and eval imports work correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from eval.loader import (
    load_averitec_dataset,
    normalize_label,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_NOT_ENOUGH,
    LABEL_CONFLICTING,
    BenchmarkSample
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval.run_benchmark")

CLASSES = [
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_NOT_ENOUGH,
    LABEL_CONFLICTING,
]


def compute_metrics(
    predictions: List[str], ground_truths: List[str]
) -> Dict[str, Any]:
    """
    Computes Precision, Recall, F1 score per class and overall Macro-F1 & Accuracy.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths list lengths must match.")

    n_samples = len(predictions)
    if n_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    correct_total = 0
    per_class_stats: Dict[str, Dict[str, float]] = {}

    for c in CLASSES:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g == c)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g != c)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != c and g == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_stats[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in ground_truths if g == c)
        }

    correct_total = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = round(correct_total / n_samples, 4)
    macro_f1 = round(sum(stats["f1"] for stats in per_class_stats.values()) / len(CLASSES), 4)

    return {
        "sample_count": n_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class_stats
    }


def evaluate_via_api(
    samples: List[BenchmarkSample],
    api_url: str = "http://127.0.0.1:8000/api/v1/analyze",
    timeout: float = 300.0,
    delay: float = 5.0,
) -> Tuple[List[str], List[str], List[float]]:
    """Evaluates samples by sending POST requests to the active TrustNet API server."""
    predictions = []
    ground_truths = []
    latencies = []

    logger.info(f"Starting API evaluation on {len(samples)} samples via {api_url} (delay={delay}s between samples)...")

    with httpx.Client(timeout=timeout) as client:
        for idx, sample in enumerate(samples, start=1):
            logger.info(f"[{idx}/{len(samples)}] Claim: '{sample.claim[:60]}...' | GT: {sample.label}")
            start_time = time.time()
            try:
                response = client.post(api_url, json={"claim": sample.claim})
                elapsed = time.time() - start_time
                latencies.append(elapsed)

                if response.status_code == 200:
                    res_json = response.json()
                    raw_verdict = res_json.get("verdict_label") or res_json.get("verdict", "")
                    pred_label = normalize_label(raw_verdict)
                    logger.info(f" -> Pred: {pred_label} (verdict: {raw_verdict}) | Time: {elapsed:.2f}s")
                else:
                    logger.warning(f" -> HTTP {response.status_code}: {response.text}")
                    pred_label = LABEL_NOT_ENOUGH

            except Exception as e:
                elapsed = time.time() - start_time
                latencies.append(elapsed)
                logger.error(f" -> Request failed: {e}")
                pred_label = LABEL_NOT_ENOUGH

            predictions.append(pred_label)
            ground_truths.append(sample.label)

            if delay > 0 and idx < len(samples):
                logger.info(f" Sleeping {delay} seconds to prevent API rate-limit exhaustion...")
                time.sleep(delay)

    return predictions, ground_truths, latencies


def print_evaluation_report(results: Dict[str, Any], avg_latency: float = 0.0) -> None:
    """Formats and prints the quantitative evaluation results table."""
    print("\n" + "=" * 75)
    print(f"📊 TRUSTNET BENCHMARK EVALUATION REPORT")
    print("=" * 75)
    print(f"Total Samples Evaluated : {results['sample_count']}")
    print(f"Overall Accuracy         : {results['accuracy'] * 100:.2f}%")
    print(f"Overall Macro-F1 Score   : {results['macro_f1']:.4f}")
    if avg_latency > 0:
        print(f"Average Request Latency  : {avg_latency:.2f} seconds")
    print("-" * 75)
    print(f"{'Class Label':<36} | {'Prec':<7} | {'Recall':<7} | {'F1-Score':<8} | {'Support'}")
    print("-" * 75)
    for c, stats in results["per_class"].items():
        print(
            f"{c:<36} | {stats['precision']:<7.4f} | {stats['recall']:<7.4f} | {stats['f1']:<8.4f} | {stats['support']}"
        )
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="TrustNet AVeriTeC Benchmark Evaluation")
    parser.add_argument("--split", choices=["dev", "train"], default="dev", help="Dataset split to evaluate")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum samples to evaluate")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api/v1/analyze", help="TrustNet API endpoint")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds between sample requests")
    args = parser.parse_args()

    logger.info(f"Loading {args.split} dataset (max samples={args.max_samples})...")
    samples = load_averitec_dataset(split=args.split, max_samples=args.max_samples)
    logger.info(f"Loaded {len(samples)} samples for evaluation.")

    predictions, ground_truths, latencies = evaluate_via_api(
        samples, api_url=args.api_url, delay=args.delay
    )
    results = compute_metrics(predictions, ground_truths)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print_evaluation_report(results, avg_latency=avg_latency)


if __name__ == "__main__":
    main()
