"""
AVeriTeC Benchmark Dataset Loader for TrustNet Evaluation Suite.

Parses dev.json and train.json benchmark datasets and standardizes ground truth labels
for quantitative accuracy evaluation.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Standardized 4-class label constants
LABEL_SUPPORTED = "Supported"
LABEL_REFUTED = "Refuted"
LABEL_NOT_ENOUGH = "Not Enough Evidence"
LABEL_CONFLICTING = "Conflicting Evidence / Cherrypicking"

# Mapping raw dataset label variants to standardized labels
LABEL_MAP: Dict[str, str] = {
    "supported": LABEL_SUPPORTED,
    "refuted": LABEL_REFUTED,
    "not enough evidence": LABEL_NOT_ENOUGH,
    "conflicting evidence/cherrypicking": LABEL_CONFLICTING,
    "conflicting evidence / cherrypicking": LABEL_CONFLICTING,
    "insufficient evidence": LABEL_NOT_ENOUGH,
}


@dataclass
class BenchmarkSample:
    """Represents a single fact-checking sample from AVeriTeC dataset."""
    sample_id: int
    claim: str
    label: str
    justification: Optional[str] = None
    questions: Optional[List[Dict[str, Any]]] = None
    speaker: Optional[str] = None
    claim_date: Optional[str] = None


def normalize_label(raw_label: str) -> str:
    """Standardizes label strings into 4 canonical benchmark classes."""
    if not raw_label:
        return LABEL_NOT_ENOUGH
    cleaned = raw_label.strip().lower()
    return LABEL_MAP.get(cleaned, raw_label.strip())


def load_averitec_dataset(
    split: str = "dev",
    max_samples: Optional[int] = None,
    dataset_dir: Optional[str] = None
) -> List[BenchmarkSample]:
    """
    Loads and parses AVeriTeC dataset samples.

    Args:
        split: 'dev' or 'train'
        max_samples: Optional limit on number of samples to load
        dataset_dir: Directory containing averitec json files (defaults to relative path)

    Returns:
        List of BenchmarkSample instances
    """
    if dataset_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, "datasets", "averitec")

    file_name = "dev.json" if split.lower() == "dev" else "train.json"
    file_path = os.path.join(dataset_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"AVeriTeC dataset file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[BenchmarkSample] = []
    for idx, item in enumerate(data):
        if max_samples and idx >= max_samples:
            break

        norm_label = normalize_label(item.get("label", ""))
        sample = BenchmarkSample(
            sample_id=idx + 1,
            claim=item.get("claim", "").strip(),
            label=norm_label,
            justification=item.get("justification"),
            questions=item.get("questions", []),
            speaker=item.get("speaker"),
            claim_date=item.get("claim_date"),
        )
        samples.append(sample)

    return samples
