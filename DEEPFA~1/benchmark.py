"""
Deepfake Detector Benchmark
============================
Companion technical project to the research proposal:
"Investigating the Impact of Deepfake Technology on Digital Media Trust
and the Role of Detection Tools" (Madhurima Mani, University of West London)

This implements the "Detection Benchmarking Methodology" section of the
proposal: it compares pretrained deepfake detectors on accuracy, false
positive rate (FPR), and inference latency, following the same evaluation
axes used in the DFDC benchmark (Dolhansky et al., 2020).

HOW TO RUN
----------
This script needs to download models and a dataset from Hugging Face, so
run it somewhere with normal internet access -- Google Colab (free, no
setup) or your own machine.

1. Open a new Google Colab notebook (colab.research.google.com)
2. First cell:  !pip install -q transformers datasets torch pillow scikit-learn
3. Upload this file or paste its contents into a cell
4. Run it. First run will print the dataset's feature/label names --
   check that output before trusting the results (see NOTE below).

NOTE ON LABEL MAPPING
----------------------
Hugging Face deepfake datasets don't use a standard label convention --
some use 0=real/1=fake, others 0=fake/1=real, others string labels like
"Fake"/"Real". This script inspects `ds.features["label"].names` at
runtime and prints it so you can confirm the mapping is correct for
whichever dataset you point it at, rather than assuming silently.
"""

import csv
import time
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Pretrained deepfake detectors compared in this benchmark (all public on
# Hugging Face as of Aug 2026, image-classification task, real vs fake).
MODELS = [
    "dima806/deepfake_vs_real_image_detection",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "prithivMLmods/Deepfake-Detect-Siglip2",
]

# Sample dataset. Swap this for a Kaggle source via load_from_kaggle() below
# if you'd rather use manjilkarki/deepfake-and-real-images or 140k-real-and-
# fake-faces (both referenced in the research proposal's dataset options).
DATASET_NAME = "Hemg/deepfake-and-real-images"
DATASET_SPLIT = "test"   # fall back to "train" in the except block below
SAMPLE_SIZE = 200        # start small to sanity-check, then raise (e.g. 1000+)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample():
    """Load a small labeled sample and normalise to (image, is_fake) pairs."""
    try:
        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    except (ValueError, KeyError):
        ds = load_dataset(DATASET_NAME, split="train")

    print("Dataset features:", ds.features)

    label_names = getattr(ds.features.get("label"), "names", None)
    print("Label names:", label_names)
    fake_index = label_names.index("fake") if label_names and "fake" in [n.lower() for n in label_names] else 1

    ds = ds.shuffle(seed=42).select(range(min(SAMPLE_SIZE, len(ds))))
    return [(row["image"], row["label"] == fake_index) for row in ds]


def load_from_kaggle(kaggle_dataset="manjilkarki/deepfake-and-real-images"):
    """Alternative loader pulling from Kaggle instead of the HF Hub.
    Requires `pip install kagglehub` and a Kaggle account (kaggle.json API
    token placed in ~/.kaggle/)."""
    import os
    import random

    import kagglehub
    from PIL import Image

    path = kagglehub.dataset_download(kaggle_dataset)
    real_dir = os.path.join(path, "Dataset", "Test", "Real")
    fake_dir = os.path.join(path, "Dataset", "Test", "Fake")

    samples = []
    for d, is_fake in [(real_dir, False), (fake_dir, True)]:
        for fname in os.listdir(d)[: SAMPLE_SIZE // 2]:
            samples.append((Image.open(os.path.join(d, fname)).convert("RGB"), is_fake))
    random.shuffle(samples)
    return samples

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model_name: str
    accuracy: float = 0.0
    false_positive_rate: float = 0.0
    avg_latency_ms: float = 0.0
    n_samples: int = 0
    confusion: list = field(default_factory=list)


def label_is_fake(pred_label: str) -> bool:
    """Normalise each model's own label vocabulary to a real/fake bool."""
    fake_terms = {"fake", "deepfake", "ai", "synthetic", "generated"}
    return any(term in pred_label.lower() for term in fake_terms)


def benchmark_model(model_name: str, samples) -> ModelResult:
    clf = pipeline("image-classification", model=model_name)

    y_true, y_pred, latencies = [], [], []
    for image, is_fake in samples:
        start = time.perf_counter()
        top_pred = clf(image)[0]
        latencies.append((time.perf_counter() - start) * 1000)

        y_true.append(int(is_fake))
        y_pred.append(int(label_is_fake(top_pred["label"])))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return ModelResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        false_positive_rate=fpr,
        avg_latency_ms=sum(latencies) / len(latencies),
        n_samples=len(samples),
        confusion=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )


def main():
    samples = load_sample()
    results = [benchmark_model(m, samples) for m in MODELS]

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "false_positive_rate", "avg_latency_ms", "n_samples"])
        for r in results:
            writer.writerow([
                r.model_name,
                f"{r.accuracy:.4f}",
                f"{r.false_positive_rate:.4f}",
                f"{r.avg_latency_ms:.2f}",
                r.n_samples,
            ])
            print(
                f"{r.model_name:45s} acc={r.accuracy:.3f}  "
                f"FPR={r.false_positive_rate:.3f}  "
                f"latency={r.avg_latency_ms:.1f}ms  confusion={r.confusion}"
            )

    print("\nSaved results.csv -- paste this table into README.md's Results section.")


if __name__ == "__main__":
    main()
