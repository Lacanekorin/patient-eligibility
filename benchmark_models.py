"""
Benchmark script for comparing medical NLP models.
Runs locally without Docker for faster iteration.

Usage:
    c:\other\defy\venv\Scripts\python.exe benchmark_models.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.excel_handler import load_excel
from src.data_preprocessor import preprocess_excel
from src.nlp_analyzer import get_model_path, MedicalNLPAnalyzer

# Models to benchmark
MODELS = [
    "pubmedbert",
    "pubmedbert-neuml",
    "sapbert",
    "bioclinicalbert",
    "clinicalbert",
    "biomedbert",
    "mpnet",
    "minilm",
]

# Test file
TEST_FILE = "data/uploads/Batch1withGroundTruth.xlsx"


def run_benchmark(model_type: str, patients: list) -> dict:
    """Run benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_type}")
    print(f"{'='*60}")

    start_load = time.time()

    # Load model
    model_path = get_model_path(model_type)
    print(f"Model path: {model_path}")

    try:
        analyzer = MedicalNLPAnalyzer(model_path)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return {
            "model": model_type,
            "error": str(e),
            "accuracy": None,
            "load_time": None,
            "inference_time": None,
        }

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    # Run inference
    print(f"Processing {len(patients)} patients...")
    start_inference = time.time()

    correct = 0
    total_with_gt = 0
    errors_detail = {"included->excluded": 0, "excluded->included": 0, "other": 0}

    for i, patient in enumerate(patients):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(patients)}")

        result = analyzer.evaluate_patient(patient)

        if result["ground_truth"] != "N/A":
            total_with_gt += 1
            if result["is_correct"]:
                correct += 1
            else:
                # Track error type
                gt = result["ground_truth"].lower()
                pred = result["predicted_status"].lower()
                if "include" in gt and "exclude" in pred:
                    errors_detail["included->excluded"] += 1
                elif "exclude" in gt and "include" in pred:
                    errors_detail["excluded->included"] += 1
                else:
                    errors_detail["other"] += 1

    inference_time = time.time() - start_inference
    accuracy = (correct / total_with_gt * 100) if total_with_gt > 0 else 0

    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total_with_gt})")
    print(f"Inference time: {inference_time:.1f}s")
    print(f"Errors: {errors_detail}")

    # Free memory
    del analyzer

    return {
        "model": model_type,
        "accuracy": accuracy,
        "correct": correct,
        "total": total_with_gt,
        "load_time": load_time,
        "inference_time": inference_time,
        "errors": errors_detail,
        "error": None,
    }


def main():
    print("=" * 60)
    print("Medical NLP Models Benchmark")
    print("=" * 60)

    # Load test data once
    print(f"\nLoading test data: {TEST_FILE}")
    df = load_excel(TEST_FILE)
    patients = preprocess_excel(df)
    print(f"Loaded {len(patients)} patients")

    # Run benchmarks
    results = []
    for model in MODELS:
        try:
            result = run_benchmark(model, patients)
            results.append(result)
        except Exception as e:
            print(f"ERROR with {model}: {e}")
            results.append({
                "model": model,
                "error": str(e),
                "accuracy": None,
            })

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Load(s)':>10} {'Infer(s)':>10}")
    print("-" * 60)

    # Sort by accuracy
    sorted_results = sorted(
        [r for r in results if r.get("accuracy") is not None],
        key=lambda x: x["accuracy"],
        reverse=True,
    )

    for r in sorted_results:
        print(
            f"{r['model']:<20} {r['accuracy']:>9.1f}% {r.get('load_time', 0):>10.1f} {r.get('inference_time', 0):>10.1f}"
        )

    # Print errors
    for r in results:
        if r.get("error"):
            print(f"{r['model']:<20} ERROR: {r['error']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/results/benchmark_{timestamp}.txt"

    with open(results_file, "w") as f:
        f.write("Medical NLP Models Benchmark Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test file: {TEST_FILE}\n")
        f.write(f"Patients: {len(patients)}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':>10} {'Load(s)':>10} {'Infer(s)':>10}\n")
        f.write("-" * 60 + "\n")

        for r in sorted_results:
            f.write(
                f"{r['model']:<20} {r['accuracy']:>9.1f}% {r.get('load_time', 0):>10.1f} {r.get('inference_time', 0):>10.1f}\n"
            )

        f.write("\n" + "=" * 60 + "\n")
        f.write("Error Analysis:\n")
        for r in sorted_results:
            if r.get("errors"):
                f.write(f"\n{r['model']}:\n")
                f.write(f"  included->excluded: {r['errors']['included->excluded']}\n")
                f.write(f"  excluded->included: {r['errors']['excluded->included']}\n")
                f.write(f"  other: {r['errors']['other']}\n")

    print(f"\nResults saved to: {results_file}")

    # Return best model
    if sorted_results:
        best = sorted_results[0]
        print(f"\nBEST MODEL: {best['model']} with {best['accuracy']:.1f}% accuracy")


if __name__ == "__main__":
    main()
