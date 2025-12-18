"""Analyze errors to understand why excluded patients are classified as included."""
import sys
import os

sys.path.insert(0, 'c:/other/defy')
os.environ['MODEL_TYPE'] = 'sapbert'

from src.excel_handler import load_excel
from src.data_preprocessor import preprocess_excel
from src.nlp_analyzer import MedicalNLPAnalyzer

def analyze_errors():
    df = load_excel('c:/other/defy/data/uploads/Batch1withGroundTruth.xlsx')
    patients = preprocess_excel(df)
    analyzer = MedicalNLPAnalyzer()

    excluded_as_included = []

    for patient in patients:
        result = analyzer.evaluate_patient(patient)
        gt = result['ground_truth'].lower() if result['ground_truth'] else ''
        pred = result['predicted_status']

        # Focus on excluded->included errors
        if 'exclude' in gt and pred == 'included':
            excluded_as_included.append({
                'patient_id': result['patient_id'],
                'ground_truth': gt,
                'predicted': pred,
                'summary': result['summary'],
                'exclusion_results': result['exclusion_results']
            })

    print(f"\n{'='*80}")
    print(f"ANALYSIS: {len(excluded_as_included)} patients should be EXCLUDED but classified as INCLUDED")
    print(f"{'='*80}\n")

    for i, err in enumerate(excluded_as_included[:10]):  # First 10 errors
        print(f"\n--- Patient {i+1}: {err['patient_id']} ---")
        print(f"Ground truth: {err['ground_truth']}")
        print(f"Predicted: {err['predicted']}")
        print(f"Summary: {err['summary']}")
        print(f"\nExclusion criteria analysis:")

        for exc in err['exclusion_results']:
            status = "MET" if exc['is_met'] else ("UNKNOWN" if exc['is_met'] is None else "NOT MET")
            print(f"  [{status}] score={exc['score']:.3f} neg={exc['has_negation']}")
            print(f"       Criterion: {exc['criterion'][:80]}...")
            print(f"       Matched: {exc['matched_sentence'][:80]}...")
            print()

if __name__ == "__main__":
    analyze_errors()
