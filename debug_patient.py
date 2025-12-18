"""Debug specific patient to understand exclusion reason."""
import sys
import os

sys.path.insert(0, 'c:/other/defy')
os.environ['MODEL_TYPE'] = 'sapbert'

from src.excel_handler import load_excel
from src.data_preprocessor import preprocess_excel

df = load_excel('c:/other/defy/data/uploads/Batch1withGroundTruth.xlsx')
patients = preprocess_excel(df)

# Find patients that should be excluded but are classified as included
target_patients = ['dapa-003', 'dapa-018', 'dapa-032']
for p in patients:
    if p.patient_id in target_patients:
        print(f"\n{'='*60}")
        print(f"Patient ID: {p.patient_id}")
        print(f"Ground truth: {p.ground_truth}")
        print(f"\n=== NOTE ===")
        print(p.note)
        print(f"\n=== INCLUSION CRITERIA ===")
        for c in p.inclusion_criteria:
            print(f"- {c.text[:80]}...")
        print()
