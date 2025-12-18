"""Quick accuracy test script for different approaches."""
import sys
import os
import time

# Parse command-line argument for model type
model_type = 'sapbert'
if len(sys.argv) > 1:
    model_type = sys.argv[1]

sys.path.insert(0, 'c:/other/defy')
os.environ['MODEL_TYPE'] = model_type

from src.excel_handler import load_excel
from src.data_preprocessor import preprocess_excel
from src.nlp_analyzer import MedicalNLPAnalyzer

def test_accuracy(analyzer=None):
    """Run accuracy test and return results."""
    df = load_excel('c:/other/defy/data/uploads/Batch1withGroundTruth.xlsx')
    patients = preprocess_excel(df)

    if analyzer is None:
        analyzer = MedicalNLPAnalyzer()

    correct = 0
    total = 0
    errors = {'included->excluded': 0, 'excluded->included': 0, 'other': 0}

    start_time = time.time()

    for patient in patients:
        result = analyzer.evaluate_patient(patient)
        gt = result['ground_truth'].lower() if result['ground_truth'] else ''
        pred = result['predicted_status']

        if gt and gt != 'n/a':
            total += 1
            if result['is_correct']:
                correct += 1
            else:
                if 'include' in gt and pred == 'excluded':
                    errors['included->excluded'] += 1
                elif 'exclude' in gt and pred == 'included':
                    errors['excluded->included'] += 1
                else:
                    errors['other'] += 1

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'errors': errors,
        'time': elapsed
    }

def print_results(results, name="Test"):
    """Print formatted results."""
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")
    print(f"Time: {results['time']:.1f}s")
    print(f"Errors:")
    print(f"  included->excluded: {results['errors']['included->excluded']}")
    print(f"  excluded->included: {results['errors']['excluded->included']}")
    print(f"  other: {results['errors']['other']}")

if __name__ == "__main__":
    results = test_accuracy()
    print_results(results, f"MODEL_TYPE={model_type}")
