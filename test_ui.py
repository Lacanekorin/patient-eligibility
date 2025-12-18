"""Quick test of the updated UI data."""
import sys
import os
sys.path.insert(0, 'c:/other/defy')
os.environ['MODEL_TYPE'] = 'sapbert'

from src.excel_handler import load_excel
from src.data_preprocessor import preprocess_excel
from src.nlp_analyzer import get_analyzer

df = load_excel('c:/other/defy/data/uploads/Batch1withGroundTruth.xlsx')
print(f'Loaded {len(df)} rows')

# Process first 5 patients only for quick test
patients = preprocess_excel(df)[:5]
print(f'Preprocessing done: {len(patients)} patients')

analyzer = get_analyzer()

def format_reasoning(result):
    """Формирует краткое обоснование решения для UI."""
    reasons = []
    inclusion_met = [r for r in result['inclusion_results'] if r['is_met'] is True]
    inclusion_not_met = [r for r in result['inclusion_results'] if r['is_met'] is False]
    exclusion_met = [r for r in result['exclusion_results'] if r['is_met'] is True]

    if result['predicted_status'] == 'included':
        if inclusion_met:
            top_criteria = inclusion_met[:2]
            for c in top_criteria:
                short_criterion = c['criterion'][:50] + '...' if len(c['criterion']) > 50 else c['criterion']
                reasons.append(f"+ {short_criterion} (score: {c['score']:.2f})")
        reasons.append(f"No exclusion criteria met")
    elif result['predicted_status'] == 'excluded':
        if exclusion_met:
            for c in exclusion_met[:2]:
                short_criterion = c['criterion'][:50] + '...' if len(c['criterion']) > 50 else c['criterion']
                neg_mark = " [negated]" if c.get('has_negation') else ""
                reasons.append(f"- EXCLUSION: {short_criterion}{neg_mark} (score: {c['score']:.2f})")
        if inclusion_not_met:
            for c in inclusion_not_met[:2]:
                short_criterion = c['criterion'][:50] + '...' if len(c['criterion']) > 50 else c['criterion']
                reasons.append(f"- NOT MET: {short_criterion} (score: {c['score']:.2f})")
    else:
        reasons.append("Could not determine eligibility from available data")
        unknown_count = sum(1 for r in result['inclusion_results'] if r['is_met'] is None)
        if unknown_count:
            reasons.append(f"{unknown_count} inclusion criteria unknown")
    return reasons


def get_key_evidence(result):
    """Извлекает ключевые доказательства из результатов."""
    evidence = []
    all_results = result['inclusion_results'] + result['exclusion_results']
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    for r in sorted_results[:3]:
        if r['score'] >= 0.4 and r.get('matched_sentence'):
            evidence.append({
                'criterion': r['criterion'][:60] + '...' if len(r['criterion']) > 60 else r['criterion'],
                'matched_text': r['matched_sentence'][:100] + '...' if len(r['matched_sentence']) > 100 else r['matched_sentence'],
                'score': r['score'],
                'is_met': r['is_met'],
                'type': r.get('type', 'unknown'),
                'has_negation': r.get('has_negation', False)
            })
    return evidence


print('\n=== Sample Results ===')
for patient in patients:
    result = analyzer.evaluate_patient(patient)
    reasoning = format_reasoning(result)
    evidence = get_key_evidence(result)

    print(f'\nPatient: {result["patient_id"]}')
    print(f'Status: {result["predicted_status"]}')
    print(f'GT: {result["ground_truth"]}')
    print(f'Correct: {result["is_correct"]}')
    print('Reasoning:')
    for r in reasoning:
        print(f'  {r}')
    if evidence:
        print('Evidence:')
        for e in evidence[:2]:
            crit = e["criterion"][:40]
            print(f'  - {crit}... (score: {e["score"]:.2f}, type: {e["type"]})')
