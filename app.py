"""Веб-приложение Flask для проверки соответствия пациентов критериям."""
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import pandas as pd

from src.excel_handler import load_excel, save_results
from src.data_preprocessor import (
    preprocess_excel,
    save_preprocessed_data,
    get_preprocessing_stats,
    PatientData
)
from src.nlp_analyzer import get_analyzer

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Конфигурация
UPLOAD_FOLDER = Path('data/uploads')
RESULTS_FOLDER = Path('data/results')
PREPROCESSED_FOLDER = Path('data/preprocessed')
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
PREPROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


def allowed_file(filename: str) -> bool:
    """Проверяет допустимость расширения файла."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_reasoning(result: dict) -> str:
    """Формирует краткое обоснование решения для UI."""
    reasons = []

    # Анализируем inclusion критерии
    inclusion_met = [r for r in result['inclusion_results'] if r['is_met'] is True]
    inclusion_not_met = [r for r in result['inclusion_results'] if r['is_met'] is False]

    # Анализируем exclusion критерии
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

    else:  # not enough information
        reasons.append("Could not determine eligibility from available data")
        unknown_count = sum(1 for r in result['inclusion_results'] if r['is_met'] is None)
        if unknown_count:
            reasons.append(f"{unknown_count} inclusion criteria unknown")

    return reasons


def get_key_evidence(result: dict) -> list:
    """Извлекает ключевые доказательства из результатов."""
    evidence = []

    # Собираем все результаты
    all_results = result['inclusion_results'] + result['exclusion_results']

    # Сортируем: сначала по is_met (True первые), затем по score
    def sort_key(r):
        # Приоритет: met criteria > not met > unknown
        if r['is_met'] is True:
            priority = 0
        elif r['is_met'] is False:
            priority = 1
        else:
            priority = 2
        return (priority, -r['score'])

    sorted_results = sorted(all_results, key=sort_key)

    # Берём все критерии с результатом (не только топ-3)
    for r in sorted_results:
        if r.get('matched_sentence'):
            evidence.append({
                'criterion': r['criterion'],
                'matched_text': r['matched_sentence'],
                'score': r['score'],
                'is_met': r['is_met'],
                'type': r.get('type', 'unknown'),
                'has_negation': r.get('has_negation', False),
                'confidence': r.get('confidence', 'medium')
            })

    return evidence


def process_patients_v2(df: pd.DataFrame) -> dict:
    """
    Двухэтапная обработка данных:
    1. Препроцессинг Excel -> структурированные данные
    2. NLP анализ каждого пациента

    Returns:
        {
            'preprocessing_stats': dict,
            'results': {
                'included': list,
                'excluded': list,
                'not_enough_info': list
            },
            'accuracy': {
                'correct': int,
                'total': int,
                'percentage': float
            },
            'detailed_results': list  # Для UI с обоснованиями
        }
    """
    # Этап 1: Препроцессинг
    print("Stage 1: Preprocessing Excel data...")
    patients = preprocess_excel(df)
    preprocessing_stats = get_preprocessing_stats(patients)

    # Этап 2: NLP анализ
    print("Stage 2: NLP Analysis...")
    analyzer = get_analyzer()

    included = []
    excluded = []
    not_enough_info = []
    detailed_results = []  # Для UI

    correct = 0
    total_with_gt = 0

    for i, patient in enumerate(patients):
        if (i + 1) % 10 == 0:
            print(f"  Processing patient {i + 1}/{len(patients)}...")

        result = analyzer.evaluate_patient(patient)

        # Формируем строку для Excel
        row = {
            'patient_id': result['patient_id'],
            'predicted_status': result['predicted_status'],
            'ground_truth': result['ground_truth'],
            'is_correct': '✓' if result['is_correct'] else '✗' if result['ground_truth'] != 'N/A' else '-',
            'summary': result['summary'],
            'inclusion_details': format_criteria_results(result['inclusion_results']),
            'exclusion_details': format_criteria_results(result['exclusion_results']),
        }

        # Формируем детальный результат для UI
        detailed_row = {
            'patient_id': result['patient_id'],
            'predicted_status': result['predicted_status'],
            'ground_truth': result['ground_truth'],
            'is_correct': result['is_correct'],
            'summary': result['summary'],
            'reasoning': format_reasoning(result),
            'evidence': get_key_evidence(result),
            'inclusion_met': sum(1 for r in result['inclusion_results'] if r['is_met'] is True),
            'inclusion_not_met': sum(1 for r in result['inclusion_results'] if r['is_met'] is False),
            'inclusion_unknown': sum(1 for r in result['inclusion_results'] if r['is_met'] is None),
            'exclusion_met': sum(1 for r in result['exclusion_results'] if r['is_met'] is True),
            'exclusion_not_met': sum(1 for r in result['exclusion_results'] if r['is_met'] is False),
        }
        detailed_results.append(detailed_row)

        # Считаем accuracy
        if result['ground_truth'] != 'N/A':
            total_with_gt += 1
            if result['is_correct']:
                correct += 1

        # Распределяем по категориям
        if result['predicted_status'] == 'included':
            included.append(row)
        elif result['predicted_status'] == 'excluded':
            excluded.append(row)
        else:
            not_enough_info.append(row)

    accuracy = {
        'correct': correct,
        'total': total_with_gt,
        'percentage': round(correct / total_with_gt * 100, 1) if total_with_gt > 0 else 0
    }

    return {
        'preprocessing_stats': preprocessing_stats,
        'results': {
            'included': included,
            'excluded': excluded,
            'not_enough_info': not_enough_info
        },
        'accuracy': accuracy,
        'detailed_results': detailed_results
    }


def format_criteria_results(criteria_results: list) -> str:
    """Форматирует результаты критериев для Excel."""
    lines = []
    for r in criteria_results:
        mark = '+' if r['is_met'] else ('-' if r['is_met'] is False else '?')
        neg = ' [NEG]' if r.get('has_negation') else ''
        lines.append(f"[{mark}] {r['criterion'][:60]}... | Score: {r['score']:.2f}{neg}")
        if r.get('matched_sentence'):
            lines.append(f"    -> \"{r['matched_sentence'][:80]}...\"")
    return '\n'.join(lines)


@app.route('/')
def index():
    """Главная страница с формой загрузки."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загруженного файла."""
    if 'file' not in request.files:
        flash('Файл не выбран')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('Файл не выбран')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)

        try:
            # Загружаем данные
            print(f"Loading Excel file: {filepath}")
            df = load_excel(str(filepath))
            print(f"Loaded {len(df)} rows")

            # Обрабатываем
            results = process_patients_v2(df)

            # Сохраняем результаты в Excel
            result_filename = f'results_{filename}'
            result_path = RESULTS_FOLDER / result_filename

            save_results(
                pd.DataFrame(results['results']['included']),
                pd.DataFrame(results['results']['excluded']),
                pd.DataFrame(results['results']['not_enough_info']),
                str(result_path)
            )

            # Статистика для отображения
            stats = {
                'total': len(df),
                'eligible': len(results['results']['included']),
                'not_eligible': len(results['results']['excluded']),
                'clarification': len(results['results']['not_enough_info']),
                'accuracy': results['accuracy'],
                'preprocessing': results['preprocessing_stats']
            }

            return render_template(
                'results.html',
                stats=stats,
                result_file=result_filename,
                patients=results['detailed_results']
            )

        except Exception as e:
            import traceback
            error_msg = f'Ошибка обработки файла: {str(e)}'
            print(error_msg)
            print(traceback.format_exc())
            flash(error_msg)
            return redirect(url_for('index'))

    flash('Недопустимый формат файла. Используйте .xlsx или .xls')
    return redirect(url_for('index'))


@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание результирующего файла."""
    filepath = RESULTS_FOLDER / secure_filename(filename)
    if filepath.exists():
        return send_file(filepath, as_attachment=True)
    flash('Файл не найден')
    return redirect(url_for('index'))


@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    """API эндпоинт для препроцессинга (возвращает JSON)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)

        df = load_excel(str(filepath))
        patients = preprocess_excel(df)
        stats = get_preprocessing_stats(patients)

        # Сохраняем препроцессированные данные
        json_filename = filename.rsplit('.', 1)[0] + '.json'
        json_path = PREPROCESSED_FOLDER / json_filename
        save_preprocessed_data(patients, str(json_path))

        return jsonify({
            'success': True,
            'stats': stats,
            'preprocessed_file': json_filename
        })

    return jsonify({'error': 'Invalid file format'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
