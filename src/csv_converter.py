"""
Конвертер CSV файлов пациента в формат, понятный NLP модели.

Поддерживаемые типы файлов:
- *_anamnesis.csv / *_clinical_note.csv — основные данные пациента + narrative_note
- *_blood_labs.csv / *_lipid_labs.csv / *_renal_labs.csv — лабораторные данные
- *_urinalysis.csv — анализ мочи
- *_encounters.csv — история визитов
- criteria.txt — критерии включения/исключения
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает имена столбцов от лишних символов.

    Excel иногда добавляет точки с запятой или другие символы в конец заголовков
    при сохранении CSV. Эта функция удаляет такие артефакты.
    """
    # Создаём маппинг старых имён на очищенные
    new_columns = {}
    for col in df.columns:
        # Удаляем точки с запятой, пробелы и другие спецсимволы в конце
        clean_name = col.rstrip(';, \t\n\r')
        # Также удаляем в начале (на всякий случай)
        clean_name = clean_name.lstrip(';, \t\n\r')
        new_columns[col] = clean_name

    return df.rename(columns=new_columns)


def parse_criteria_file(content: str) -> Tuple[List[str], List[str]]:
    """
    Парсит файл criteria.txt и извлекает критерии включения и исключения.

    Returns:
        (inclusion_criteria, exclusion_criteria)
    """
    inclusion_criteria = []
    exclusion_criteria = []

    lines = content.strip().split('\n')
    current_section = None
    current_criterion = []

    for line in lines:
        line_stripped = line.strip()

        # Определяем секцию
        if 'INCLUSION CRITERIA' in line_stripped.upper():
            # Сохраняем предыдущий критерий если есть
            if current_criterion and current_section:
                text = ' '.join(current_criterion).strip()
                if text:
                    if current_section == 'inclusion':
                        inclusion_criteria.append(text)
                    elif current_section == 'exclusion':
                        exclusion_criteria.append(text)
            current_section = 'inclusion'
            current_criterion = []
            continue
        elif 'EXCLUSION CRITERIA' in line_stripped.upper():
            # Сохраняем предыдущий критерий
            if current_criterion and current_section:
                text = ' '.join(current_criterion).strip()
                if text:
                    if current_section == 'inclusion':
                        inclusion_criteria.append(text)
                    elif current_section == 'exclusion':
                        exclusion_criteria.append(text)
            current_section = 'exclusion'
            current_criterion = []
            continue
        elif line_stripped.upper().startswith('NOTES') or line_stripped.upper().startswith('DEFINITIONS'):
            # Сохраняем последний критерий и выходим
            if current_criterion and current_section:
                text = ' '.join(current_criterion).strip()
                if text:
                    if current_section == 'inclusion':
                        inclusion_criteria.append(text)
                    elif current_section == 'exclusion':
                        exclusion_criteria.append(text)
            current_section = None
            current_criterion = []
            continue

        if current_section is None:
            continue

        # Проверяем, начинается ли новый критерий (номер или буква)
        # Паттерны: "1)", "2)", "A)", "B)", "1.", "2.", "A.", "B."
        new_criterion_match = re.match(r'^([0-9]+|[A-Z])[)\.]', line_stripped)

        if new_criterion_match:
            # Сохраняем предыдущий критерий
            if current_criterion:
                text = ' '.join(current_criterion).strip()
                if text:
                    if current_section == 'inclusion':
                        inclusion_criteria.append(text)
                    elif current_section == 'exclusion':
                        exclusion_criteria.append(text)
            # Начинаем новый критерий (убираем номер/букву)
            criterion_text = re.sub(r'^([0-9]+|[A-Z])[)\.]?\s*', '', line_stripped)
            current_criterion = [criterion_text] if criterion_text else []
        elif line_stripped and current_criterion is not None:
            # Продолжение текущего критерия (многострочный)
            current_criterion.append(line_stripped)

    # Сохраняем последний критерий
    if current_criterion and current_section:
        text = ' '.join(current_criterion).strip()
        if text:
            if current_section == 'inclusion':
                inclusion_criteria.append(text)
            elif current_section == 'exclusion':
                exclusion_criteria.append(text)

    return inclusion_criteria, exclusion_criteria


def format_lab_results(df: pd.DataFrame) -> str:
    """
    Форматирует лабораторные результаты в текст для NLP.
    """
    if df.empty:
        return ""

    sentences = []

    # Группируем по дате
    if 'collection_date' in df.columns:
        df_sorted = df.sort_values('collection_date', ascending=False)
    else:
        df_sorted = df

    # Берём последние результаты для каждого теста
    seen_tests = set()
    for _, row in df_sorted.iterrows():
        test_name = row.get('test_name', '')
        if test_name in seen_tests:
            continue
        seen_tests.add(test_name)

        value = row.get('value', '')
        unit = row.get('unit', '')
        date = row.get('collection_date', '')
        flag = row.get('flag', '')

        if pd.notna(value) and value != '':
            sentence = f"{test_name}: {value}"
            if pd.notna(unit) and unit:
                sentence += f" {unit}"
            if pd.notna(date) and date:
                sentence += f" (on {date})"
            if pd.notna(flag) and flag:
                sentence += f" [{flag}]"
            sentences.append(sentence)

    return ". ".join(sentences)


def format_encounters(df: pd.DataFrame) -> str:
    """
    Форматирует историю визитов в текст.
    """
    if df.empty:
        return ""

    sentences = []
    count = len(df)

    if 'encounter_date' in df.columns:
        dates = df['encounter_date'].dropna().tolist()
        if dates:
            sentences.append(f"Patient had {count} outpatient visits")

    return ". ".join(sentences)


def build_note_from_csvs(csv_files: Dict[str, pd.DataFrame]) -> str:
    """
    Собирает note из нескольких CSV файлов.

    Args:
        csv_files: словарь {тип_файла: DataFrame}
            Типы: 'anamnesis', 'clinical_note', 'blood_labs', 'lipid_labs',
                  'renal_labs', 'urinalysis', 'encounters'

    Returns:
        Текст note для NLP модели
    """
    note_parts = []

    # 1. Основные данные пациента (anamnesis или clinical_note)
    main_df = csv_files.get('anamnesis')
    if main_df is None or (isinstance(main_df, pd.DataFrame) and main_df.empty):
        main_df = csv_files.get('clinical_note')

    if main_df is not None and not main_df.empty:
        row = main_df.iloc[0]

        # Демографические данные
        demographics = []
        age = row.get('age')
        sex = row.get('sex')
        if pd.notna(age):
            demographics.append(f"{int(age)}-year-old")
        if pd.notna(sex):
            demographics.append(str(sex))
        if demographics:
            note_parts.append(" ".join(demographics) + " patient")

        # Диагнозы
        diagnoses = row.get('diagnoses')
        if pd.notna(diagnoses) and diagnoses:
            note_parts.append(f"Diagnoses: {diagnoses}")

        # Медикаменты
        medications = row.get('medications')
        if pd.notna(medications) and medications:
            note_parts.append(f"Current medications: {medications}")

        # BMI
        bmi = row.get('bmi')
        if pd.notna(bmi):
            note_parts.append(f"BMI: {bmi} kg/m²")

        # Статус беременности
        pregnancy = row.get('pregnancy_status')
        if pd.notna(pregnancy) and pregnancy and pregnancy.lower() not in ['n/a', 'na', '']:
            if pregnancy.lower() in ['yes', 'pregnant']:
                note_parts.append("Patient is pregnant")
            elif pregnancy.lower() == 'no':
                note_parts.append("Patient is not pregnant")

        # Кардиоваскулярные события
        cv_event = row.get('recent_cardiovascular_event_6m') or row.get('recent_cardiovascular_event_3m')
        if pd.notna(cv_event) and str(cv_event).lower() == 'yes':
            note_parts.append("Recent cardiovascular event")
        elif pd.notna(cv_event) and str(cv_event).lower() == 'no':
            note_parts.append("No recent cardiovascular event")

        # Стероиды
        steroids = row.get('systemic_steroids_chronic')
        if pd.notna(steroids) and str(steroids).lower() == 'yes':
            note_parts.append("On chronic systemic steroids")
        elif pd.notna(steroids) and str(steroids).lower() == 'no':
            note_parts.append("No chronic systemic steroids")

        # Активная малигнизация
        malignancy = row.get('active_malignancy_systemic_tx_12m')
        if pd.notna(malignancy) and str(malignancy).lower() == 'yes':
            note_parts.append("Active malignancy with systemic therapy")
        elif pd.notna(malignancy) and str(malignancy).lower() == 'no':
            note_parts.append("No active malignancy")

        # Statin-специфичные поля (Study W02)
        statin_intolerance = row.get('statin_intolerance')
        if pd.notna(statin_intolerance) and str(statin_intolerance).lower() == 'yes':
            note_parts.append("Documented statin intolerance")

        prior_statin = row.get('prior_statin_use_180d')
        if pd.notna(prior_statin) and str(prior_statin).lower() == 'yes':
            note_parts.append("Prior statin use within 180 days")
        elif pd.notna(prior_statin) and str(prior_statin).lower() == 'no':
            note_parts.append("No prior statin use within 180 days")

        # Участие в клинических исследованиях
        trial = row.get('interventional_trial_last_3m')
        if pd.notna(trial) and str(trial).lower() == 'yes':
            note_parts.append("Participation in interventional trial within last 3 months")

        # Narrative note (самое важное!)
        narrative = row.get('narrative_note')
        if pd.notna(narrative) and narrative:
            note_parts.append(str(narrative))

    # 2. Лабораторные данные
    lab_types = ['blood_labs', 'lipid_labs', 'renal_labs']
    for lab_type in lab_types:
        lab_df = csv_files.get(lab_type)
        if lab_df is not None and not lab_df.empty:
            lab_text = format_lab_results(lab_df)
            if lab_text:
                note_parts.append(f"Lab results: {lab_text}")

    # 3. Анализ мочи
    urinalysis_df = csv_files.get('urinalysis')
    if urinalysis_df is not None and not urinalysis_df.empty:
        urine_text = format_lab_results(urinalysis_df)
        if urine_text:
            note_parts.append(f"Urinalysis: {urine_text}")

    # 4. История визитов
    encounters_df = csv_files.get('encounters')
    if encounters_df is not None and not encounters_df.empty:
        encounters_text = format_encounters(encounters_df)
        if encounters_text:
            note_parts.append(encounters_text)

    return ". ".join(note_parts)


def detect_file_type(filename: str) -> Optional[str]:
    """
    Определяет тип файла по имени.
    """
    filename_lower = filename.lower()

    if 'anamnesis' in filename_lower:
        return 'anamnesis'
    elif 'clinical_note' in filename_lower:
        return 'clinical_note'
    elif 'blood_labs' in filename_lower or 'blood_lab' in filename_lower:
        return 'blood_labs'
    elif 'lipid_labs' in filename_lower or 'lipid_lab' in filename_lower:
        return 'lipid_labs'
    elif 'renal_labs' in filename_lower or 'renal_lab' in filename_lower:
        return 'renal_labs'
    elif 'urinalysis' in filename_lower:
        return 'urinalysis'
    elif 'encounters' in filename_lower or 'encounter' in filename_lower:
        return 'encounters'
    elif filename_lower.endswith('.csv'):
        # Попробуем определить по содержимому позже
        return 'unknown_csv'

    return None


def extract_patient_id(filename: str) -> str:
    """
    Извлекает patient_id из имени файла.
    Примеры: P0001_anamnesis.csv -> P0001, S0001_clinical_note.csv -> S0001
    """
    # Паттерн: начало имени до первого подчёркивания
    match = re.match(r'^([A-Za-z0-9]+)_', filename)
    if match:
        return match.group(1)

    # Если не найдено, используем имя файла без расширения
    return Path(filename).stem


def convert_csv_files_to_patient_data(
    csv_files: Dict[str, bytes],
    criteria_content: str
) -> Dict:
    """
    Конвертирует загруженные CSV файлы в формат для NLP модели.

    Args:
        csv_files: словарь {имя_файла: содержимое_bytes}
        criteria_content: содержимое файла criteria.txt

    Returns:
        Словарь с данными пациента в формате модели
    """
    # Парсим критерии
    inclusion_criteria, exclusion_criteria = parse_criteria_file(criteria_content)

    # Группируем файлы по patient_id
    patients_files: Dict[str, Dict[str, pd.DataFrame]] = {}

    for filename, content in csv_files.items():
        file_type = detect_file_type(filename)
        if file_type is None:
            continue

        patient_id = extract_patient_id(filename)

        if patient_id not in patients_files:
            patients_files[patient_id] = {}

        try:
            # Читаем CSV
            from io import BytesIO
            df = pd.read_csv(BytesIO(content))

            # Очищаем имена столбцов от артефактов Excel (лишние ; в конце)
            df = clean_column_names(df)

            # Если тип unknown_csv, пробуем определить по столбцам
            if file_type == 'unknown_csv':
                if 'test_name' in df.columns and 'value' in df.columns:
                    file_type = 'blood_labs'
                elif 'narrative_note' in df.columns:
                    if 'statin_name' in df.columns:
                        file_type = 'clinical_note'
                    else:
                        file_type = 'anamnesis'
                elif 'encounter_date' in df.columns:
                    file_type = 'encounters'

            patients_files[patient_id][file_type] = df

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    # Конвертируем каждого пациента
    patients_data = []

    for patient_id, files in patients_files.items():
        # Собираем note
        note = build_note_from_csvs(files)

        # Получаем ground truth если есть
        ground_truth = None
        main_df = files.get('anamnesis')
        if main_df is None or (isinstance(main_df, pd.DataFrame) and main_df.empty):
            main_df = files.get('clinical_note')
        if main_df is not None and not main_df.empty:
            eligible_label = main_df.iloc[0].get('eligible_label')
            if pd.notna(eligible_label):
                ground_truth = str(eligible_label).lower()
                if ground_truth == 'eligible':
                    ground_truth = 'included'
                elif ground_truth == 'ineligible':
                    ground_truth = 'excluded'

        patient_data = {
            'patient_id': patient_id,
            'note': note,
            'trial_inclusion': '\n'.join(inclusion_criteria),
            'trial_exclusion': '\n'.join(exclusion_criteria),
            'expert_eligibility': ground_truth
        }

        patients_data.append(patient_data)

    return {
        'patients': patients_data,
        'inclusion_criteria': inclusion_criteria,
        'exclusion_criteria': exclusion_criteria
    }


def convert_to_dataframe(data: Dict) -> pd.DataFrame:
    """
    Конвертирует данные в DataFrame, совместимый с текущим пайплайном.
    """
    rows = []
    for patient in data['patients']:
        rows.append({
            'patient_id\nstring': patient['patient_id'],
            'note\nstring': patient['note'],
            'trial_inclusion': patient['trial_inclusion'],
            'trial_exclusion': patient['trial_exclusion'],
            'expert_eligibility\nstring': patient.get('expert_eligibility', '')
        })

    return pd.DataFrame(rows)
