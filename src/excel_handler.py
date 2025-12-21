"""Модуль для работы с Excel файлами."""

import re
from collections import defaultdict
from typing import List, Dict, Any, Optional

import pandas as pd


def load_excel(file_path: str) -> pd.DataFrame:
    """
    Загружает Excel файл и возвращает DataFrame.

    Args:
        file_path: Путь к Excel файлу

    Returns:
        DataFrame с данными пациентов
    """
    df = pd.read_excel(file_path)
    return df


def _get_study_name_for_patient(patient: Dict[str, Any]) -> str:
    """
    Определяет название исследования для пациента.

    Приоритет:
    1. trial_id из данных пациента (если есть)
    2. Определение по patient_id (P0001 -> Study W01, S0001 -> Study W02)
    3. "Study" по умолчанию
    """
    # Сначала проверяем trial_id
    trial_id = patient.get("trial_id", "")
    if trial_id:
        return str(trial_id)

    # Fallback на определение по patient_id
    patient_id = patient.get("patient_id", "")
    if not patient_id:
        return "Study"

    # Пробуем найти паттерны исследования
    if patient_id.startswith("P"):
        return "Study W01"
    elif patient_id.startswith("S"):
        return "Study W02"

    # Ищем W01, W02 и т.д. в ID
    match = re.search(r"W(\d+)", patient_id, re.IGNORECASE)
    if match:
        return f"Study W{match.group(1)}"

    return "Study"


def _build_criteria_matrix(patients: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Строит матрицу пациенты x критерии с результатами MET/NOT MET/UNKNOWN.

    Args:
        patients: Список результатов по пациентам одного исследования

    Returns:
        DataFrame с матрицей критериев:
        - Первый столбец: Patient ID
        - Остальные столбцы: полный текст критерия (заголовок),
          значения MET/NOT MET/UNKNOWN
    """
    if not patients:
        return pd.DataFrame()

    rows = []

    for patient in patients:
        row = {
            "Patient ID": patient.get("patient_id", ""),
        }

        # Добавляем inclusion критерии
        inclusion_results = patient.get("inclusion_results", [])
        for criterion_result in inclusion_results:
            criterion_text = criterion_result.get("criterion", "")
            col_name = f"[INC] {criterion_text}"

            is_met = criterion_result.get("is_met")
            if is_met is True:
                value = "MET"
            elif is_met is False:
                value = "NOT MET"
            else:
                value = "UNKNOWN"

            row[col_name] = value

        # Добавляем exclusion критерии
        exclusion_results = patient.get("exclusion_results", [])
        for criterion_result in exclusion_results:
            criterion_text = criterion_result.get("criterion", "")
            col_name = f"[EXC] {criterion_text}"

            is_met = criterion_result.get("is_met")
            if is_met is True:
                value = "MET"
            elif is_met is False:
                value = "NOT MET"
            else:
                value = "UNKNOWN"

            row[col_name] = value

        rows.append(row)

    return pd.DataFrame(rows)


def _group_patients_by_study(
    detailed_results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Группирует пациентов по исследованиям.

    Returns:
        Словарь {название_исследования: [список_пациентов]}
    """
    studies = defaultdict(list)

    for patient in detailed_results:
        study_name = _get_study_name_for_patient(patient)
        studies[study_name].append(patient)

    return dict(studies)


def save_results(
    eligible: pd.DataFrame,
    not_eligible: pd.DataFrame,
    need_clarification: pd.DataFrame,
    output_path: str,
    detailed_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Сохраняет результаты в Excel файл.

    Вкладки:
    - Подходящие
    - Не подходящие
    - Требуют уточнения
    - Отдельная вкладка для каждого исследования с матрицей критериев
      (название = trial_id или определяется по patient_id)

    Args:
        eligible: DataFrame с подходящими пациентами
        not_eligible: DataFrame с неподходящими пациентами
        need_clarification: DataFrame с пациентами, требующими уточнения
        output_path: Путь для сохранения результата
        detailed_results: Детальные результаты для матрицы критериев (опционально)

    Returns:
        Путь к сохраненному файлу
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        eligible.to_excel(writer, sheet_name="Подходящие", index=False)
        not_eligible.to_excel(writer, sheet_name="Не подходящие", index=False)
        need_clarification.to_excel(writer, sheet_name="Требуют уточнения", index=False)

        # Добавляем матрицы критериев для каждого исследования
        if detailed_results:
            studies = _group_patients_by_study(detailed_results)

            for study_name, patients in studies.items():
                criteria_matrix = _build_criteria_matrix(patients)
                if not criteria_matrix.empty:
                    # Excel ограничивает длину имени листа до 31 символа
                    sheet_name = str(study_name)[:31]
                    criteria_matrix.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_path
