"""Модуль для препроцессинга данных из Excel в структурированный формат."""

import pandas as pd
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Criterion:
    """Один критерий включения/исключения."""

    text: str
    type: str  # 'inclusion' или 'exclusion'
    keywords: List[str]


@dataclass
class PatientData:
    """Структурированные данные пациента."""

    patient_id: str
    trial_id: str
    trial_title: str
    note: str
    note_sentences: List[str]  # Разбитые предложения
    inclusion_criteria: List[Criterion]
    exclusion_criteria: List[Criterion]
    ground_truth: Optional[str]


def extract_keywords(text: str) -> List[str]:
    """Извлекает ключевые медицинские термины из текста."""
    # Медицинские паттерны
    patterns = [
        r"\b(?:LVEF|EF)\s*[<>=≤≥]\s*\d+%?",  # LVEF <= 40%
        r"\b(?:eGFR|GFR)\s*[<>=≤≥]\s*\d+",  # eGFR >= 30
        r"\b(?:NT-proBNP|BNP)\b",
        r"\b(?:NYHA)\s+(?:class\s+)?[I-IV]+",  # NYHA class II-IV
        r"\b(?:HFrEF|HFpEF|HF)\b",  # Heart failure types
        r"\b(?:diabetes|DM|T1DM|T2DM)\b",
        r"\b(?:MI|myocardial infarction)\b",
        r"\b(?:stroke|TIA)\b",
        r"\b(?:BP|blood pressure)\s*[<>=≤≥]?\s*\d+",
        r"\b(?:SGLT2|SGLT-2)\b",
        r"\b(?:CRT|ICD|pacemaker)\b",
        r"\b(?:cardiomyopathy|myocarditis|pericarditis)\b",
        r"\b(?:bradycardia|AV block|heart block)\b",
        r"\b(?:valvular disease|valve)\b",
        r"\baged?\s*[<>=≤≥]\s*\d+",  # age >= 18
        r"\b\d+\s*(?:years?|months?|weeks?)\s+(?:old|prior|ago)\b",
    ]

    keywords = []
    text_lower = text.lower()

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        keywords.extend(matches)

    # Также извлекаем важные существительные
    important_terms = [
        "consent",
        "symptomatic",
        "hypotension",
        "hospitalization",
        "decompensated",
        "revascularization",
        "transplantation",
        "implantation",
        "pregnant",
        "renal",
        "hepatic",
    ]

    for term in important_terms:
        if term in text_lower:
            keywords.append(term)

    return list(set(keywords))


def split_criteria(text: str) -> List[str]:
    """Разбивает текст критериев на отдельные критерии."""
    if not text or text == "nan":
        return []

    # Разбиваем по переносам строк
    criteria = text.split("\n")

    # Фильтруем пустые
    criteria = [c.strip() for c in criteria if c.strip() and len(c.strip()) > 5]

    return criteria


def split_note_into_sentences(note: str) -> List[str]:
    """Разбивает клинические заметки на предложения."""
    if not note or note == "nan":
        return []

    # Паттерн для медицинских значений (короткие предложения которые важны)
    medical_value_pattern = re.compile(
        r"\b(?:LVEF|EF|eGFR|GFR|NT-proBNP|BNP|SBP|DBP|BP|HbA1c)[:\s]*\d+|"
        r"\bNYHA\s*(?:class\s*)?[I-IV]+",
        re.IGNORECASE,
    )

    def is_valid_sentence(item: str) -> bool:
        """Проверяет валидность предложения."""
        if not item:
            return False
        # Короткие предложения допускаем если содержат медицинские значения
        if medical_value_pattern.search(item):
            return len(item) >= 5
        # Остальные должны быть длиннее
        return len(item) > 10

    # Сначала пробуем разбить по нумерации (0. 1. 2. ...)
    # Номер предложения: в начале текста или после ". " (точка с пробелом)
    # Находим все позиции маркеров
    marker_pattern = re.compile(r"(?:^|(?<=\.\s))(\d+)\.\s+", re.MULTILINE)
    markers = list(marker_pattern.finditer(note))

    if len(markers) > 2:
        sentences = []
        for i, m in enumerate(markers):
            start = m.end()  # Позиция после "N. "
            # Конец — начало следующего маркера или конец строки
            if i + 1 < len(markers):
                end = markers[i + 1].start()
            else:
                end = len(note)

            item = note[start:end].strip()
            # Убираем завершающую точку/пробел если есть
            item = item.rstrip(". ")
            if item and is_valid_sentence(item):
                sentences.append(item + ".")  # Добавляем точку обратно

        if sentences:
            return sentences

    # Иначе разбиваем по точкам
    sentences = re.split(r"(?<=[.!?])\s+", note)
    return [s.strip() for s in sentences if is_valid_sentence(s.strip())]


def get_column_value(row, *possible_names, default=""):
    """Получает значение столбца, пробуя разные варианты имён."""
    for name in possible_names:
        if name in row.index:
            val = row[name]
            if pd.notna(val):
                return str(val)
    return default


def preprocess_excel(df: pd.DataFrame) -> List[PatientData]:
    """
    Конвертирует DataFrame в список структурированных данных пациентов.
    """
    patients = []

    # Нормализуем имена столбцов (убираем \nstring и т.п.)
    print(f"Original columns: {list(df.columns)}")

    # Определяем столбец ground truth
    ground_truth_col = None
    for col in df.columns:
        col_str = str(col).lower()
        if "expert" in col_str or "eligibility" in col_str:
            ground_truth_col = col
            break

    for idx, row in df.iterrows():
        # Извлекаем данные с учётом разных вариантов имён столбцов
        patient_id = get_column_value(
            row, "patient_id\nstring", "patient_id", default=str(idx)
        )
        trial_id = get_column_value(row, "trial_id\nstring", "trial_id")
        trial_title = get_column_value(row, "trial_title\nstring", "trial_title")
        note = get_column_value(row, "note\nstring", "note")

        inclusion_text = str(row.get("trial_inclusion", ""))
        exclusion_text = str(row.get("trial_exclusion", ""))

        # Ground truth
        ground_truth = None
        if ground_truth_col:
            gt_value = row.get(ground_truth_col, "")
            if pd.notna(gt_value):
                ground_truth = str(gt_value).strip().lower()

        # Разбиваем на критерии
        inclusion_criteria = [
            Criterion(text=c, type="inclusion", keywords=extract_keywords(c))
            for c in split_criteria(inclusion_text)
        ]

        exclusion_criteria = [
            Criterion(text=c, type="exclusion", keywords=extract_keywords(c))
            for c in split_criteria(exclusion_text)
        ]

        # Разбиваем note на предложения
        note_sentences = split_note_into_sentences(note)

        # Debug: выводим первого пациента
        if idx == 0:
            print("\n=== DEBUG: First patient ===")
            print(f"patient_id: {patient_id}")
            print(f"note length: {len(note)}")
            print(f"note first 300 chars: {note[:300]}")
            print(f"note_sentences count: {len(note_sentences)}")
            if note_sentences:
                print(f"first sentence: {note_sentences[0][:100]}")
            print(f"inclusion_criteria count: {len(inclusion_criteria)}")
            print(f"exclusion_criteria count: {len(exclusion_criteria)}")
            print("=" * 50)

        patient = PatientData(
            patient_id=patient_id,
            trial_id=trial_id,
            trial_title=trial_title,
            note=note,
            note_sentences=note_sentences,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            ground_truth=ground_truth,
        )

        patients.append(patient)

    return patients


def save_preprocessed_data(patients: List[PatientData], output_path: str) -> str:
    """Сохраняет препроцессированные данные в JSON."""
    data = [asdict(p) for p in patients]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def load_preprocessed_data(input_path: str) -> List[PatientData]:
    """Загружает препроцессированные данные из JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patients = []
    for item in data:
        inclusion_criteria = [Criterion(**c) for c in item["inclusion_criteria"]]
        exclusion_criteria = [Criterion(**c) for c in item["exclusion_criteria"]]

        patient = PatientData(
            patient_id=item["patient_id"],
            trial_id=item["trial_id"],
            trial_title=item["trial_title"],
            note=item["note"],
            note_sentences=item["note_sentences"],
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            ground_truth=item.get("ground_truth"),
        )
        patients.append(patient)

    return patients


def get_preprocessing_stats(patients: List[PatientData]) -> Dict[str, Any]:
    """Возвращает статистику по препроцессированным данным."""
    total_inclusion = sum(len(p.inclusion_criteria) for p in patients)
    total_exclusion = sum(len(p.exclusion_criteria) for p in patients)
    total_sentences = sum(len(p.note_sentences) for p in patients)

    gt_counts = {
        "included": 0,
        "excluded": 0,
        "not enough information": 0,
        "unknown": 0,
    }
    for p in patients:
        if p.ground_truth:
            if "include" in p.ground_truth:
                gt_counts["included"] += 1
            elif "exclude" in p.ground_truth:
                gt_counts["excluded"] += 1
            elif "not enough" in p.ground_truth or "info" in p.ground_truth:
                gt_counts["not enough information"] += 1
            else:
                gt_counts["unknown"] += 1

    return {
        "total_patients": len(patients),
        "total_inclusion_criteria": total_inclusion,
        "total_exclusion_criteria": total_exclusion,
        "avg_inclusion_per_patient": (
            round(total_inclusion / len(patients), 1) if patients else 0
        ),
        "avg_exclusion_per_patient": (
            round(total_exclusion / len(patients), 1) if patients else 0
        ),
        "total_note_sentences": total_sentences,
        "avg_sentences_per_patient": (
            round(total_sentences / len(patients), 1) if patients else 0
        ),
        "ground_truth_distribution": gt_counts,
    }
