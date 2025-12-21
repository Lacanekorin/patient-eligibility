"""Модуль для проверки соответствия пациентов критериям исследования."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .nlp_analyzer import get_analyzer


class EligibilityStatus(Enum):
    """Статус соответствия пациента критериям."""

    ELIGIBLE = "included"
    NOT_ELIGIBLE = "excluded"
    NEEDS_CLARIFICATION = "not enough information"


@dataclass
class CriterionResult:
    """Результат проверки одного критерия."""

    criterion: str
    is_met: Optional[bool]
    explanation: str
    quotes: List[str]
    similarity_score: float = 0.0


@dataclass
class PatientResult:
    """Результат проверки пациента."""

    patient_id: str
    status: EligibilityStatus
    inclusion_results: List[CriterionResult]
    exclusion_results: List[CriterionResult]
    summary: str


def evaluate_patient(
    patient_id: str,
    clinical_note: str,
    inclusion_criteria: List[str],
    exclusion_criteria: List[str],
) -> PatientResult:
    """
    Оценивает пациента по всем критериям включения и исключения.
    """
    analyzer = get_analyzer()

    status_str, summary, inc_results, exc_results = analyzer.evaluate_all_criteria(
        clinical_note=clinical_note,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
    )

    # Преобразуем в наши структуры
    inclusion_results = [
        CriterionResult(
            criterion=r["criterion"],
            is_met=r["is_met"],
            explanation=r["explanation"],
            quotes=r["quotes"],
            similarity_score=r["score"],
        )
        for r in inc_results
    ]

    exclusion_results = [
        CriterionResult(
            criterion=r["criterion"],
            is_met=r["is_met"],
            explanation=r["explanation"],
            quotes=r["quotes"],
            similarity_score=r["score"],
        )
        for r in exc_results
    ]

    # Преобразуем статус
    status_map = {
        "included": EligibilityStatus.ELIGIBLE,
        "excluded": EligibilityStatus.NOT_ELIGIBLE,
        "not enough information": EligibilityStatus.NEEDS_CLARIFICATION,
    }
    status = status_map.get(status_str, EligibilityStatus.NEEDS_CLARIFICATION)

    # Генерируем подробное описание
    detailed_summary = generate_summary(status, inclusion_results, exclusion_results)

    return PatientResult(
        patient_id=patient_id,
        status=status,
        inclusion_results=inclusion_results,
        exclusion_results=exclusion_results,
        summary=detailed_summary,
    )


def generate_summary(
    status: EligibilityStatus,
    inclusion_results: List[CriterionResult],
    exclusion_results: List[CriterionResult],
) -> str:
    """Генерирует текстовое описание результата оценки."""
    lines = []

    status_text = {
        EligibilityStatus.ELIGIBLE: "INCLUDED (Подходит)",
        EligibilityStatus.NOT_ELIGIBLE: "EXCLUDED (Не подходит)",
        EligibilityStatus.NEEDS_CLARIFICATION: "NOT ENOUGH INFO (Требуется уточнение)",
    }

    lines.append(f"Status: {status_text[status]}")
    lines.append("")

    lines.append("=== Inclusion Criteria ===")
    for r in inclusion_results:
        mark = "+" if r.is_met else ("-" if r.is_met is False else "?")
        lines.append(f"[{mark}] {r.criterion[:100]}")
        lines.append(f"    {r.explanation}")
        if r.quotes:
            lines.append(f'    Quote: "{r.quotes[0][:150]}..."')
        lines.append("")

    lines.append("=== Exclusion Criteria ===")
    for r in exclusion_results:
        # Для exclusion: + означает НЕ найден (хорошо), - найден (плохо)
        mark = "-" if r.is_met else ("+" if r.is_met is False else "?")
        lines.append(f"[{mark}] {r.criterion[:100]}")
        lines.append(f"    {r.explanation}")
        if r.quotes and r.is_met:
            lines.append(f'    Quote: "{r.quotes[0][:150]}..."')
        lines.append("")

    return "\n".join(lines)
