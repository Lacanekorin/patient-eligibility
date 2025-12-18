"""Модуль для работы с Excel файлами."""
import pandas as pd
from pathlib import Path


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


def save_results(
    eligible: pd.DataFrame,
    not_eligible: pd.DataFrame,
    need_clarification: pd.DataFrame,
    output_path: str
) -> str:
    """
    Сохраняет результаты в Excel файл с тремя листами.

    Args:
        eligible: DataFrame с подходящими пациентами
        not_eligible: DataFrame с неподходящими пациентами
        need_clarification: DataFrame с пациентами, требующими уточнения
        output_path: Путь для сохранения результата

    Returns:
        Путь к сохраненному файлу
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        eligible.to_excel(writer, sheet_name='Подходящие', index=False)
        not_eligible.to_excel(writer, sheet_name='Не подходящие', index=False)
        need_clarification.to_excel(writer, sheet_name='Требуют уточнения', index=False)

    return output_path
