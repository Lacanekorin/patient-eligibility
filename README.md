# Patient Eligibility Checker

Система автоматической оценки пациентов на соответствие критериям клинических исследований с использованием NLP.

## Быстрый старт

### Запуск из Docker Hub

```bash
docker pull lacanekorin/patient-eligibility-checker:latest
docker run -p 5000:5000 -v ./data:/app/data lacanekorin/patient-eligibility-checker:latest
```

Открыть: http://localhost:5000

### Локальная разработка

```bash
git clone <repo-url>
cd defy

# Создать venv и установить зависимости
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Скачать модель SapBERT (~420MB)
python download_models.py

# Запустить
python app.py
```

### Сборка Docker образа

```bash
docker-compose up --build
```

## Структура проекта

```
patient_Eligibility/
├── src/
│   ├── nlp_analyzer.py       # NLP анализ (sentence-transformers + SapBERT)
│   ├── data_preprocessor.py  # Препроцессинг Excel -> JSON
│   └── excel_handler.py      # Работа с Excel
├── templates/                # HTML шаблоны
├── app.py                    # Flask приложение
├── Dockerfile
└── docker-compose.yml
```

## Как это работает

1. Загружаешь Excel с данными пациентов
2. Система парсит критерии включения/исключения
3. NLP модель (SapBERT) сравнивает критерии с клиническими заметками
4. Результат: included / excluded / not enough information

## Ключевые файлы для модификации

| Файл | Что делает |
|------|------------|
| `src/nlp_analyzer.py` | Логика NLP анализа, пороги, детекция отрицаний |
| `src/data_preprocessor.py` | Парсинг Excel, разбиение на предложения |
| `app.py` | Flask роуты, финальное решение по пациенту |
| `templates/results.html` | UI результатов |

## Настройки (nlp_analyzer.py)

```python
match_threshold = 0.45    # Минимальный score для совпадения
high_confidence = 0.60    # Высокая уверенность
exclusion_threshold = 0.65  # Порог для exclusion критериев
```

## Текущая точность

**70.3%** на тестовом датасете (101 пациент)

## Формат входных данных

Excel со столбцами:
- `patient_id` — ID пациента
- `note` — клинические заметки (нумерованные: 0. ... 1. ... 2. ...)
- `trial_inclusion` — критерии включения (разделены \n)
- `trial_exclusion` — критерии исключения (разделены \n)
- `expert_eligibility` — ground truth (опционально)

## API

- `POST /upload` — загрузка Excel, возвращает HTML с результатами
- `POST /api/preprocess` — только препроцессинг, возвращает JSON
