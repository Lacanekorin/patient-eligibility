# Patient Eligibility Checker

Автоматическая оценка пациентов на соответствие критериям клинических исследований с использованием NLP (SapBERT).

## Быстрый старт

### Docker Hub

```bash
docker pull lacanekorin/subject-eligibility-checker:latest
docker run -p 5000:5000 -v ./data:/app/data lacanekorin/subject-eligibility-checker:latest
```

Открыть: http://localhost:5000

### Локально

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
python download_models.py  # Скачать SapBERT (~420MB)
python app.py
```

### Docker сборка

```bash
docker-compose up --build
```

## Структура

```
defy/
├── src/
│   ├── nlp_analyzer.py       # NLP анализ (SapBERT)
│   ├── data_preprocessor.py  # Препроцессинг Excel
│   ├── csv_converter.py      # Конвертация CSV
│   └── excel_handler.py      # Работа с Excel
├── templates/                # HTML шаблоны
├── app.py                    # Flask приложение
├── download_models.py        # Скачивание модели
├── Dockerfile
└── docker-compose.yml
```

## Как работает

1. Загрузка Excel/CSV с данными пациентов
2. Парсинг критериев включения/исключения
3. NLP модель сравнивает критерии с клиническими заметками
4. Результат: included / excluded / not enough information

## Формат входных данных

### Excel

| Столбец | Описание |
|---------|----------|
| `patient_id` | ID пациента |
| `note` | Клинические заметки (0. ... 1. ... 2. ...) |
| `trial_id` | ID исследования (опционально) |
| `trial_inclusion` | Критерии включения (разделены \n) |
| `trial_exclusion` | Критерии исключения (разделены \n) |
| `expert_eligibility` | Ground truth (опционально) |

### CSV режим

- `criteria.txt` — критерии включения/исключения
- `*_anamnesis.csv`, `*_clinical_note.csv` — данные пациента
- `*_blood_labs.csv`, `*_lipid_labs.csv` и др. — лабораторные данные

## Настройки (nlp_analyzer.py)

```python
match_threshold = 0.45      # Минимальный score для совпадения
high_confidence = 0.60      # Высокая уверенность
exclusion_threshold = 0.65  # Порог для exclusion критериев
UNKNOWN_THRESHOLD = 0.20    # Порог unknown для "not enough info"
```

## API

| Endpoint | Описание |
|----------|----------|
| `POST /upload` | Загрузка Excel |
| `POST /upload_csv` | Загрузка CSV + criteria.txt |
| `POST /api/analyze` | JSON API для анализа |
| `GET /download/<file>` | Скачивание результатов |

## Производительность

- **Accuracy**: 70% на тестовом датасете
- **Время на пациента**: ~1.7 сек (CPU)
- **Модель**: SapBERT (~420MB)
