# Subject Eligibility Checker

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
subject-eligibility/
├── src/
│   ├── nlp_analyzer.py       # NLP анализ (SapBERT)
│   ├── data_preprocessor.py  # Препроцессинг Excel
│   ├── csv_converter.py      # Конвертация CSV
│   ├── excel_handler.py      # Работа с Excel
│   └── eligibility_checker.py
├── templates/
│   ├── index.html            # Страница загрузки
│   └── results.html          # Страница результатов
├── data/
│   ├── uploads/              # Загруженные файлы
│   ├── results/              # Результаты обработки
│   └── preprocessed/         # Препроцессированные данные
├── models/                   # NLP модель SapBERT (~420MB)
├── app.py                    # Flask приложение
├── download_models.py        # Скачивание модели
├── requirements.txt
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

### POST /api/analyze

JSON API для программного анализа пациентов.

**Request:**

```json
{
    "patients": [
        {
            "patient_id": "P0001",
            "note": "0. Patient is 65 years old. 1. History of heart failure. 2. LVEF 35%.",
            "ground_truth": "included"
        },
        {
            "patient_id": "P0002",
            "note": "0. 45-year-old male. 1. Type 1 diabetes since 2010.",
            "ground_truth": "excluded"
        }
    ],
    "inclusion_criteria": [
        "Age >= 18 years",
        "Heart failure with reduced ejection fraction (LVEF <= 40%)"
    ],
    "exclusion_criteria": [
        "Type 1 diabetes",
        "Pregnancy"
    ]
}
```

**Поля запроса:**

| Поле | Тип | Описание |
|------|-----|----------|
| `patients` | array | Список пациентов для анализа |
| `patients[].patient_id` | string | ID пациента |
| `patients[].note` | string | Клинические заметки (нумерованные: 0. ... 1. ...) |
| `patients[].ground_truth` | string/null | Ожидаемый результат для расчета accuracy (опционально) |
| `inclusion_criteria` | array | Список критериев включения |
| `exclusion_criteria` | array | Список критериев исключения |

**Response:**

```json
{
    "total_patients": 2,
    "accuracy": 100.0,
    "results": [
        {
            "patient_id": "P0001",
            "eligibility": "included",
            "ground_truth": "included",
            "correct": true,
            "inclusion_results": [...],
            "exclusion_results": [...]
        }
    ]
}
```

**Пример вызова:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d @patients.json \
     http://localhost:5000/api/analyze
```

## Производительность

| Режим | Датасет | Пациентов | Accuracy |
|-------|---------|-----------|----------|
| Excel | Batch1 | 101 | 70.3% |
| CSV+TXT | Study W01 | 10 | 90% |
| CSV+TXT | Study W02 | 30 | 63.3% |
| **Combined** | **All** | **141** | **70.2%** |

- **Время на пациента**: ~1.7 сек (CPU)
- **Модель**: SapBERT (~420MB)
