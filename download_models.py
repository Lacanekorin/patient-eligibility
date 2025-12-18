"""
Скрипт для скачивания медицинских NLP моделей локально.
Модели сохраняются в папку models/ для использования в Docker.

Использование:
    c:\other\defy\venv\Scripts\python.exe download_models.py
"""

from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import shutil

# Список моделей для скачивания
MODELS = [
    # Медицинские модели
    {
        "name": "pritamdeka/S-Biomed-Roberta-MS-MARCO",
        "description": "BioMed RoBERTa, trained on MS MARCO for semantic search",
    },
    {
        "name": "NeuML/pubmedbert-base-embeddings",
        "description": "PubMedBERT fine-tuned for embeddings",
    },
    {
        "name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "description": "SapBERT - biomedical entity embeddings from PubMedBERT",
    },
    {
        "name": "emilyalsentzer/Bio_ClinicalBERT",
        "description": "ClinicalBERT - trained on MIMIC clinical notes",
    },
    {
        "name": "medicalai/ClinicalBERT",
        "description": "Alternative ClinicalBERT implementation",
    },
    # Дополнительные полезные модели
    {
        "name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "description": "Microsoft BiomedBERT - trained on PubMed abstracts and full text",
    },
]


def get_cache_dir():
    """Возвращает путь к кэшу huggingface."""
    # Windows default
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return cache_dir


def model_name_to_folder(model_name: str) -> str:
    """Конвертирует имя модели в имя папки в кэше."""
    return "models--" + model_name.replace("/", "--")


def download_model(model_name: str, description: str) -> bool:
    """Скачивает модель через sentence-transformers."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    try:
        # Пробуем загрузить как sentence-transformer
        model = SentenceTransformer(model_name)

        # Тестируем что модель работает
        test_embedding = model.encode("This is a test sentence.")
        print(f"Model loaded successfully! Embedding size: {len(test_embedding)}")

        # Освобождаем память
        del model

        return True

    except Exception as e:
        print(f"ERROR downloading {model_name}: {e}")
        return False


def copy_to_models_dir():
    """Копирует скачанные модели из кэша в папку models/."""
    cache_dir = get_cache_dir()
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("Copying models to local models/ directory...")
    print(f"{'='*60}")

    for model_info in MODELS:
        model_name = model_info["name"]
        folder_name = model_name_to_folder(model_name)
        src = cache_dir / folder_name
        dst = models_dir / folder_name

        if src.exists():
            if dst.exists():
                print(f"Already exists: {folder_name}")
            else:
                print(f"Copying: {folder_name}")
                shutil.copytree(src, dst)
                print(f"  -> Done")
        else:
            print(f"Not found in cache: {folder_name}")


def list_available_models():
    """Показывает список доступных моделей."""
    models_dir = Path(__file__).parent / "models"

    print(f"\n{'='*60}")
    print("Available models in models/ directory:")
    print(f"{'='*60}")

    if models_dir.exists():
        for folder in sorted(models_dir.iterdir()):
            if folder.is_dir() and folder.name.startswith("models--"):
                # Получаем размер папки
                size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"  {folder.name}: {size_mb:.1f} MB")
    else:
        print("  (no models directory)")


def main():
    print("="*60)
    print("Medical NLP Models Downloader")
    print("="*60)

    # Показываем что будем качать
    print("\nModels to download:")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model['name']}")
        print(f"     {model['description']}")

    # Скачиваем модели
    results = []
    for model in MODELS:
        success = download_model(model["name"], model["description"])
        results.append((model["name"], success))

    # Копируем в локальную папку
    copy_to_models_dir()

    # Показываем результаты
    print(f"\n{'='*60}")
    print("Download Results:")
    print(f"{'='*60}")

    for model_name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {model_name}")

    # Показываем доступные модели
    list_available_models()

    print("\nDone! Models are ready for use in Docker.")
    print("Update docker-compose.pytorch.yml to mount the models/ directory.")


if __name__ == "__main__":
    main()
