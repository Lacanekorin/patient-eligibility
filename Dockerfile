FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch CPU-only (намного меньше чем CUDA версия)
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем sentence-transformers и transformers
RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    sentence-transformers==2.7.0

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    werkzeug==3.0.1 \
    pandas==2.1.4 \
    openpyxl==3.1.2 \
    scikit-learn==1.3.2 \
    python-dotenv==1.0.0

# Копируем предзагруженную модель SapBERT
COPY models /root/.cache/huggingface/hub

# Копируем исходный код
COPY . .

# Создаём директории для данных
RUN mkdir -p data/uploads data/results data/preprocessed

# Порт приложения
EXPOSE 5000

# Запуск приложения
CMD ["python", "app.py"]
