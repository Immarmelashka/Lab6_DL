FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Установка зависимостей
RUN pip install --no-cache-dir ultralytics opencv-python pillow torchvision

# ИСПРАВЛЕНИЕ: Копируем ВСЕ файлы из текущей папки (скрипт + оба файла весов)
COPY . /app

ENTRYPOINT ["python", "/app/predict.py"]
