# Используем официальный легковесный PyTorch образ
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Установка системных библиотек, необходимых для работы OpenCV в Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем необходимые Python библиотеки
RUN pip install --no-cache-dir ultralytics opencv-python pillow

# Копируем наш скрипт инференса внутрь контейнера
COPY predict.py /app/predict.py

# Команда по умолчанию, которая объясняет как запускать контейнер
ENTRYPOINT ["python", "/app/predict.py"]
