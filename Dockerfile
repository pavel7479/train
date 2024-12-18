# Базовый образ
FROM python:3.10-slim

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Обновляем pip
RUN python -m pip install --upgrade pip

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Указываем команду по умолчанию
CMD ["python", "main.py"]
