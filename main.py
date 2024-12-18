
import cv2
import numpy as np
import os

# Путь к видеофайлу
video_path = './video_data/train.mov'
output_path = './video_data/output_video.mp4'


# Открываем видеофайл
cap = cv2.VideoCapture(video_path)

# Проверяем успешность открытия видео
if not cap.isOpened():
    print("Ошибка: не удалось открыть видео.")
    exit()

# Получаем параметры видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Кодек и объект для записи видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Читаем первый кадр и преобразуем в оттенки серого
ret, prev_frame = cap.read()
if not ret:
    print("Ошибка: не удалось прочитать первый кадр.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Основной цикл обработки видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем текущий кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Вычисляем оптический поток
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Сглаживаем оптический поток
    smoothed_flow = cv2.blur(flow, (5, 5))

    # Вычисляем среднее направление потока
    avg_flow = np.mean(smoothed_flow, axis=(0, 1))
    fx, fy = avg_flow[0], avg_flow[1]

    # Параметры стрелки для отображения направления
    start_point = (frame_width // 2, frame_height // 2)
    scale = 30  # Масштаб для отображения
    end_point = (int(start_point[0] + fx * scale), int(start_point[1] + fy * scale))

    # Отображаем стрелку на кадре
    cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 5)

    # Записываем обработанный кадр в выходное видео
    out.write(frame)

    # Обновляем предыдущий кадр
    prev_gray = gray

# Освобождаем ресурсы
cap.release()
out.release()
print(f"Обработанное видео сохранено по пути: {output_path}")