import pandas as pd
import json
import os
import requests
from PIL import Image
import numpy as np


# 1. Загрузка данных из файла JSONL
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Чтение каждой строки как отдельного JSON объекта
            data.append(json.loads(line))

    # Преобразуем в DataFrame
    return pd.DataFrame(data)


# 2. Извлечение URL изображений и меток
def extract_images_and_labels(df):
    images = []
    labels = []

    # Проходим по всем строкам
    for index, row in df.iterrows():
        messages = row['messages']

        for message in messages:
            if 'content' in message:
                content = message['content']

                if isinstance(content, list):  # Проверка на список изображений
                    for image_data in content:
                        if image_data.get('type') == 'image_url':
                            image_url = image_data['image_url']['url']
                            images.append(image_url)

                            # Мы предполагаем, что метка - это информация из других частей сообщения
                            label = "unknown"  # Здесь нужно настроить по твоим данным
                            labels.append(label)

    return images, labels


# 3. Скачивание изображений по URL
def download_images(image_urls, save_dir='downloaded_images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    downloaded_files = []
    for idx, url in enumerate(image_urls):
        try:
            # Загружаем изображение
            img_data = requests.get(url).content
            img_name = f"image_{idx}.jpg"
            img_path = os.path.join(save_dir, img_name)

            # Сохраняем изображение
            with open(img_path, 'wb') as f:
                f.write(img_data)

            downloaded_files.append(img_path)
            print(f"Изображение {img_name} успешно скачано.")
        except Exception as e:
            print(f"Ошибка при скачивании {url}: {e}")

    return downloaded_files


# 4. Предобработка изображений
def preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Нормализация
        images.append(img_array)

    return np.array(images)


# Основная функция загрузки и обработки данных
def load_and_process_data(file_path='_annotations.valid.jsonl'):
    # Загружаем данные
    df = load_data(file_path)

    # Извлекаем изображения и метки
    images, labels = extract_images_and_labels(df)

    # Скачиваем изображения
    downloaded_images = download_images(images)

    # Предобрабатываем изображения
    processed_images = preprocess_images(downloaded_images)

    return processed_images, labels
