import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Пути к папкам с изображениями
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')


# Функция для загрузки аннотаций
def load_annotations(csv_path):
    # Загружаем CSV
    annotations = pd.read_csv(csv_path)

    print("📌 Загружен CSV-файл с колонками:", annotations.columns)  # Отладочный вывод

    # Используем 'class' вместо 'label'
    if 'class' not in annotations.columns:
        raise ValueError(f"❌ В файле {csv_path} не найден столбец 'class'. Проверь структуру CSV!")

    # Удаляем строки с пустыми значениями в 'class'
    annotations = annotations.dropna(subset=['class'])

    # Преобразуем метки в строки (если они не в строковом формате)
    annotations['class'] = annotations['class'].astype(str)

    # Переименовываем столбец 'class' в 'label' для удобства работы с Keras
    annotations.rename(columns={'class': 'label'}, inplace=True)

    print("✅ Метки успешно загружены. Пример данных:\n", annotations.head())

    return annotations


# Функция для создания генератора данных
def create_data_generator(directory, csv_path, batch_size=32, target_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1. / 255)  # Нормализация изображений

    annotations = load_annotations(csv_path)

    print("Пример данных из CSV:")
    print(annotations.head())  # Проверяем, какие данные идут в y_col

    generator = datagen.flow_from_dataframe(
        dataframe=annotations,
        directory=directory,
        x_col="filename",
        y_col="label",  # Теперь мы используем 'label', который был 'class'
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",  # Используем 'categorical', так как метки теперь строки
        shuffle=True
    )

    return generator
