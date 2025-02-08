import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Пути к папкам с изображениями
BASE_DIR = "dataset"  # Укажи путь к твоему датасету
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')


# Функция для загрузки аннотаций
def load_annotations(csv_path):
    annotations = pd.read_csv(csv_path)

    # Преобразуем метки в числа (съедобный = 0, несъедобный = 1)
    label_map = {'съедобный': 0, 'несъедобный': 1}
    annotations['label'] = annotations['label'].map(label_map)

    return annotations


# Функция для создания генератора данных
def create_data_generator(directory, csv_path, batch_size=32, target_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1. / 255)  # Нормализация изображений

    annotations = load_annotations(csv_path)

    generator = datagen.flow_from_dataframe(
        dataframe=annotations,
        directory=directory,
        x_col="filename",
        y_col="label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True
    )

    return generator
