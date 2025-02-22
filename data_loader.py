import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Пути
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
CSV_PATH = os.path.join(BASE_DIR, "annotations.csv")  # Файл с разметкой
IMAGE_DIR = BASE_DIR  # Изображения находятся в той же папке

# Доли разбиения TRAIN/VALID/TEST
TRAIN_SPLIT = 0.7  # 70% - обучение
VALID_SPLIT = 0.15  # 15% - валидация
TEST_SPLIT = 0.15  # 15% - тестирование

# Функция загрузки CSV
def load_annotations(csv_path):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')  # Читаем CSV с правильной кодировкой

    if 'filename' not in df.columns or 'class' not in df.columns:
        raise ValueError("❌ В CSV должны быть столбцы 'filename' и 'class'!")

    df.dropna(subset=['class'], inplace=True)  # Убираем пустые значения
    df['class'] = df['class'].astype(str).str.strip()  # Приводим классы к строковому типу без пробелов
    df.rename(columns={'class': 'label'}, inplace=True)  # Переименовываем в label
    return df

# Функция разбиения датасета
def split_dataset(df):
    train, temp = train_test_split(df, test_size=1 - TRAIN_SPLIT, stratify=df['label'], random_state=42)
    valid, test = train_test_split(temp, test_size=TEST_SPLIT / (VALID_SPLIT + TEST_SPLIT), stratify=temp['label'],
                                   random_state=42)
    return train, valid, test

# Функция создания генератора данных
def create_data_generator(df, batch_size=32, target_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=IMAGE_DIR,  # Папка с изображениями
        x_col="filename",
        y_col="label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

# Главная функция загрузки
def load_data():
    df = load_annotations(CSV_PATH)  # Загружаем разметку
    train_df, valid_df, test_df = split_dataset(df)  # Разбиваем на train, valid, test

    train_gen = create_data_generator(train_df)
    valid_gen = create_data_generator(valid_df)
    test_gen = create_data_generator(test_df)

    return train_gen, valid_gen, test_gen

# Запуск
if __name__ == "__main__":
    train_gen, valid_gen, test_gen = load_data()
    print("✅ Данные загружены. Размеры:")
    print(f"Train: {len(train_gen.filenames)} изображений")
    print(f"Valid: {len(valid_gen.filenames)} изображений")
    print(f"Test: {len(test_gen.filenames)} изображений")