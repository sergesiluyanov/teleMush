# model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def create_mobilenetv2_model(input_shape=(224, 224, 3), num_classes=2):
    # Загружаем предобученную модель MobileNetV2 без верхних слоев
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Замораживаем веса предобученной модели (чтобы сначала обучать только новые слои)
    base_model.trainable = False

    # Добавляем свои слои
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Усреднение признаков
        layers.Dense(128, activation='relu'),  # Полносвязный слой
        layers.Dropout(0.3),  # Dropout для борьбы с переобучением
        layers.Dense(num_classes, activation='softmax')  # Выходной слой
    ])

    # Компиляция модели
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Используем sparse, если метки - числа (0 и 1)
                  metrics=['accuracy'])

    return model
