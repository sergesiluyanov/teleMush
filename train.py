# model.py
import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2 класса: съедобный/несъедобный
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Функция для обучения модели
def train_model(X_train, y_train, X_val, y_val):
    model = create_model()  # Создаем модель
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))  # Обучаем модель
    model.save('mushroom_model_test.h5')  # Сохраняем обученную модель
    return model
