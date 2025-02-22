import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from data_loader import create_data_generator, TRAIN_DIR, VALID_DIR, load_annotations
from model import create_mobilenetv2_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Пути к аннотациям
TRAIN_CSV = os.path.join(TRAIN_DIR, "_annotations.csv")
VALID_CSV = os.path.join(VALID_DIR, "_annotations.csv")

# Загружаем аннотации
train_df = load_annotations(TRAIN_CSV)
valid_df = load_annotations(VALID_CSV)

# Создаем генераторы данных с аугментацией
train_gen = create_data_generator(train_df, augment=True)
valid_gen = create_data_generator(valid_df)

# Определяем количество классов
num_classes = len(train_gen.class_indices)
model = create_mobilenetv2_model(num_classes=num_classes)

print(f"Обнаружено {num_classes} классов: {train_gen.class_indices}")

# Проверка баланса классов
train_counts = Counter(train_gen.classes)
valid_counts = Counter(valid_gen.classes)

print(f"📊 Распределение классов в Train: {dict(train_counts)}")
print(f"📊 Распределение классов в Valid: {dict(valid_counts)}")

# Колбэки для контроля обучения
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Обучение модели
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Сохраняем модель
model.save("mobilenetv2_mushroom.h5")
print("✅ Обучение завершено! Модель сохранена как mobilenetv2_mushroom.h5")

# Визуализация обучения
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()