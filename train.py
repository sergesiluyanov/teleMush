from data_loader import create_data_generator, TRAIN_DIR, VALID_DIR
from model import create_mobilenetv2_model
import os

# Пути к аннотациям
TRAIN_CSV = os.path.join(TRAIN_DIR, "_annotations.csv")
VALID_CSV = os.path.join(VALID_DIR, "_annotations.csv")

# Создаем генераторы данных
train_gen = create_data_generator(TRAIN_DIR, TRAIN_CSV)
valid_gen = create_data_generator(VALID_DIR, VALID_CSV)

# Создаем модель MobileNetV2
model = create_mobilenetv2_model()

# Обучаем модель
model.fit(train_gen, validation_data=valid_gen, epochs=10)

# Сохраняем модель
model.save("mobilenetv2_mushroom.h5")

print("✅ Обучение завершено! Модель сохранена как mobilenetv2_mushroom.h5")
