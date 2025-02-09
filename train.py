from data_loader import create_data_generator, TRAIN_DIR, VALID_DIR
from model import create_mobilenetv2_model
import os

# Пути к аннотациям
TRAIN_CSV = os.path.join(TRAIN_DIR, "_annotations.csv")
VALID_CSV = os.path.join(VALID_DIR, "_annotations.csv")

# Создаем генераторы данных
train_gen = create_data_generator(TRAIN_DIR, TRAIN_CSV)
valid_gen = create_data_generator(VALID_DIR, VALID_CSV)

# Определяем количество классов
num_classes = len(train_gen.class_indices)
model = create_mobilenetv2_model(num_classes=num_classes)  # Передаём правильное число классов
print(f"Обнаружено {num_classes} классов: {train_gen.class_indices}")
print(f"Обнаружено {len(train_gen.class_indices)} классов: {train_gen.class_indices}")


# Создаем модель с правильным количеством классов
model = create_mobilenetv2_model(num_classes=num_classes)

train_classes = set(train_gen.class_indices.keys())
valid_classes = set(valid_gen.class_indices.keys())

missing_classes = train_classes - valid_classes
if missing_classes:
    print(f"⚠️ ВНИМАНИЕ! Эти классы отсутствуют в valid: {missing_classes}")

batch = next(iter(train_gen))  # Получаем один batch данных
images, labels = batch

print(f"Тип images: {type(images)}, Форма: {images.shape}")
print(f"Тип labels: {type(labels)}, Форма: {labels.shape}")
print(f"Пример labels:\n{labels[:5]}")

print(f"Количество классов в train_gen: {len(train_gen.class_indices)}")
print(f"Количество классов в valid_gen: {len(valid_gen.class_indices)}")
print(f"train_gen.class_indices: {train_gen.class_indices}")
print(f"valid_gen.class_indices: {valid_gen.class_indices}")

# Обучаем модель
model.fit(train_gen, validation_data=valid_gen, epochs=10)

# Сохраняем модель
model.save("mobilenetv2_mushroom.h5")

print("✅ Обучение завершено! Модель сохранена как mobilenetv2_mushroom.h5")
