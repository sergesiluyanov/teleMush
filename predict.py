from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Загружаем обученную модель
model = load_model("mobilenetv2_mushroom.h5")

# Словарь классов
class_names = ['съедобный', 'несъедобный']


def predict_mushroom(image_path):
    # Загружаем изображение
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Нормализация
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем batch dimension

    # Делаем предсказание
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    return class_names[class_idx]


# Пример использования
image_path = "test_mushroom.jpg"  # Укажи путь к изображению гриба
result = predict_mushroom(image_path)
print(f"Этот гриб: {result}")
