from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Загружаем обученную модель
model = load_model("mobilenetv2_mushroom.h5")

# Список классов
class_names = ['Опенок осенний', 'Сыроежка', 'Мухомор вонючий', 'Моховик']  # Замени на свои классы


def predict_mushroom(image_path):
    # Загружаем изображение
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Нормализация
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем batch размерность

    # Делаем предсказание
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)  # Индекс наиболее вероятного класса

    return class_names[class_idx]


# Пример использования
image_path = "Снимок экрана 2025-02-09 в 22.42.51.png"  # Замени на путь к изображению гриба
result = predict_mushroom(image_path)
print(f"🔍 Этот гриб: {result}")