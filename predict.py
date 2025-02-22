from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Загружаем обученную модель
model = load_model("mobilenetv2_mushroom.h5")

# Список классов
class_names = ['съедобный', 'несъедобный', 'другие_классы_грибов...']  # Замени на свои классы


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
image_path = "(Gemeine_Steinpilz)_Boletus_edulis.jpg"  # Замени на путь к изображению гриба
result = predict_mushroom(image_path)
print(f"🔍 Этот гриб: {result}")