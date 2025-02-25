import asyncio
import logging
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType
from aiogram.utils import executor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Настройки бота
TOKEN = "8100675753:AAEdyuvkFJgX7i3FGU7XTeHUlTq7Bh0TnRk"
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# Загружаем модель
MODEL_PATH = "mobilenetv2_mushroom.h5"
model = load_model(MODEL_PATH)

# Классы грибов (пример, обнови в соответствии с твоими классами)
CLASS_NAMES = ["Опенок осенний", "Моховик", "Сыроежка", "Мухомор вонючий"]
EDIBILITY_MAP = {"Опенок осенний": "✅ Съедобный", "Моховик": "✅ Съедобный",
                 "Сыроежка": "✅ Съедобный", "Мухомор вонючий": "❌ Ядовитый"}

# Логирование
logging.basicConfig(level=logging.INFO)


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.reply("Привет! Отправь мне фото гриба, и я попробую определить его вид и съедобность! 🍄")


@dp.message_handler(content_types=ContentType.PHOTO)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]  # Берем изображение наивысшего качества
    photo_bytes = await photo.download(destination=io.BytesIO())  # Скачиваем изображение

    # Открываем изображение и подготавливаем его для модели
    img = Image.open(photo_bytes)
    img = img.resize((224, 224))  # Изменяем размер
    img_array = image.img_to_array(img) / 255.0  # Нормализация
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем batch-измерение

    # Предсказание
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]  # Получаем класс с максимальной вероятностью
    edibility = EDIBILITY_MAP.get(predicted_class, "Неизвестно")

    # Отправляем результат пользователю
    await message.reply(f"🔍 Гриб: {predicted_class}\n🍽 Съедобность: {edibility}")


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
