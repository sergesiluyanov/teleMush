import asyncio
import logging
import numpy as np
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ContentType
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Настройки бота
TOKEN = "8100675753:AAEdyuvkFJgX7i3FGU7XTeHUlTq7Bh0TnRk"
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Загружаем модель
MODEL_PATH = "mobilenetv2_mushroom.h5"
model = load_model(MODEL_PATH)
model.compile()  # Добавил, чтобы не было предупреждения

# Классы грибов
CLASS_NAMES = ["Опенок осенний", "Моховик", "Сыроежка", "Мухомор вонючий"]
EDIBILITY_MAP = {
    "Опенок осенний": "✅ Съедобный",
    "Моховик": "✅ Съедобный",
    "Сыроежка": "✅ Съедобный",
    "Мухомор вонючий": "❌ Ядовитый"
}

# Логирование
logging.basicConfig(level=logging.INFO)


async def start_handler(message: types.Message):
    await message.reply("Привет! Отправь мне фото гриба, и я попробую определить его вид и съедобность! 🍄")


async def handle_photo(message: types.Message):
    try:
        photo = message.photo[-1]  # Берем изображение наивысшего качества
        photo_bytes = io.BytesIO()

        # !!! Используем bot.download вместо photo.download
        await bot.download(photo, destination=photo_bytes)
        photo_bytes.seek(0)  # Перематываем в начало

        # Открываем изображение и подготавливаем для модели
        img = Image.open(photo_bytes).convert("RGB")
        img = img.resize((224, 224))  # Изменяем размер
        img_array = image.img_to_array(img) / 255.0  # Нормализация
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем batch-измерение

        # Предсказание
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]  # Класс с максимальной вероятностью
        edibility = EDIBILITY_MAP.get(predicted_class, "❓ Неизвестно")

        # Отправляем результат
        await message.reply(f"🔍 Гриб: {predicted_class}\n🍽 Съедобность: {edibility}")

    except Exception as e:
        logging.error(f"Ошибка обработки фото: {e}")
        await message.reply("❌ Произошла ошибка при обработке фото. Попробуйте еще раз.")


# Регистрируем хэндлеры
dp.message.register(start_handler, F.text == "/start")
dp.message.register(handle_photo, F.photo)


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
