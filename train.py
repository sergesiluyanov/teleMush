from data_loader import load_and_process_data
from sklearn.model_selection import train_test_split
from model import create_model, train_model


def main():
    # Загружаем и обрабатываем данные
    processed_images, labels = load_and_process_data('_annotations.valid.jsonl')

    # Разделяем данные на обучающие и тестовые
    X_train, X_val, y_train, y_val = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

    # Создаем и обучаем модель
    model = train_model(X_train, y_train, X_val, y_val)

    # Оценка точности модели
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Точность модели на тестовых данных: {accuracy:.4f}")


if __name__ == "__main__":
    main()

