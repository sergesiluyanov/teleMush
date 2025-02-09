import pandas as pd

# Загружаем valid аннотации
df_valid = pd.read_csv("dataset/valid/annotations.csv")

# Выводим список уникальных классов
unique_classes = df_valid['class'].unique()
num_classes = len(unique_classes)

print(f"📌 В `valid` сейчас {num_classes} классов.")
print("📌 Список классов в valid:")
print(df_valid['class'].value_counts())
