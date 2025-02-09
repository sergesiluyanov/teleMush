import os
import pandas as pd
import shutil

train_dir = "dataset/train"
valid_dir = "dataset/valid"

# Загружаем аннотации
df_train = pd.read_csv("dataset/train/_annotations.csv")
df_valid = pd.read_csv("dataset/valid/_annotations.csv")

# Классы, которые отсутствуют в valid
missing_classes = {'----------------- ----------', '-------- ----------- ----------',
                   '-------- -------- ------', '------- ------- ----------',
                   '----------- ---------------- ----------', 'Hydnellum peckii ----------',
                   '--------- ----------- ----------', '-------- ---------- ----------',
                   '----------- ------------ ----------'}

print(f"⚠️ Эти классы отсутствуют в valid: {missing_classes}")

# Перемещаем все изображения недостающих классов в valid
for missing_class in missing_classes:
    class_images = df_train[df_train['class'] == missing_class]

    for _, row in class_images.iterrows():
        filename = row['filename']
        src_path = os.path.join(train_dir, filename)
        dest_path = os.path.join(valid_dir, filename)

        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)

# Обновляем valid CSV
df_valid = pd.concat([df_valid, df_train[df_train['class'].isin(missing_classes)]], ignore_index=True)
df_valid.to_csv("dataset/valid/annotations.csv", index=False)

print("✅ Недостающие классы добавлены в valid!")
