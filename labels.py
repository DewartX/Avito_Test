import os
import csv
from paddleocr import TextLineOrientationClassification

# модель классификации ориентации текстовой строки (0 / 180 градусов)
clf = TextLineOrientationClassification(model_name="PP-LCNet_x1_0_textline_ori")

test_dir = "test_set"
batch_size = 32

# картинки из папки с тестом
img_names = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

results = []

for i in range(0, len(img_names), batch_size):
    batch_names = img_names[i:i + batch_size]
    batch_paths = [os.path.join(test_dir, name) for name in batch_names]

    preds = clf.predict(batch_paths, batch_size=batch_size)

    for name, pred in zip(batch_names, preds):
        top_label = pred["label_names"][0]   # предсказанный класс
        top_score = float(pred["scores"][0])  # уверенность в этом классе

        # переводим в вероятность p_180
        if top_label == "180_degree":
            p180 = top_score
        else:
            p180 = 1.0 - top_score

        results.append((name, round(p180, 6)))

    if i % 1000 == 0:
        print(f"обработано {i}/{len(img_names)}")

# сохраняем в нужном формате: image_id, p_180
with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "p_180"])
    writer.writerows(results)

print("готово:", len(results), "строк записано в submission.csv")