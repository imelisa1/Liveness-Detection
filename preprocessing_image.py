import os
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# 📁 Girdi / Çıktı klasörleri
input_dir = "Dataset/test"
output_dir = "preprocessed_dataset/test"
img_size = (224, 224)
margin = 20  # Yüz kutusunu genişletme

detector = MTCNN()

# Live ve Spoof klasörleri için işle
for label in ['Live', 'Spoof']:
    input_path = os.path.join(input_dir, label)
    output_path = os.path.join(output_dir, label)
    os.makedirs(output_path, exist_ok=True)

    for img_name in tqdm(os.listdir(input_path), desc=f"{label} klasörü işleniyor"):
        try:
            img_path = os.path.join(input_path, img_name)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img)

            if faces:
                x, y, w, h = faces[0]['box']
                x = max(x - margin, 0)
                y = max(y - margin, 0)
                w = w + 2 * margin
                h = h + 2 * margin
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, img_size)
                save_path = os.path.join(output_path, img_name)
                Image.fromarray(face).save(save_path)
        except Exception as e:
            print(f"Hata ({img_name}): {e}")
