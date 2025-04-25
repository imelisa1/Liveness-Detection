import os
from skimage.feature import local_binary_pattern
from PIL import Image
import numpy as np
from tqdm import tqdm

# LBP parametreleri
radius = 3
n_points = 8 * radius
method = 'uniform'

input_dir = "preprocessed_dataset/test"
output_dir = "lbp_dataset/test"
os.makedirs(output_dir, exist_ok=True)
img_size = (224, 224)

for label in ['Live', 'Spoof']:
    in_path = os.path.join(input_dir, label)
    out_path = os.path.join(output_dir, label)
    os.makedirs(out_path, exist_ok=True)

    for img_name in tqdm(os.listdir(in_path), desc=f"{label} LBP işleniyor"):
        try:
            img_path = os.path.join(in_path, img_name)
            img = Image.open(img_path).convert('L')  # grayscale
            img = img.resize(img_size)
            img_np = np.array(img)

            lbp = local_binary_pattern(img_np, n_points, radius, method)
            lbp_norm = (lbp / lbp.max() * 255).astype(np.uint8)

            Image.fromarray(lbp_norm).save(os.path.join(out_path, img_name))
        except Exception as e:
            print(f"Hata: {img_name} - {e}")
