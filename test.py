from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# === Test veri hazırlığı ===
test_datagen = ImageDataGenerator(rescale=1./255)

test = test_datagen.flow_from_directory(
    "lbp_dataset/test",  # test klasörünü buraya koy
    target_size=(224, 224),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# === Model yükle ===
model = load_model("best_model.keras")  # veya .h5

# === Tahmin ve değerlendirme ===
preds = model.predict(test)
y_pred = (preds > 0.5).astype("int32")
y_true = test.classes

print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Spoof', 'Live']))

print("\n🧱 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
