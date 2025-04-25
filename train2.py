from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

# === Ayarlar ===
img_size = (224, 224)
batch_size = 16
epochs = 5  # sadece ek epoch sayısı

train_dir = "lbp_dataset/train"
val_dir = "lbp_dataset/test"

# === Image Loaders ===
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# === Class Weight Hesapla ===
y_train = train.classes
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# === Kaldığın modeli yükle ===
model = load_model("balanced_model_v2.keras")

# === Yeni checkpoint (eskisini ezmemesi için farklı isim) ===
checkpoint = ModelCheckpoint("balanced_model_v3.keras", monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Eğitime devam et ===
history2 = model.fit(
    train,
    validation_data=val,
    epochs=epochs,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop]
)

# === Grafik ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history2.history['accuracy'], label='Train')
plt.plot(history2.history['val_accuracy'], label='Val')
plt.title("Continued Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history2.history['loss'], label='Train')
plt.plot(history2.history['val_loss'], label='Val')
plt.title("Continued Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_continued2.png")
plt.show()
