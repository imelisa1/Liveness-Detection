import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

# === Ayarlar ===
img_size = (224, 224)
batch_size = 16
epochs = 10

train_dir = "lbp_dataset/train"
val_dir = "lbp_dataset/test"

# === Image Loader ===
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary'
)

val = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary'
)

# === Class Weight Hesapla ===
y_train = train.classes
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# === Model: MobileNetV2 + Yüksek Dropout ===
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)  # ⬆️ Dropout yükseltildi
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("balanced_model.keras", monitor='val_accuracy', save_best_only=True, mode='max')

# === Eğitim ===
history = model.fit(
    train,
    validation_data=val,
    epochs=epochs,
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint]
)

# === Grafik ===
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("class_weight_training.png")
plt.show()
