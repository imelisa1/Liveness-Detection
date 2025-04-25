import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Ayarlar ===
model_path = "balanced_model_v3.keras"
test_dir = "lbp_dataset/test"
img_size = (224, 224)

# === Test verisini yükle ===
test_datagen = ImageDataGenerator(rescale=1./255)
test = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# === Modeli yükle ===
model = load_model(model_path)

# === Tahmin yap ===
probs = model.predict(test)  # Sigmoid çıktı (0-1 arası olasılık)
y_true = test.classes        # Gerçek etiketler (0: Spoof, 1: Live)

# === ROC Eğrisi ve Optimal Threshold Hesabı ===
fpr, tpr, thresholds = roc_curve(y_true, probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"🔍 Optimal Threshold: {optimal_threshold:.4f}")
print(f"📈 AUC Skoru: {roc_auc_score(y_true, probs):.4f}")

# === ROC Eğrisi çizimi ===
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, probs):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', label=f'Optimal Threshold: {optimal_threshold:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Liveness Detection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# === Optimal threshold ile yeni tahminler ve rapor ===
y_pred = (probs > optimal_threshold).astype("int32")

print("\n✅ Classification Report (Optimal Threshold ile):")
print(classification_report(y_true, y_pred, target_names=["Spoof", "Live"]))

print("\n🧱 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
