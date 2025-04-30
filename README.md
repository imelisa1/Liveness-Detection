## 📊 Model Performansı

Bu proje, gerçek zamanlı olarak kamera görüntüsünden alınan yüzün **gerçek (live)** mi yoksa **sahte (spoof)** mi olduğunu tespit etmek amacıyla geliştirilmiş bir Liveness Detection sistemidir.

### 🔍 Kullanılan Model
- **Taban model:** MobileNetV2 (ImageNet ön eğitimli)
- **Giriş boyutu:** 224x224
- **Katmanlar:** GlobalAveragePooling2D + Dropout(0.6) + Dense(128, relu) + Dense(1, sigmoid)
- **Format:** `TFLite` (mobil uyumlu)

### 🧪 Eğitim Detayları
- **Veri kümesi:** `lbp_dataset` (Live/Spoof görüntüler)
- **Ön işleme:** MTCNN ile yüz kırpma ve `224x224` resize
- **Epoch:** 15 (10 + 5 devam eğitimi)
- **Batch size:** 16
- **Optimizer:** Adam (lr=1e-5)
- **Loss:** Binary crossentropy
- **Class weights:** Dengesiz veri dengelendi

### 📈 Değerlendirme Metrikleri

- ✅ **Accuracy:** `81.00%`
- ✅ **AUC Skoru:** `0.8733`
- 🎯 **Optimal Threshold:** `0.6793`  
  *(Model çıktısı bu eşikten büyükse "Live", değilse "Spoof" olarak kabul edilir.)*

  <img src="screenshots/class_weight_training.png" alt="accuracy-loss" width="400"/>
  <img src="screenshots/Ekran görüntüsü 2025-04-25 231312.png" alt="accuracy-loss continued" width="400"/>
  <img src="screenshots/Ekran görüntüsü 2025-04-26 000848.png" alt="ROC curve" width="400"/>

#### 📋 Classification Report

| Sınıf  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Spoof  | 0.83      | 0.82   | 0.83     | 150     |
| Live   | 0.78      | 0.79   | 0.79     | 120     |
| **Accuracy**       |         |         | **0.81** | 270     |
| **Macro Avg**      | 0.80    | 0.81    | 0.81    | 270     |
| **Weighted Avg**   | 0.81    | 0.81    | 0.81    | 270     |

#### 🔢 Confusion Matrix

|               | Predicted: Spoof | Predicted: Live |
|---------------|------------------|-----------------|
| Actual: Spoof | 123              | 27              |
| Actual: Live  | 25               | 95              |

### ScreenShots 

<img src="screenshots/demo.png" alt="demo" width="400"/>


