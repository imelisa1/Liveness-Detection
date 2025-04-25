import tensorflow as tf
model = tf.keras.models.load_model("balanced_model_v3.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("liveness_model.tflite", "wb") as f:
    f.write(tflite_model)
