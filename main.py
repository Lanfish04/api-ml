from tensorflow.keras.applications import efficientnet
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

model_path = '/home/my-compute/code-repository/api-backend-capstone/fresee.h5'

model = tf.keras.models.load_model(model_path)

# model.summary()

img_path = "/home/my-compute/code-repository/api-backend-capstone/try-model/rotternapple.png"

img = load_img(img_path, target_size=(150, 150))  # Sesuaikan ukuran dengan input model
img_array = img_to_array(img)  # Konversi ke array
img_array = img_array / 255.0  # Normalisasi (0-255 menjadi 0-1)
img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

# Prediksi gambar
prediction = model.predict(img_array)

# Interpretasi hasil
threshold = 0.5  # Gunakan ambang batas (default 0.5 untuk biner)
if prediction[0] > threshold:
    print("Prediksi: Tidak segar")
else:
    print("Prediksi: Segar")

# Menampilkan probabilitas
print(f"Probabilitas: {prediction[0][0]:.2f}")