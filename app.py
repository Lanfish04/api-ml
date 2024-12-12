from flask import Flask, request, jsonify
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Path ke model
model_path = 'fresee.h5'

# Load model
model = tf.keras.models.load_model(model_path)

# Max Size
max_size = 2 * 1024 * 1024

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files.get('img')
    
    if not img:
        return jsonify({
            "status": False,
            "message": "Gambar harus ada"
        }), 400

    if not img.content_type.startswith('image/'):
        return jsonify({
            "status": False,
            "message": "File bukan gambar"
        }), 400

    if len(img.read()) > max_size:
        return jsonify({"status": False, "message": "Ukuran gambar terlalu besar"}), 400

    try:
        # Reset posisi stream setelah membaca ukuran
        img.seek(0)
        
        # Baca file gambar ke dalam stream
        img_stream = io.BytesIO(img.read())

        # Pastikan ukuran input cocok dengan model
        input_shape = (160, 160)  # Ubah sesuai model Anda
        img_path = load_img(img_stream, target_size=input_shape)

        # Konversi gambar ke array
        img_array = img_to_array(img_path)

        # Normalisasi gambar (skala 0-1)
        img_array = img_array / 255.0

        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)

        # Debugging ukuran input
        print(f"Input shape before prediction: {img_array.shape}")

        # Prediksi dengan model
        prediction = model.predict(img_array)

        # Interpretasi hasil prediksi
        threshold = 0.5  # Ambang batas biner
        if prediction[0][0] <= threshold:
            return jsonify({
                "status": True,
                "message": "Produk ini Segar!",
                "probability": float(prediction[0][0])
            })
        else:
            return jsonify({
                "status": False,
                "message": "Produk ini Tidak Segar!",
                "probability": float(prediction[0][0])
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4500)
