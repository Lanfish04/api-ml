from flask import Flask, request, jsonify
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Path ke model
model_path = '/home/my-compute/code-repository/api-backend-capstone/fresee.h5'

# Load model
model = tf.keras.models.load_model(model_path)

# Max Size
max_size = 2 * 1024 * 1024

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file gambar dari request
    img = request.files.get('img')
    
    if not img:
        return jsonify({
            "status": False,
            "message": "Gambar harus ada"
        }), 400

    if len(img.read()) > max_size:
        return jsonify({"status": False, "message": "Ukuran gambar terlalu besar"}), 400

    try:
        # Baca file gambar ke dalam stream
        img_stream = io.BytesIO(img.read())
        
        # Muat gambar dari stream dan ubah ukurannya
        img_path = load_img(img_stream, target_size=(150, 150))
        
        # Konversi gambar ke array
        img_array = img_to_array(img_path)
        
        # Normalisasi gambar (skala 0-1)
        img_array = img_array / 255.0
        
        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediksi dengan model
        prediction = model.predict(img_array)
        
        # Interpretasi hasil prediksi
        threshold = 0.5  # Ambang batas biner
        if prediction[0] <= threshold:
            return jsonify({
                "status": True,
                "message": "Buah atau sayur segar",
                "probability": float(prediction[0][0])
            })
        else:
            return jsonify({
                "status": False,
                "message": "Buah atau sayu tidak segar",
                "probability": float(prediction[0][0])
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4500)