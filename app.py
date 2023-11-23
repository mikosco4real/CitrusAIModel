import io

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)
# Load the trained model
model = load_model("./bin/citrus_ai_model.h5")  # Replace with the actual path to your trained model file

# Class labels mapping
class_labels = {
    0: "Anthracnose",
    1: "Black Spot",
    2: "Canker",
    3: "Fruit Fly",
    4: "Green Mould",
    5: "Greening",
    6: "Healthy",
    7: "Mechanical Damage",
    8: "Scab",
}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the uploaded image file
        file = request.files["image"]

        # Ensure it's a valid image file
        if not file:
            return jsonify({"error": "No image provided"}), 400

        # Load and preprocess the input image
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0  # Rescale to the same scale as the training data

        # Make predictions
        predictions = model.predict(img)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Get the percentage of accuracy
        accuracy_percentage = predictions[0][predicted_class] * 100

        return jsonify({"category": predicted_label, "accuracy": f"{accuracy_percentage:.2f}%"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
