from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('./bin/citrus_ai_model.h5')

# Load and preprocess the input image
img_path = '/Users/okolomichael/Desktop/Black-Spot-Citrus.jpg'  # Replace with the path to your input image
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.0  # Rescale to the same scale as the training data

# Make predictions
predictions = model.predict(img)

# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)[0]

# Map the class index to the actual class label (e.g., using a dictionary)
class_labels = {
    0: 'Anthracnose',
    1: 'Black Spot',
    2: 'Canker',
    3: 'Fruit Fly',
    4: 'Green Mould',
    5: 'Greening',
    6: 'Healthy',
    7: 'Mechanical Damage',
    8: 'Scab'
}

predicted_label = class_labels[predicted_class]

print(f"Predicted class: {predicted_label}")
