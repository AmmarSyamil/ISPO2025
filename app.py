import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Load the deep learning models
model_CDT = tf.keras.models.load_model("real2.keras")   # Model for CDT
model_CCT = tf.keras.models.load_model("rubix.keras")   # Model for CCT

# Class labels for each model
class_labels_CDT = ["0", "1", "2", "3", "4", "5"]
class_labels_CCT = ["1", "2", "3"]

# Configure upload directory
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to preprocess the image (convert to grayscale, resize, etc.)
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)     # Add batch dimension
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Ensure both files are provided
    if "cdt_file" not in request.files or "cct_file" not in request.files:
        return jsonify({"error": "Missing CDT or CCT file"}), 400

    cdt_file = request.files["cdt_file"]
    cct_file = request.files["cct_file"]

    if cdt_file.filename == "" or cct_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save CDT file
    cdt_filename = secure_filename(cdt_file.filename)
    cdt_filepath = os.path.join(app.config["UPLOAD_FOLDER"], cdt_filename)
    cdt_file.save(cdt_filepath)

    # Save CCT file
    cct_filename = secure_filename(cct_file.filename)
    cct_filepath = os.path.join(app.config["UPLOAD_FOLDER"], cct_filename)
    cct_file.save(cct_filepath)

    # Preprocess images
    cdt_image = preprocess_image(cdt_filepath)
    cct_image = preprocess_image(cct_filepath)

    # Make predictions using corresponding models
    cdt_prediction = model_CDT.predict(cdt_image)
    cct_prediction = model_CCT.predict(cct_image)

    cdt_predicted_index = np.argmax(cdt_prediction)
    cdt_predicted_class = class_labels_CDT[cdt_predicted_index]
    cdt_probability = float(cdt_prediction[0][cdt_predicted_index])

    cct_predicted_index = np.argmax(cct_prediction)
    cct_predicted_class = class_labels_CCT[cct_predicted_index]
    cct_probability = float(cct_prediction[0][cct_predicted_index])

    # Determine MCI result based on the given criteria
    pass_condition = False
    if cdt_predicted_class == "5" and (cct_predicted_class == "3" or cct_predicted_class == "2"):
        pass_condition = True
    elif cdt_predicted_class == "4" and cct_predicted_class == "3":
        pass_condition = True

    if pass_condition:
        mci_result = "MCI Negative"
        accumulated_probability = round(cdt_probability + cct_probability, 4)
        average_probability = round((cdt_probability + cct_probability) / 2.0, 4)
    else:
        mci_result = "MCI Positive"
        accumulated_probability = round(cdt_probability + cct_probability, 4)
        average_probability = round((cdt_probability + cct_probability) / 2.0, 4)

    return jsonify({
        "cdt_filepath": f"/static/uploads/{cdt_filename}",
        "cct_filepath": f"/static/uploads/{cct_filename}",
        "cdt_predicted_class": cdt_predicted_class,
        "cdt_probability": round(cdt_probability, 4),
        "cct_predicted_class": cct_predicted_class,
        "cct_probability": round(cct_probability, 4),
        "mci_result": mci_result,
        "accumulated_probability": accumulated_probability,
        "average_probability": average_probability
    })

if __name__ == "__main__":
    app.run(debug=True)
