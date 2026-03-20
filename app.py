from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import json

app = Flask(__name__)

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "cow_breed_prototype3.h5"
DATASET_DIR = "dataset/balanced_breed"
EXCEL_PATH ="cattle_excel_sheet.xlsx"
IMG_SIZE = (128, 128)


UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load class names dynamically from dataset
import json

with open("class_names.json") as f:
    class_names = json.load(f)
print(class_names)

# Load Excel info into DataFrame
breed_info_df = pd.read_excel(EXCEL_PATH)

# ----------------------------
# Helper functions
# ----------------------------
def process_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_breed_info(breed_name):
    row = breed_info_df[breed_info_df["Breed"].str.lower() == breed_name.lower()]
    if not row.empty:
        return row.iloc[0].to_dict()
    return None

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    files = request.files
    preds_all = []
    saved_paths = []

    for view in ["front", "back", "side"]:
        if view in files and files[view].filename != "":
            filename = files[view].filename
            path = os.path.join(UPLOAD_FOLDER, filename)

            # Save file in uploads folder
            files[view].save(path)
            saved_paths.append(path)

            # Process image
            img = process_image(path)
            preds = model.predict(img, verbose=0)[0]
            preds_all.append(preds)

    if not preds_all:
        return jsonify({"error": "No image uploaded!"}), 400

    # Average predictions
    avg_preds = np.mean(preds_all, axis=0)
    top5_idx = avg_preds.argsort()[-5:][::-1]
    top5 = {class_names[i]: float(avg_preds[i]) for i in top5_idx}
    best_idx = np.argmax(avg_preds)
    breed_name = class_names[best_idx]

    return jsonify({
        "breed": breed_name,
        "confidence": float(avg_preds[best_idx]),
        "top5": top5,
        "uploaded": saved_paths
    })

@app.route("/results")
def results():
    breed = request.args.get("breed")
    confidence = float(request.args.get("confidence"))
    top5 = json.loads(request.args.get("top5"))

    breed_info = get_breed_info(breed)

    return render_template(
        "results.html",
        breed=breed,
        confidence=confidence,
        top5=top5,
        breed_info=breed_info
    )

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)