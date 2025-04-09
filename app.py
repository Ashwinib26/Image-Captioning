from flask import Flask, render_template, request
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploaded"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and tokenizer
model = load_model("caption_model.keras")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Define vocab size and max_length
vocab_size = len(tokenizer.word_index) + 1
max_length = 34  # Based on your earlier outputs or captions.max(). You can also recalculate from train/test captions.

# Load DenseNet201 for feature extraction
cnn = DenseNet201(weights="imagenet", include_top=False, pooling="avg")
fe = Model(inputs=cnn.input, outputs=cnn.output)

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    feature = fe.predict(img, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, feature, tokenizer, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += " " + word
    return in_text.replace('startseq', '').replace('endseq', '').strip()

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            feature = extract_features(file_path)
            caption = predict_caption(model, feature, tokenizer, max_length)

    return render_template("index.html", caption=caption, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
