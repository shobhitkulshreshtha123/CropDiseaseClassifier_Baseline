###############################################################
# Plant disease detection 
# API V2
# Version 1.1
# API Main Script
###############################################################

import imageprocess
import predictor
import pickle
import os
from flask import Flask, request, render_template
import numpy as np
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load models
apple_model = pickle.load(open('models/Applemodel.sav', 'rb'))
corn_model = pickle.load(open('models/Cornmodel.sav', 'rb'))
grapes_model = pickle.load(open('models/Grapesmodel.sav', 'rb'))
potato_model = pickle.load(open('models/Potatomodel.sav', 'rb'))
tomato_model = pickle.load(open('models/Tomatomodel.sav', 'rb'))

@app.route("/")
def home():
    version = "1.1"
    return render_template('index.html', version1=version)

@app.route("/predict", methods=['POST'])
def submit():
    imagefile = request.files["data_file"]
    image_bytes = imagefile.read()
    dname = request.form.get('Name')
    response = str(dname)[0]
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    f_vector = imageprocess.feature_extractor(img)

    if response == 'n':
        return render_template("prediction.html", prediction="Invalid Plant Type Selected", confidence=0, image_url="#", bar_color="gray")

    # Convert image to base64 for preview
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    image_url = f"data:image/png;base64,{img_str}"

    # Determine model and feature vector
    if response == 'a':
        p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['red_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['green_std'], f_vector['f4'], f_vector['f6'], f_vector['f7']]
        res, conf = predictor.apple_p(p_vector, apple_model)

    elif response == 'c':
        p_vector = [f_vector['red_mean'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'], f_vector['blue_std'], f_vector['f7'], f_vector['f8']]
        res, conf = predictor.corn_p(p_vector, corn_model)

    elif response == 'g':
        p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['red_mean'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'], f_vector['green_std'], f_vector['blue_std'], f_vector['f4'], f_vector['f5'], f_vector['f6'], f_vector['f7'], f_vector['f8']]
        res, conf = predictor.grapes_p(p_vector, grapes_model)

    elif response == 'p':
        p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'], f_vector['green_std'], f_vector['blue_std'], f_vector['f4'], f_vector['f5'], f_vector['f7'], f_vector['f8']]
        res, conf = predictor.potato_p(p_vector, potato_model)

    elif response == 't':
        del f_vector['f1']
        p_vector = list(f_vector.values())
        res, conf = predictor.tomato_p(p_vector, tomato_model)

    else:
        res = "Unknown"
        conf = 0

    # Color logic for confidence
    if conf >= 80:
        bar_color = 'limegreen'
    elif conf >= 50:
        bar_color = 'gold'
    else:
        bar_color = 'crimson'

    return render_template("prediction.html", prediction=res, confidence=conf, image_url=image_url, bar_color=bar_color, shap_img_url=None)

if __name__ == "__main__":
    app.run(port=5000)
