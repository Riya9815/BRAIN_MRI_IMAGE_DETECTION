from flask import Flask,render_template,url_for,request
import pandas as pd 

import pickle
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


model=load_model('mod_vgg16.h5')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    s = str(request.form['image-path'])
    
    img=image.load_img(s,target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)
    classes=model.predict(img_data)
    if classes[0][0]>classes[0][1]:
        return render_template('index.html',pred="No Brain Tumor Detected")
    else:
        return render_template('index.html',pred="Brain Tumor Detected")



if __name__ == '__main__':
	app.run(debug=True)
