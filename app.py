from flask.helpers import send_file
from jinja2 import Template

import matplotlib.pyplot as plt
from os import path
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

# Feature Scaling

import pandas as pd
import numpy as np
import matplotlib
import numpy as np
from flask import Flask, request, jsonify, render_template

import pandas as pd

# coding=utf-8
import sys

import glob
import re
import cv2
from  PIL import Image, ImageOps
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import io


matplotlib.use('Agg')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#------------------------------ Saving dataset---------------------------------
# this is the path to save dataset for preprocessing
pathfordataset = "static/data-preprocess/"
pathfordatasetNew = "data-preprocess/new/"  
 
app.config['DFPr'] = pathfordataset
app.config['DFPrNew'] = pathfordatasetNew
#------------------------------ Saving dataset for Linear regression-------------------------------------------
# this is the path to save dataset for single variable LR
pathforonevarLR = "static/Regression/onevarLR"
pathforonevarLRplot = "Regression/onevarLR/plot"
app.config['LR1VAR'] = pathforonevarLR
app.config['LR1VARplot'] = pathforonevarLRplot

#------------------------------ Saving image for K means-------------------------------------------
# this is the path to save figure of K menas
pathforelbowplot = "kmeans/plot"
#pathforonevarLRplot = "Regression/onevarLR/plot"
#app.config['LR1VAR'] = pathforonevarLR
app.config['elbowplot'] = pathforelbowplot
#print(app.config['elbowplot'])

# for index page
#------------------------------ Launcing undex page-------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')



#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
#------------------------------Artificial Neural network-------------------------------------------


@app.route('/ann')
def ann():
    return render_template('/ann/ann.html')


#-----------------------Digit Recognition---------------------------------------------
model_digit = load_model("static/data-preprocess/model/MNISTANN.h5")

def import_and_predict(image_data):
  
  image_resized = cv2.resize(image_data, (28, 28)) 
   
  prediction = model_digit.predict(image_resized.reshape(1,784))
  print('Prediction Score:\n',prediction[0])
  thresholded = (prediction>0.5)*1
  print('\nThresholded Score:\n',thresholded[0])
  print('\nPredicted Digit:',np.where(thresholded == 1)[1][0])
  digit = np.where(thresholded == 1)[1][0]
  #st.image(image_data, use_column_width=True)
  return digit



@app.route('/ann/digit/digit')
def digit():
    return render_template('/ann/digit/digit.html')


@app.route('/ann/digit/digit',  methods=['GET', 'POST'])
def digit1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        print(input_image)
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict(image)

        

        return render_template('/ann/digit/digitoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )



#-------------------Flask Application--------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
    
# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
app.config["CACHE_TYPE"] = "null"




