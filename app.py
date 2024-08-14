import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps 


model_dir = os.path.join(os.path.dirname(os.getcwd()), r'Plant Disease Classification')
model_path = os.path.join(model_dir, 'Plant_model.keras')


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.write("""
         # Flower Classification
         """)

file = st.file_uploader("Please upload an flower image", type = ['jpg', 'png'])

def import_and_predict(image_data, model):

    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text('Please upload an image file')
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image, model)
    class_names = ['Pepper__bell___Bacterial_spot',
               'Pepper__bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Tomato_Bacterial_spot',
               'Tomato_Early_blight',
               'Tomato_Late_blight',
               'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus',
               'Tomato_healthy']
    
    string = "This image most likely is: "+ class_names[np.argmax(predictions)]
    st.success(string)