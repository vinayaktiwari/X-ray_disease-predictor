import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("X-ray disease classifier")
st.text("Please upload your X-ray image in JPEG format")


@st.cache_data()
def load_model():
  model = tf.keras.models.load_model('/app/models')
  # model = tf.keras.models.load_model('C:/Users/DELL/OneDrive/Desktop/ContinousModelDeploy/models')

  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['COVID190','NORMAL','PNEUMONIA','TUBERCULOSIS']

def decode_img(image):
  img = tf.io.decode_image(image, channels=1)  
  img = tf.image.resize(img,[150,150])
  return np.expand_dims(img, axis=0)

uploaded_file = st.file_uploader("Upload your X-ray image...")

if uploaded_file is not None:
    content = uploaded_file.read()

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying X-ray Image', use_column_width=True)
