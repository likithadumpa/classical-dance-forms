import streamlit as st

from keras.models import load_model

from PIL import Image

import requests
from io import BytesIO

import cv2
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np


model = load_model('danceforms.h5')



def detect(frame):
    img=cv2.resize(frame,(256,256))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #scaling to 0 to 1 range 
    if(np.max(img)>1):
            img = img/255.0
    img=np.array([img])
    prediction = model.predict(img)
    prediction = np.argmax(prediction)
    st.write('Prediction Digit:',prediction)
    label=['bharatanatyam', 'kathak', 'kathakali', 'kuchipudi', 'manipuri', 'mohiniyattam',
                   'odissi', 'sattriya']
    return label[prediction]


def main():
    st.title('Identification of Dance Form')
    st.markdown('Just Enter the image url or upload image')
    col1, col2 , col3 = st.columns([8,3,15])
    with col1:
        url = st.text_input("Enter Image Url Here:")
    with col2:
        st.write("Or")
    with col3:
        pic = st.file_uploader("Select Image File")
    
    submit = st.button('Analyze')
    
    if submit:
        col1, col2 = st.columns([8,5])
        with col1:
            st.text("Given Image")
            if url!='':
                response = requests.get(url)
                st.image(Image.open(BytesIO(response.content)))
            else:
                image = Image.open(pic)
                st.image(image)
        
        with col2:
            st.text("Analyzed Dance Form")
            with st.spinner('Screening...'):
                if url!='':
                    response = requests.get(url)
                    pic  = BytesIO(response.content)
                else:
                    pic = pic
            # load and prepare the photograph
            image = Image.open(pic)
            image = img_to_array(image)
            form = detect(image)
            st.write("Prediction Class:",form)

if __name__ == '__main__':
    main()