import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("Brain MRI Metastasis Segmentation")

uploaded_image = st.file_uploader("Choose a brain MRI image", type="tif")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    payload = {"file": uploaded_image.getvalue()}
    response = requests.post("http://localhost:8000/predict", files=payload)
    
    prediction_mask = np.array(response.json()['segmentation'])
    st.image(prediction_mask, caption="Predicted Segmentation Mask", use_column_width=True)
