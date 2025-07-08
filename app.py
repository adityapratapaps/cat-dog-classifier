import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load the trained model
model = load_model("cat_dog_model.h5")

# Predict function
def predict(img):
    img = img.resize((150, 150))
    img_array = keras_image.img_to_array(img) / 255.
    img_tensor = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_tensor)[0][0]
    return "🐶 Dog" if pred > 0.5 else "🐱 Cat"

# Streamlit UI
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and I'll guess what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    label = predict(img)
    st.success(f"Prediction: {label}")
