import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Title for the Streamlit app
st.title("Text Classification Model Testing")

# Sidebar for user actions
st.sidebar.header("Actions")
action = st.sidebar.selectbox(
    "Choose an action",
    ("Load Model", "Test Model Prediction")
)

# Load Model Section
if action == "Load Model":
    try:
        st.subheader("Load the Model")
        model = load_model('text_classification.h5')
        st.success("Model loaded successfully!")
        st.write("Model Summary:")
        model.summary(print_fn=lambda x: st.text(x))  # Display model summary in the app
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Test Model Prediction Section
elif action == "Test Model Prediction":
    try:
        st.subheader("Test Model Prediction")

        # Option to load the model
        st.write("Loading model...")
        model = load_model('text_classification.h5')
        st.success("Model loaded successfully!")

        # Generate dummy input data
        st.write("Generating dummy input data...")
        dummy_data = np.random.randint(0, 1000, size=(1, 100))
        st.write("Dummy input data shape:", dummy_data.shape)

        # Make prediction
        st.write("Making prediction...")
        prediction = model.predict(dummy_data)
        st.write("Prediction Output Shape:", prediction.shape)
        st.write("Prediction Value:", prediction)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
