import streamlit as st
import joblib
import pandas as pd
import zipfile
import pickle

# Combine split zip files
def combine_split_zips(output_zipfile, *input_parts):
    with open(output_zipfile, 'wb') as output_file:
        for part in input_parts:
            with open(part, 'rb') as part_file:
                output_file.write(part_file.read())

# Load models
@st.cache_resource
def load_models():
    # Combine split zip files into a single zip
    combine_split_zips('heartbeat_model.zip', 'heartbeat_model.z01', 'heartbeat_model.z02')

    # Extract the combined zip file
    with zipfile.ZipFile('heartbeat_model.zip', 'r') as zip_ref:
        zip_ref.extractall()

    # Load the models from the extracted files
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    with open('Abnormal_model.pkl', 'rb') as f:
        Abnormal_model = pickle.load(f)

    return xgb_model, Abnormal_model

# Load the models
xgb_model, Abnormal_model = load_models()

# Streamlit app title
st.title("Heartbeat Classification")

# Create a file uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file containing 100 values", type=['csv'])

# When a file is uploaded
if uploaded_file is not None:
    # Read the CSV file
    input_data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(input_data)

    # Clean column names: strip quotes and whitespaces
    input_data.columns = input_data.columns.str.replace('"', '').str.strip()

    # Check if the DataFrame has the correct number of columns
    if input_data.shape[1] == 100:  # Expecting 100 values (columns)
        # Ensure the column names match expected feature names
        expected_feature_names = [str(i) for i in range(100)]  # Adjust according to your model's training
        input_data.columns = expected_feature_names  # Rename if necessary

        # Make predictions using the loaded model (xgb_model or Abnormal_model)
        predictions = xgb_model.predict(input_data)

        # Convert predictions to DataFrame for better display
        predictions_df = pd.DataFrame(predictions, columns=["Prediction"])

        # Map predictions to labels (0: Normal, 1: Abnormal)
        predictions_df["Label"] = predictions_df["Prediction"].map({0: "Normal", 1: "Abnormal"})

        # Display predictions
        st.write("Predictions (0: Normal, 1: Abnormal):")
        st.write(predictions_df)
    else:
        st.error("The uploaded CSV must contain exactly 100 values (columns). Please check your file.")
