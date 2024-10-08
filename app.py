import streamlit as st
import joblib
import pandas as pd
import zipfile
import os

# Combine split zip files into a single zip file
def combine_split_zips(output_zipfile, *input_parts):
    with open(output_zipfile, 'wb') as output_file:
        for part in input_parts:
            if not os.path.exists(part):
                st.error(f"Error: The file {part} does not exist.")
                return None
            with open(part, 'rb') as part_file:
                output_file.write(part_file.read())
    return output_zipfile

# Unarchive the models
def extract_model_from_zip(zipfile_name, model_filename):
    try:
        with zipfile.ZipFile(zipfile_name, 'r') as zip_ref:
            zip_ref.extractall()
        if not os.path.exists(model_filename):
            st.error(f"Error: {model_filename} was not found in the extracted zip.")
            return None
        return model_filename
    except Exception as e:
        st.error(f"Error extracting zip file: {str(e)}")
        return None

# Load the model
@st.cache_resource
def load_model():
    # Combine the split zip files into one zip
    zipfile_name = combine_split_zips('heartbeat_model.zip', 'heartbeat_model.z01', 'heartbeat_model.z02')

    if zipfile_name is None:
        st.error("Failed to combine the zip files.")
        return None

    # Extract the model file from the combined zip
    model_filename = extract_model_from_zip(zipfile_name, 'heartbeat_model.pkl')

    if model_filename is None:
        st.error("Failed to extract the model.")
        return None

    # Load the model using joblib
    try:
        model = joblib.load(model_filename)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the model
model = load_model()

# If the model failed to load, stop the app
if model is None:
    st.stop()

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
    if input_data.shape[1] == 100:  # Expecting 100 values
        # Ensure the column names match expected feature names
        expected_feature_names = [str(i) for i in range(100)]  # Adjust according to your model's training
        input_data.columns = expected_feature_names  # Rename if necessary

        # Make predictions
        predictions = model.predict(input_data)

        # Convert predictions to DataFrame for better display
        predictions_df = pd.DataFrame(predictions, columns=["Prediction"])

        # Map predictions to labels
        predictions_df["Label"] = predictions_df["Prediction"].map({0: "Normal", 1: "Abnormal"})

        # Display predictions
        st.write("Predictions (0: Normal, 1: Abnormal):")
        st.write(predictions_df)
    else:
        st.error("The uploaded CSV must contain exactly 100 values (columns). Please check your file.")
