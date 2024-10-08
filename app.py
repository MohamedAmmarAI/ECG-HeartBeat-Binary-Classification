import streamlit as st
import joblib
import pandas as pd

# Load your model (replace with the actual path to your model)
model = joblib.load('heartbeat_model.pkl')

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

    # Optionally print the cleaned column names for debugging
    # st.write("Cleaned Column Names:")
    # st.write(input_data.columns)

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
