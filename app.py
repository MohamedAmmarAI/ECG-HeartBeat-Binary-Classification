import streamlit as st
import joblib
import gdown
import os

# URL of the model file on Google Drive
MODEL_URL = 'https://drive.google.com/uc?id=1G9K3-xyz'  # Replace with your actual file ID

# Download the model if not already downloaded
def download_model():
    model_filename = 'heartbeat_model.pkl'
    
    if not os.path.exists(model_filename):
        try:
            st.write("Downloading model file from Google Drive...")
            gdown.download(MODEL_URL, model_filename, quiet=False)
            st.write("Model downloaded successfully.")
            
            # Check if the file is fully downloaded (e.g., 123 MB)
            file_size = os.path.getsize(model_filename) / (1024 * 1024)  # Size in MB
            st.write(f"Downloaded file size: {file_size:.2f} MB")
            if file_size < 120:  # Expecting around 123 MB
                st.error("Error: The model file seems to be incomplete.")
                return None
            
            # Verify if the file starts with a binary header
            with open(model_filename, 'rb') as f:
                first_bytes = f.read(4)
                if first_bytes.startswith(b'<!DO') or first_bytes.startswith(b'<htm'):
                    st.error("Error: The downloaded file is not a valid model file (looks like HTML).")
                    return None
            
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None
    return model_filename

# Load the model
@st.cache_resource
def load_model():
    model_filename = download_model()
    
    if model_filename is None or not os.path.exists(model_filename):
        st.error("Model file not found. Could not load the model.")
        return None

    try:
        # Try loading the model with joblib
        model = joblib.load(model_filename)
        return model
    except Exception as e:
        st.error(f"Error loading model with joblib: {str(e)}")
        
        # Fallback to using pickle if joblib fails
        try:
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
                return model
        except Exception as e2:
            st.error(f"Error loading model with pickle: {str(e2)}")
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
