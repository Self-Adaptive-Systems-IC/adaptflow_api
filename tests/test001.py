import requests
import pandas as pd
from base64 import b64decode
import os
import random
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")


# Function to load base64-encoded pickle file
def load_base64_pickle(base64_string, filename):
    """Load a base64-encoded pickle file.

    Args:
        base64_string (str): Base64-encoded string.
        filename (str): Location to save the file.
    """
    with open(filename, 'wb') as f:
        f.write(b64decode(base64_string))
        print(f"file saved on {filename}")
    
# Generate random numbers based on dataset
def generate_new_param(file_path,ignore_column):
    """Generate random parameter values for testing the model.

    Args:
        file_path (str): Path to the dataset.
        ignore_column (str): Target column.

    Returns:
        dict: Random parameter values to test the model.
    """
    df = pd.read_csv(file_path)
    params = {}
    for column in df.columns:
        if column != ignore_column:
            params[column] = random.randint(int(df[column].min()), int(df[column].max()))
    return params

# Function to send a dataset and load the returned model
def send_dataset_and_load_model(api_endpoint,models_dir,file_path, n, metric):
    """Send a dataset, receive the trained model, and load it.

    Args:
        api_endpoint (str): API endpoint to send the dataset.
        models_dir (str): Directory to save the loaded models.
        file_path (str): Path to the dataset.
        target_column (str): Target column in the dataset.
        n (int): Number of models to request.
        metric (str): Metric for selecting the model.

    Returns:
        object: Loaded machine learning model.
    """
    # Open the file and prepare it for upload
    with open(file_path, 'rb') as file:
        files = {"file": (os.path.basename(file_path), file, "text/csv")}
        # Make a POST request to the select_model endpoint with the file
        response = requests.post(api_endpoint, files=files, data={'n': n, 'metric': metric})

        # Check the response status code
        if response.status_code == 200:
            # The request was successful, and you can process the response JSON
            response_json = response.json()
        else:
            # The request failed, and you can print an error message
            raise RuntimeError(f"Error: {response.status_code} - {response.text}")

    try:
        model_data = response_json['data'][0]['pickle']
        save_location = f"{models_dir}/{model_data['name']}.pkl"
        load_base64_pickle(model_data['data'], save_location)
        loaded_model = joblib.load(save_location)

        return loaded_model
        
    except requests.exceptions.HTTPError as errh:
        raise RuntimeError(f"HTTP Error: {errh}")
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

if __name__ == "__main__":
    api_endpoint = "http://192.168.2.131:8000/select_model"
    file_path = "/home/romulolass/Codes/ml_datasets/dataset_edit.csv"
    target = "label"
    models_dir = "./tmp/loaded_models"
    n = 1
    metric = "Accuracy"
    # Example usage
    model = send_dataset_and_load_model(api_endpoint, models_dir,file_path,n,metric)
    
    print("Testing")
        
    for i in range(0,3):
        new_data = generate_new_param(file_path,target)
        predictions = model.predict( pd.DataFrame(new_data, index=[0]).values)
        print(new_data,predictions,end="\n\n")
            

