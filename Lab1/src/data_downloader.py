import os
import requests
from dotenv import load_dotenv

load_dotenv()

def download_data():
    """
    Downloads the dataset from the URL specified in the .env file.
    """
    dataset_url = os.getenv("DATASET_URL")
    save_path = os.getenv("DATA_FILE_PATH")

    if not os.path.exists(save_path):
        response = requests.get(dataset_url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Dataset downloaded and saved to {save_path}")
    else:
        print(f"Dataset already exists at {save_path}")
