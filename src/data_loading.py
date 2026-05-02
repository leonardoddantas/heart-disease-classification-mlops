import kagglehub
import shutil
import os
import pandas as pd

def download_dataset():
    
    path = kagglehub.dataset_download(
        "abhishek14398/heart-disease-classification"
    )
    
    print(f"Downloaded to: {path}")
    return path

def copy_to_project(path):
    
    destination = os.path.join(os.getcwd(), "data", "raw")
    
    os.makedirs(destination, exist_ok=True)
    
    shutil.copytree(path, destination, dirs_exist_ok=True)
    
    print(f"Dataset copied to: {destination}")

def load_data():

    path = download_dataset()
    copy_to_project(path)
    
    df_heart_dissease = pd.read_csv("data/raw/heart.csv")
    
    return df_heart_dissease