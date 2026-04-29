import kagglehub
import shutil
import os
import pandas as pd


def download_dataset():
    """Download dataset from Kaggle using kagglehub"""
    
    path = kagglehub.dataset_download(
        "abhishek14398/heart-disease-classification"
    )
    
    print(f"Downloaded to: {path}")
    return path


def copy_to_project(path):
    """Copy dataset to project data/raw folder"""
    
    destination = os.path.join(os.getcwd(), "data", "raw")
    
    os.makedirs(destination, exist_ok=True)
    
    shutil.copytree(path, destination, dirs_exist_ok=True)
    
    print(f"Dataset copied to: {destination}")

def load_data():
    path = download_dataset()
    copy_to_project(path)
    
    df = pd.read_csv("data/raw/heart.csv")
    
    return df