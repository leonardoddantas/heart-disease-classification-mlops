import wandb
import pandas as pd
import numpy as np
import os


def log_raw_data(df: pd.DataFrame):
    
    wandb.init(
        project="heart-disease-classification-mlops",
        job_type="load_raw",
        name="load_raw"
    )
    
    artifact = wandb.Artifact(
        name="raw_data",
        type="dataset",
        description="Heart disease dataset from Kaggle"
    )
    
    temp_path = "temp_raw.csv"
    df.to_csv(temp_path, index=False)
    
    artifact.add_file(temp_path)
    
    wandb.log_artifact(artifact)
    
    wandb.summary["rows"] = len(df)
    wandb.summary["columns"] = list(df.columns)
    
    wandb.finish()

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    
    before = len(df)
    
    df = df.drop_duplicates().reset_index(drop=True)
    
    removed = before - len(df)
    
    if removed > 0:
        print(f"Removed {removed} duplicate rows ")
    else:
        print("No duplicate rows found.")
    
    return df

def handle_outliers(df, columns):
    
    for col in columns:
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower, upper)
        
        print(f"{col}: treated outliers")
    
    return df

def handle_skew(df: pd.DataFrame) -> pd.DataFrame:
    
    if "oldpeak" in df.columns:
        df["oldpeak"] = np.log1p(df["oldpeak"])
        print("Applied log1p transformation on 'oldpeak'")
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = remove_duplicates(df)
    
    numeric_cols = [
        "resting_bp",
        "cholestoral",
        "max_hr",
        "oldpeak"
    ]
    
    df = handle_outliers(df, numeric_cols)
    
    df = handle_skew(df)
    
    print("Data cleaning finished.\n")
    
    return df

def log_processed_data(df: pd.DataFrame):
    
    wandb.init(
        project="heart-disease-classification-mlops",
        job_type="process_data",
        name="processed_data"
    )
    
    artifact = wandb.Artifact(
        name="processed_data",
        type="dataset",
        description="Cleaned and preprocessed heart disease dataset"
    )
    
    temp_path = "temp_processed.csv"
    df.to_csv(temp_path, index=False)
    
    artifact.add_file(temp_path)
    
    wandb.log_artifact(artifact)
    
    wandb.summary["rows"] = len(df)
    wandb.summary["columns"] = list(df.columns)
    
    wandb.finish()