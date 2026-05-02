import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    
    categorical_cols = [
        "chest_pain_type",
        "restecg",
        "slope",
        "thal"
    ]
    
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )
    
    print("One-hot encoding aplicado.")
    
    return df

def scale_features(df):
    
    numeric_cols = [
        "age",
        "resting_bp",
        "cholestoral",
        "max_hr",
        "oldpeak",
        "num_major_vessels"
    ]
    
    scaler = StandardScaler()
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("Applied standardization.")
    
    return df