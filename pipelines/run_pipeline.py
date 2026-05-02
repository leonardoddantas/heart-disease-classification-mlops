from src.data_loading import load_data
from src.data_cleaning import clean_data, log_processed_data, log_raw_data
from src.data_preprocessing import encode_categorical, scale_features

def run():
    df = load_data()
    
    log_raw_data(df)
    df = clean_data(df)
    df = encode_categorical(df)
    df = scale_features(df)
    
    log_processed_data(df)
    
    print(df.head())

if __name__ == "__main__":
    run()