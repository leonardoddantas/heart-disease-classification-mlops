from src.data_loading import load_data
from src.data_cleaning import clean_data, log_processed_data, log_raw_data
from src.data_preprocessing import encode_categorical, scale_features
from src.feature_selection import correlation_analysis, select_features_f_classif, select_features_mutual_info

def run():
    df = load_data()
    
    #log_raw_data(df)
    df = clean_data(df)
    df = encode_categorical(df)
    df = scale_features(df)

    correlation_analysis(df)
    X = df.drop("target", axis=1)
    y = df["target"]

    selected_mi = select_features_mutual_info(X, y, k=8)

    print("Selected features:", selected_mi)
    
    #log_processed_data(df)

if __name__ == "__main__":
    run()