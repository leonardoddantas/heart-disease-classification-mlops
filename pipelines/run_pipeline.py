from src.data_loading import load_data
from src.data_cleaning import clean_data, log_processed_data, log_raw_data
from src.data_preprocessing import encode_categorical, scale_features
from src.feature_selection import correlation_analysis, aggregate_feature_selection, calculate_vif

def run():
    df = load_data()
    
    #log_raw_data(df)
    df = clean_data(df)
    df = encode_categorical(df)
    df = scale_features(df)

    correlation_analysis(df)
    X = df.drop("target", axis=1)
    y = df["target"]
    scores, selected_features = aggregate_feature_selection(X, y)

    X_selected = X[selected_features]
    vif = calculate_vif(X_selected)
    
    #log_processed_data(df)

if __name__ == "__main__":
    run()