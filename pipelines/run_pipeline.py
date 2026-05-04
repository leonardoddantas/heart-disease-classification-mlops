from src.data_loading import load_data
from src.data_cleaning import clean_data, log_processed_data, log_raw_data
from src.data_preprocessing import encode_categorical, scale_features
from src.feature_selection import correlation_analysis, aggregate_feature_selection, calculate_vif
from src.split_data import split_train_test, compare_distributions, log_split_data
from src.model_preparation import prepare_dataloaders
from src.model import build_model
from src.train import train_model
from src.utils import set_seed, load_config
from src.evaluation import evaluate_model
import torch

def run():

    config = load_config()
    set_seed(config["data"]["random_state"]) 

    df = load_data()
    
    log_raw_data(df)
    df = clean_data(df)
    df = encode_categorical(df)
    df = scale_features(df)
    log_processed_data(df)

    correlation_analysis(df)
    X = df.drop("target", axis=1)
    y = df["target"]
    scores, selected_features = aggregate_feature_selection(X, y)

    X_selected = X[selected_features]
    vif = calculate_vif(X_selected)

    df_model = df[selected_features + ["target"]]
    train_df, test_df = split_train_test(df_model, "target")
    results = compare_distributions(train_df,
                                    test_df,
                                    [c for c in train_df.columns if c != "target"])

    for col, res in results.items():
        print(f"{col}: {res}")

    log_split_data(train_df, test_df, results)

    train_loader, test_loader, scaler = prepare_dataloaders(
        train_df,
        test_df,
        target_col="target",
        batch_size=16)

    model = train_model(
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=len(selected_features))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc, prec, rec, f1, cm = evaluate_model(
        model,
        test_loader,
        device
    )

if __name__ == "__main__":
    run()