from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, chi2_contingency
import pandas as pd
import wandb

def split_train_test(df: pd.DataFrame,
                    target_col: str,
                    test_size: float = 0.2,
                    random_state: int = 42):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, test_df

def compare_distributions(train_df, test_df, columns):
    
    results = {}

    for col in columns:
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()

        if train_df[col].dtype in ["int64", "float64"]:
            stat, p = ks_2samp(train_vals, test_vals)
            results[col] = {"test": "KS", "statistic": stat, "p_value": p}

        else:
            train_counts = train_vals.value_counts(normalize=True)
            test_counts = test_vals.value_counts(normalize=True)

            all_cats = sorted(set(train_counts.index).union(test_counts.index))

            train_probs = [train_counts.get(c, 0) for c in all_cats]
            test_probs = [test_counts.get(c, 0) for c in all_cats]

            chi2, p, _, _ = chi2_contingency([train_probs, test_probs])

            results[col] = {"test": "Chi2", "statistic": chi2, "p_value": p}

    return results

def log_split_data(train_df, test_df, comp_results):

    wandb.init(
        project="heart-disease-classification-mlops",
        job_type="split_data",
        name="split_data"
    )

    for split_name, split_df in [("train_data", train_df), ("test_data", test_df)]:
        
        path = f"temp_{split_name}.csv"
        split_df.to_csv(path, index=False)

        artifact = wandb.Artifact(
            name=split_name,
            type="dataset",
            description=f"{split_name} after split"
        )

        artifact.add_file(path)
        wandb.log_artifact(artifact)

    comp_df = pd.DataFrame(comp_results).T.reset_index()
    comp_df.columns = ["feature", "test", "statistic", "p_value"]

    comp_table = wandb.Table(dataframe=comp_df)
    wandb.log({"distribution_comparison": comp_table})

    wandb.summary["train_size"] = len(train_df)
    wandb.summary["test_size"] = len(test_df)

    wandb.finish()

    print("Train and test artifacts saved.")