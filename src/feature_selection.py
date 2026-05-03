from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

def correlation_analysis(df: pd.DataFrame, threshold: float = 0.9):
    
    df_features = df.drop("target", axis=1)
    
    corr_matrix = df_features.corr().abs()
    
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]
    
    print("Highly correlated features:")
    print(to_drop)
    
    return to_drop

def get_correlation_scores(X, y):
    
    df = X.copy()
    df["target"] = y
    
    return df.corr()["target"].drop("target").abs()

def get_f_classif_scores(X, y):
    
    scores, _ = f_classif(X, y)
    
    return pd.Series(scores, index=X.columns)

def get_mutual_info_scores(X, y):
    
    scores = mutual_info_classif(X, y, random_state=42)
    
    return pd.Series(scores, index=X.columns)

def get_random_forest_scores(X, y):
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    return pd.Series(model.feature_importances_, index=X.columns)

def normalize(s):
    if s.max() == s.min():
        return pd.Series(0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def aggregate_feature_selection(X, y):
    
    corr = get_correlation_scores(X, y)
    f = get_f_classif_scores(X, y)
    mi = get_mutual_info_scores(X, y)
    rf = get_random_forest_scores(X, y)
    
    df_scores = pd.DataFrame({
        "Correlation": normalize(corr),
        "F_classif": normalize(f),
        "Mutual_Info": normalize(mi),
        "Random_Forest": normalize(rf)
    })
    
    df_scores["Final_Score"] = df_scores.mean(axis=1)
    
    df_scores = df_scores.sort_values(by="Final_Score", ascending=False)
    
    print("\n=== Feature Selection Comparison ===\n")
    print(df_scores.round(4))

    selected_features = df_scores.index[:10].tolist()

    print("\nSelected features:", selected_features)

    return df_scores, selected_features

def calculate_vif(X: pd.DataFrame):
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    
    print("\n=== VIF (Multicollinearity Check) ===\n")
    print(vif_data)
    
    return vif_data