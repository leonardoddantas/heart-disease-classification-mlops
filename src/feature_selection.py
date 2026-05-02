from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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

def select_features_f_classif(X, y, k=10):
    
    selector = SelectKBest(score_func=f_classif, k=k)
    
    X_new = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    
    scores = selector.scores_
    
    feature_scores = sorted(
        zip(X.columns, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nFeature scores (f_classif):")
    for name, score in feature_scores:
        print(f"{name}: {score:.4f}")
    
    return selected_features

def select_features_mutual_info(X: pd.DataFrame, y: pd.Series, k=10):
    
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_series = pd.Series(mi_scores, index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)
    
    print("\nMutual Information scores:")
    for feature, score in mi_series.items():
        print(f"{feature}: {score:.4f}")
    
    selected_features = mi_series.head(k).index.tolist()
    
    print("\nSelected features:", selected_features)
    
    return selected_features