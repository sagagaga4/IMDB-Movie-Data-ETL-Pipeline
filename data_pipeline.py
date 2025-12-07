import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

def count_duplicates(df: pd.DataFrame, subset_col: Optional[str] = None) -> int:
    """
    Counts duplicate rows. If subset_col is provided, checks specific column.
    """
    if subset_col:
        return df.duplicated(subset=subset_col).sum()
    return df.duplicated().sum()

def remove_duplicates(df: pd.DataFrame, subset_col: Optional[str] = None) -> pd.DataFrame:
    """
    Removes duplicate rows, keeping the first occurrence.
    """
    if subset_col:
        df_clean = df.drop_duplicates(subset=subset_col)
    else:
        df_clean = df.drop_duplicates()
    return df_clean

def filter_corrupt_rows(df: pd.DataFrame, max_missing_cols: int) -> pd.DataFrame:
    """
    Removes rows that have more than 'max_missing_cols' missing values.
    """
    missing_counts = df.isnull().sum(axis=1)
    # Keep rows where missing count is <= limit
    return df[missing_counts <= max_missing_cols].copy()

def impute_missing_values(df: pd.DataFrame, defaults_dict: Dict[str, any]) -> pd.DataFrame:
    """
    Imputes missing values using a three-step strategy:
    1. Predefined defaults from defaults_dict.
    2. Median for remaining numeric columns.
    3. Mode (most frequent) for remaining categorical columns.
    """
    df_clean = df.copy()

    # 1. Predefined defaults
    for col, val in defaults_dict.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(val)

    # Identify remaining columns
    remaining_cols = [c for c in df_clean.columns if c not in defaults_dict]
    
    # 2. Numeric: Fill with Median
    num_cols = df_clean[remaining_cols].select_dtypes(include=['number']).columns
    for col in num_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

    # 3. Categorical: Fill with Mode
    cat_cols = df_clean[remaining_cols].select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        if df_clean[col].notna().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            
    return df_clean

def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces outliers in numerical columns with NaN using the Interquartile Range (IQR) method.
    Formula: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        df_out.loc[outlier_mask, col] = np.nan
        
    return df_out

def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Replaces outliers in numerical columns with NaN using Z-Score.
    Cutoff is standard deviation > threshold (default 3).
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        mean_val = df_out[col].mean()
        std_val = df_out[col].std()
        
        if std_val > 0:
            z_scores = (df_out[col] - mean_val) / std_val
            outlier_mask = z_scores.abs() > threshold
            df_out.loc[outlier_mask, col] = np.nan
            
    return df_out

def feature_engineering(df: pd.DataFrame, 
                        numeric_bins: Dict[str, List[float]], 
                        categorical_cols: List[str]) -> pd.DataFrame:
    """
    Performs feature engineering:
    1. Bins numeric columns into categories (1-5) based on provided ranges.
    2. Performs One-Hot Encoding on specified categorical columns.
    """
    df_out = df.copy()
    
    # Binning
    for col, bins in numeric_bins.items():
        if col in df_out.columns:
            new_col = f"{col}_categorical"
            df_out[new_col] = pd.cut(df_out[col], bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
            
    # One-Hot Encoding
    for col in categorical_cols:
        if col in df_out.columns:
            dummies = pd.get_dummies(df_out[col], prefix=col)
            df_out = pd.concat([df_out, dummies], axis=1)
            
    return df_out

# --- Main Execution Flow ---
if __name__ == "__main__":
    # Configuration
    DATA_PATH = os.path.join('.', 'data', 'imdb_movies.csv')
    
    DEFAULTS = {
        'director_name': 'unknown', 
        'actor_1_name': 'unknown', 
        'actor_2_name': 'unknown', 
        'actor_3_name': 'unknown', 
        'genres': 'unknown', 
        'plot_keywords': 'unknown', 
        'movie_title': 'unknown', 
        'movie_imdb_link': 'unknown', 
        'country': 'unknown'
    }
    
    # Feature Engineering Config
    BIN_CONFIG = {
        'duration': [41.0, 92.0, 100.0, 108.0, 121.0, 175.0], 
        'imdb_score': [3, 5.6, 6.3, 6.799999999999999, 7.299999999999999, 9.3]
    }
    ONE_HOT_COLS = ['content_rating']

    # 1. Load
    if os.path.exists(DATA_PATH):
        df = load_dataset(DATA_PATH)
        
        # 2. Clean Duplicates
        print(f"Duplicates detected: {count_duplicates(df)}")
        df = remove_duplicates(df, subset_col='movie_imdb_link')
        
        # 3. Filter Corrupt Rows
        df = filter_corrupt_rows(df, max_missing_cols=3)
        
        # 4. Impute Missing Values
        df = impute_missing_values(df, DEFAULTS)
        
        # 5. Outlier Detection (Z-Score strategy chosen)
        df = remove_outliers_zscore(df)
        # Re-impute if outliers introduced new NaNs
        df = impute_missing_values(df, DEFAULTS) 

        # 6. Feature Engineering
        df_final = feature_engineering(df, BIN_CONFIG, ONE_HOT_COLS)
        
        print("Data Pipeline Completed.")
        print(f"Final Data Shape: {df_final.shape}")
        print(df_final.head())
    else:
        print("Data file not found. Please check the path.")
