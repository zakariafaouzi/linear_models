#\* UTF-8
"""A function to preprocess our dataframe with dropna and replace outliers with max

Author: Faouzi Zakaria
Date: 01/07/2025

"""

from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


class PreProcessing:

    def __init__(self, df, contamination= 0.02):
        self.df = df
        self.contamination = contamination

    def drop_nans(self):
        self.df = self.df.dropna()
        return self
    
    def detect_outliers_with_isolation_forest(self):
        model = IsolationForest(contamination=self.contamination, random_state=42)
        preds = model.fit_predict(self.df)
        outliers_indices = np.where(preds == -1)[0]
        print(f"[INFO] Outliers détectés : {len(outliers_indices)}")
        for col in self.df.columns:
            col_values = self.df.loc[preds == 1, col]
            replacement = col_values.max()
            self.df.loc[outliers_indices, col] = replacement
        return self
    
    def run_pipeline_preprocessing(self):
        return (self.drop_nans()
                .detect_outliers_with_isolation_forest().df)