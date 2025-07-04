# -*- coding: utf-8 -*-
"""
A function to delete multicolinearity and select variables

Author: Faouzi Zakaria
Date: 04/07/2025
"""

import pandas as pd
import scipy as sp
from scipy.stats import pearsonr

class FeaturesSeelectionPipeline:

    def __init__(self, df, threshold = 0.7):
        self.df = df
        self.threshold = threshold
        self.df_columns = self.df.columns.to_list()

    def calculate_pearson_correlation(self):
        """A functio to calculate pearson correaltion between each feature 
            and other features with returning a dictionnary {feature_i: [feature_j, corr, p_value]}
        """
        # columns of our data
        columns = self.df_columns
        # final results
        results = {}
        for i in range(len(columns)):
            col_i = columns[i]
            df_col_i = self.df[col_i]
            results[col_i] = []
            for j in range(i+1, len(columns)):
                col_j = columns[j]
                df_col_j = self.df[col_j]
                corr,  p_value = pearsonr(df_col_i, df_col_j)
                if abs(corr) >self.threshold and p_value<0.05:
                    results[col_i].append((col_j, corr, p_value))
                else:
                    pass
        return results
    
    def list_uncorrelated_features(self):
        """a function to return finale list of non correlated features

        Returns:
            list finale of non correlated features
        """        
        to_remove = set()
        results_corr = self.calculate_pearson_correlation()
        for _, corr_list in results_corr.items():
            for col_corr, _, _ in corr_list:
                to_remove.add(col_corr)
        
        selected_features = [col for col in self.df_columns if col not in to_remove]
        return selected_features