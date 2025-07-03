# -*- coding: utf-8 -*-
"""
A function to test preprocessing pipeline with dropna, detect outliers with 
isolation forest and replace outliers with max.

Author: Faouzi Zakaria
Date: 03/07/2025
"""

import unittest
import pandas as pd
from src.preprocessing_pipeline import PreProcessing

class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        data = {
            "A": [2, 5, 6, None, 5, None, 10, 3, 7.5, 9],
            "B": [4, 7, 9, 6, 10000, 4, 8, 7.8, 9, 10]
        }
        self.df = pd.DataFrame(data)
        self.preprocess = PreProcessing(self.df, contamination=0.1)

    def test_drop_nans(self):
        result = self.preprocess.drop_nans()
        self.assertIsInstance(result, PreProcessing)
        self.assertEqual(len(result.df), 8)
        self.assertListEqual(list(result.df.columns), ['A', 'B'])

    def test_detect_outliers(self):
        # Clean data first
        cleaned = self.preprocess.drop_nans()
        result = cleaned.detect_outliers_with_isolation_forest()
        
        self.assertIsInstance(result, PreProcessing)
        
        # Vérifie qu'il n'y a plus de valeur aberrante extrême
        self.assertTrue(result.df['B'].max() < 10000)

if __name__ == "__main__":
    unittest.main()
