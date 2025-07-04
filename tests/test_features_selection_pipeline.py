# -*- coding : UTF-8 -*-

"""A unittest for our feature_selection_pipeline
    """
import sys
import os

import unittest
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features_selection_pipeline import FeaturesSeelectionPipeline

class TestFeaturesSelectionPipeline(unittest.TestCase):

    def setUp(self):
        # Créer un DataFrame avec des corrélations connues
        data = {
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],   # corrélé à A
            'C': [5, 3, 6, 2, 1],    # non corrélé à A
            'D': [1, 1, 1, 1, 1],    # constante, p-value sera nan
        }
        self.df = pd.DataFrame(data)
        self.pipeline = FeaturesSeelectionPipeline(self.df, threshold=0.7)

    def test_calculate_pearson_correlation(self):
        results = self.pipeline.calculate_pearson_correlation()
        # Vérifie que A est corrélé à B
        self.assertIn('A', results)
        self.assertTrue(any(item[0] == 'B' for item in results['A']))
        # Vérifie que C n'apparaît pas comme corrélé
        self.assertFalse(any(item[0] == 'C' for item in results['A']))

    def test_list_uncorrelated_features(self):
        selected = self.pipeline.list_uncorrelated_features()
        # Puisque A est corrélé à B, B devrait être retiré
        self.assertIn('A', selected)
        self.assertNotIn('B', selected)
        # C devrait rester
        self.assertIn('C', selected)
        # D devrait rester même si sa variance est nulle (non corrélée techniquement)
        self.assertIn('D', selected)


if __name__ == '__main__':
    unittest.main()

    
