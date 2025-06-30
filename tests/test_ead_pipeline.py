"""_summary_

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from src.ead_pipeline import EAD_Dataframe


class TestEADDataframe:

    def setup_method(self):
        data = {'A': [2, 2, 2, None], 'B': [3, 4, 4, 5]}
        self.df = pd.DataFrame(data)
        self.ead = EAD_Dataframe(self.df)
    
    def test_get_columns_and_size(self):
        result = self.ead.get_columns_and_size()
        assert isinstance(result, EAD_Dataframe)
        assert result.df.shape == (4, 2)
        assert list(result.df.columns) == ['A', 'B']

    def test_detect_nans_and_duplicates(self):
        result = self.ead.detect_nans_and_duplicates()
        assert isinstance(result, EAD_Dataframe)
        assert self.df.isnull().values.any() == True
        duplicated_rows = self.df.duplicated(keep=False)
        assert duplicated_rows.sum() > 0



if __name__ == '__main__':
    unittest.main()