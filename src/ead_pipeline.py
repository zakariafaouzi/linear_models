#\* UTF-8
"""A function to proceed with EAD of our dataframe

Author: Faouzi Zakaria
Date: 30/06/2025

"""


class EAD_Dataframe:

    def __init__(self, df):
        self.df = df
    
    def get_columns_and_size(self):
        print(f"[INFO] - List of columns in our dataframe is: {list(self.df.columns)}")
        print(f"[INFO] - The size of our dataframe is: {self.df.shape}")
        return self
    
    def detect_nans_and_duplicates(self):
        # filter df with nans
        rows_with_nans = self.df[self.df.isnull().any(axis = 1)]
        # count nans
        count_nans = rows_with_nans.shape
        print(f"[INFO] - Sum of null rows is: {count_nans}")

        # count duplicates
        counts = self.df.groupby(self.df.columns.tolist()).size().reset_index(name = 'count')
        #filter rows with countb >1
        duplicates = counts[counts['count']>1]
        print(f"[INFO] - Sum of duplicates rows: {duplicates.shape}")
        
        return self
    
    def run_pipeline(self):
        return (self.get_columns_and_size()
                .detect_nans_and_duplicates().df
                )