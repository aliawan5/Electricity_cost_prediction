import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder


class Preprocessing:
    def __init__(self, raw_data : pd.DataFrame):
        self.raw_data = raw_data

    

    def preprocess_data(self):
        try:
            logging.info('Preprocessing raw data')

            if self.raw_data.isna().any().any():
                self.raw_data.dropna(inplace=True)

            le = LabelEncoder()
            self.raw_data['structure type'] = le.fit_transform(self.raw_data['structure type'])

            return self.raw_data

        except Exception as e:
            logging.error(f'An error occurred : {str(e)}')

