import pandas as pd
import logging
import os


class IngestData:
    def __init__(self, data_path : str):
        self.data_path = data_path
        self.data = None


    def load_data(self) -> pd.DataFrame:
        try:
            logging.info('Loading data from csv file')

            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"File Not found : {self.data_path}")
            
            self.data = pd.read_csv(self.data_path)
            logging.info('Data loaded successfully')
            return self.data
        
        except Exception as e:
            logging.error(f'An error occurred : {str(e)}')
            raise e