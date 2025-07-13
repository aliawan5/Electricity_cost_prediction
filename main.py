from src.ingest_data import IngestData
from src.preprocess_data import Preprocessing
from src.train_model import ModelTraining

def func():
    raw_data_path = r'C:\Users\FINE\Electricity_cost_prediction\data\raw\electricity_cost_dataset.csv'
    model_path = r'C:\Users\FINE\Electricity_cost_prediction\model'


    obj = IngestData(raw_data_path)
    data = obj.load_data()

    print(data.head())
    print(data.info())

    obj1 = Preprocessing(data)
    process_data = obj1.preprocess_data()

    print(process_data.head())

    obj2 = ModelTraining(process_data, model_path)
    obj2.train_model()

if __name__ == "__main__":
    func()