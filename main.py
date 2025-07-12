from src.ingest_data import IngestData

def func():
    raw_data_path = r'C:\Users\FINE\Electricity_cost_prediction\data\raw\electricity_cost_dataset.csv'


    obj = IngestData(raw_data_path)
    data = obj.load_data()

    print(data.head())
    print(data.info())

if __name__ == "__main__":
    func()