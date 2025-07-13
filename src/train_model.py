import os
import dagshub
import logging
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class ModelTraining:
    def __init__(self, process_data: pd.DataFrame, model_dir: str = "models"):
        self.process_data = process_data
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, "model.pkl")

    def train_model(self):
        try:
            logging.info("Starting model training...")


            X = self.process_data.drop(columns=["electricity cost"])
            y = self.process_data["electricity cost"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            dagshub.init(repo_owner='aliawan05500', repo_name='Electricity_cost_prediction', mlflow=True)

            with mlflow.start_run(run_name="RandomForest_Electricity"):
                mlflow.set_tag("Test Run", "CI Pipeline test 1")
                mlflow.log_param("model_type", "RandomForest")
                mlflow.log_param("n_estimators", 100)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                mlflow.log_metric("mse", mse)

                logging.info(f"Model trained successfully | MSE: {mse:.4f}")


                os.makedirs(self.model_dir, exist_ok=True)
                joblib.dump(model, self.model_path)
                mlflow.log_artifact(self.model_path, artifact_path="outputs")

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise e
