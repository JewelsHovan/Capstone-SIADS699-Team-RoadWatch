import mlflow
import mlflow.sklearn
from datetime import datetime


def setup_mlflow(experiment_name="WorkZonePrediction"):
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set: {experiment_name}")



def log_model_params(model, params: dict, metrics: dict, artifact_path: str = "models") -> None:
    """
    Logs model parameters, metrics, and the serialized model to MLflow.
    """
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        print(f"Model logged successfully under run: {run_name}")
