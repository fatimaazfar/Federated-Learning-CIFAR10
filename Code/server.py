import flwr as fl
import torch
import numpy as np
import pickle
from typing import Dict, List, Tuple
from torchvision import models

# Define a simple model
def create_model(num_classes):
    model = models.resnet101(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# Define custom aggregation functions
def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    accuracy_list = [m[1].get("accuracy", 0) for m in metrics]
    loss_list = [m[1].get("loss", 0) for m in metrics]
    return {
        "accuracy": np.mean(accuracy_list),
        "loss": np.mean(loss_list)
    }

def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    accuracy_list = [m[1].get("accuracy", 0) for m in metrics]
    loss_list = [m[1].get("loss", 0) for m in metrics]
    return {
        "accuracy": np.mean(accuracy_list),
        "loss": np.mean(loss_list)
    }

# Define strategy
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, fit_metrics_aggregation_fn=None, evaluate_metrics_aggregation_fn=None):
        super().__init__(
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )
        self.parameters = None

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[List, Dict[str, fl.common.Scalar]]],
        failures: List
    ) -> Tuple[List, Dict[str, fl.common.Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

def fit_config(server_round: int):
    return {"epochs": 1}

strategy = SaveModelStrategy(
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics
)

# Start the server
def start_server():
    fl.server.start_server(server_address="localhost:8082", config=fl.server.ServerConfig(num_rounds=10), strategy=strategy)
    # Save the parameters to a file after training
    if strategy.parameters:
        with open("global_parameters.pkl", "wb") as f:
            pickle.dump(fl.common.parameters_to_ndarrays(strategy.parameters), f)

if __name__ == "__main__":
    start_server()
