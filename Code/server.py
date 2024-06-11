import flwr as fl
import tensorflow as tf
from typing import Dict, List, Tuple

# Define a simple model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define strategy
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
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

strategy = SaveModelStrategy()

# Start the server
def start_server():
    fl.server.start_server(server_address="localhost:8082", config=fl.server.ServerConfig(num_rounds=10), strategy=strategy)

if __name__ == "__main__":
    start_server()
