import flwr as fl
import tensorflow as tf
import sys

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

# Get client id from command line arguments
client_id = int(sys.argv[1])

# Split the dataset for each client
if client_id == 1:
    client_data = (x_train[1:15], y_train[1:15])
elif client_id == 2:
    client_data = (x_train[15:30], y_train[15:30])
elif client_id == 3:
    client_data = (x_train[30:45], y_train[30:45])

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

# Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, data):
        self.model = create_model()
        self.x_train, self.y_train = data

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": accuracy}

# Start the client
def start_client():
    fl.client.start_client(server_address="localhost:8082", client=CifarClient(client_data).to_client())

if __name__ == "__main__":
    start_client()
