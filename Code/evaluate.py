import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from itertools import cycle
import pickle  # Import pickle for loading the saved parameters

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255.0
y_test = tf.keras.utils.to_categorical(y_test, 10)

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

# Load global model parameters
global_model = create_model()
with open("global_parameters.pkl", "rb") as f:
    global_parameters = pickle.load(f)
global_model.set_weights(global_parameters)

# Evaluate global model
global_loss, global_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
y_pred_global = global_model.predict(x_test)
y_pred_global_classes = np.argmax(y_pred_global, axis=1)

# Classification report
print("Global Classification Report")
print(classification_report(np.argmax(y_test, axis=1), y_pred_global_classes, zero_division=1))

# Multiclass ROC Curve using One-vs-Rest approach
y_test_binarized = label_binarize(np.argmax(y_test, axis=1), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_global[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting the ROC curves
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'black'])
for i, color in zip(range(10), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for CIFAR-10')
plt.legend(loc="lower right")
plt.show()

# Overall Precision, Recall, F1-Score
precision, recall, f1, _ = precision_recall_fscore_support(np.argmax(y_test, axis=1), y_pred_global_classes, average='macro')
print(f"Overall Precision: {precision}")
print(f"Overall Recall: {recall}")
print(f"Overall F1 Score: {f1}")
