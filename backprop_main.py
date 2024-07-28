import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *


sections = ['e']

# Loading Data
np.random.seed(0)  # For reproducibility
# n_train = 50000
# n_test = 10000
# x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)
#
#
#
#
#
# # Training configuration
# epochs = 30
# batch_size = 100
# learning_rate = 0.1
#
# # Network configuration
# layer_dims = [784, 40, 10]
# net = Network(layer_dims)
# net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

## b
if 'b' in sections:
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)
    training_accuracy = []
    test_accuracy = []
    training_loss = []
    learning_rates = [0.001, 0.01, 0.1, 1, 10]
    for learning_rate in learning_rates:
        net = Network([784, 40, 10])
        parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train, y_train, epochs=30, batch_size=10, learning_rate=learning_rate, x_test=x_test, y_test=y_test)
        training_accuracy.append(epoch_train_acc)
        test_accuracy.append(epoch_test_acc)
        training_loss.append(epoch_train_cost)

    epochs = range(1, 31)

    plt.figure(figsize=(14, 8))
    # Plot training accuracy
    plt.subplot(3, 1, 1)
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs, training_accuracy[i], label=f'LR={learning_rate}')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot test accuracy
    plt.subplot(3, 1, 2)
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs, test_accuracy[i], label=f'LR={learning_rate}')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot training loss
    plt.subplot(3, 1, 3)
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs, training_loss[i], label=f'LR={learning_rate}')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


## c
if 'c' in sections:
    net = Network([784, 40, 10])
    n_train = 60000
    n_test = 60000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)
    parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = (
        net.train(x_train, y_train, epochs=30, batch_size=10, learning_rate=0.1, x_test=x_test, y_test=y_test)
    )
    epochs = range(1, 31)
    plt.figure(figsize=(14, 8))
    plt.plot(epochs, epoch_test_acc)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if 'e' in sections:
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(60000, 60000)
    net = Network([784, 40, 10])
    parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = (
        net.train(x_train, y_train, epochs=300, batch_size=100, learning_rate=0.01, x_test=x_test, y_test=y_test)
    )
    print(epoch_test_acc[-1])
