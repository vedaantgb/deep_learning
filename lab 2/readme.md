Make Moons Classification Using Neural Network Aim

To implement a neural network using TensorFlow/Keras for binary classification of the Make Moons dataset and compare batch gradient descent with mini-batch/stochastic gradient descent.

Dataset Used

The Make Moons dataset is generated using sklearn.datasets.make_moons.

Number of samples: 400

Noise: 0.20

Random state: 1

Number of input features: 2

Number of classes: 2

The input data is standardized before training.

Model Used

The neural network consists of:

Input layer with 2 features

Hidden layer with 16 neurons using ReLU activation

Hidden layer with 16 neurons using ReLU activation

Output layer with 1 neuron using Sigmoid activation

Loss function: Binary Cross-Entropy

Optimizer: SGD with learning rate 0.5

Epochs: 200

The model is trained first using the complete dataset as one batch and then using a mini-batch size of 16.

Results

After training, the model achieved:

Loss: 0.05679

Accuracy: 97.75%

The high accuracy indicates that the neural network successfully learned to classify the two classes in the Make Moons dataset.