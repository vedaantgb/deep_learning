XOR Using Multilayer Perceptron (MLP) Aim

To implement a Multilayer Perceptron (MLP) neural network using Python and train it to solve the XOR logical operation using backpropagation.

Dataset Used

The XOR dataset contains four input combinations:

Input 1 Input 2 Expected Output 0 0 0 0 1 1 1 0 1 1 1 0 Results

The MLP is trained for 10,000 epochs using the sigmoid activation function and a learning rate of 0.1. After training, the predicted outputs are close to the expected XOR outputs:

[0, 0] → 0

[0, 1] → 1

[1, 0] → 1

[1, 1] → 0

Thus, the neural network successfully learns the XOR operation.
