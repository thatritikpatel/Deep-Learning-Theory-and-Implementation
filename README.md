# Deep Learning: Theory and Implementation

Welcome to the Deep Learning repository! This project aims to provide a comprehensive overview of deep learning, covering both theoretical concepts and practical implementations. Whether you're a beginner or looking to deepen your understanding, this repository has something for everyone.

## Table of Contents

1. [Introduction](#introduction)
2. [Theory](#theory)
   - [What is Deep Learning?](#what-is-deep-learning)
   - [Neural Networks](#neural-networks)
   - [Activation Functions](#activation-functions)
   - [Loss Functions](#loss-functions)
   - [Optimizers](#optimizers)
   - [Forward and Backward Propagation](#forward-and-backward-propagation)
3. [Implementation](#implementation)
   - [Perceptron](#perceptron)
   - [Multilayer Neural Network](#multilayer-neural-network)
4. [Usage](#usage)
5. [Datasets](#datasets)
6. [Results and Evaluation](#results-and-evaluation)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Deep learning is a subfield of machine learning that uses neural networks with many layers to learn from large amounts of data. It has revolutionized fields such as computer vision, natural language processing, and more. This repository aims to provide both theoretical insights and practical implementations to help you get started with deep learning.

## Theory

### What is Deep Learning?

Deep learning involves training artificial neural networks with multiple layers (hence "deep") to model complex patterns in data. It allows machines to learn from data, make predictions, and make decisions without being explicitly programmed for specific tasks.

### Neural Networks

Neural networks consist of interconnected layers of nodes or "neurons" that work together to recognize patterns, make decisions, and solve problems. Key components include:

- **Input Layer**: Receives raw data and passes it to the network.
- **Hidden Layers**: Intermediate layers that perform computations and transformations on the data. The depth and width of these layers can vary depending on the complexity of the problem.
- **Output Layer**: Produces the network's output, such as class labels or regression values.

### Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common activation functions include:

- **Sigmoid**: \(\sigma(x) = \frac{1}{1 + e^{-x}}\)
  - Range: (0, 1)
  - Used in binary classification tasks.
- **ReLU (Rectified Linear Unit)**: \(f(x) = \max(0, x)\)
  - Range: [0, ∞)
  - Widely used in deep learning due to its efficiency.
- **Tanh (Hyperbolic Tangent)**: \(\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}\)
  - Range: (-1, 1)
  - Used in tasks requiring the output to be centered around zero.

### Loss Functions

Loss functions measure the difference between predicted and actual values, guiding the training process. Examples include:

- **Mean Squared Error (MSE)**: 
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
  - Used in regression tasks.
- **Cross-Entropy Loss**:
  \[
  \text{Cross-Entropy} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  \]
  - Used in classification tasks.

### Optimizers

Optimizers adjust the network's weights and biases to minimize the loss function. Common optimizers include:

- **Gradient Descent**:
  \[
  w \leftarrow w - \eta \frac{\partial L}{\partial w}
  \]
  - Updates weights by moving them in the direction of the negative gradient.
- **Stochastic Gradient Descent (SGD)**:
  - Uses a randomly selected subset of data points to compute gradients.
  - Helps in faster convergence and introduces stochasticity.
- **Adam (Adaptive Moment Estimation)**:
  - Combines the benefits of RMSprop and momentum.
  - Uses adaptive learning rates and maintains exponentially decaying averages of past gradients and squared gradients.

### Forward and Backward Propagation

- **Forward Propagation**: Passes input data through the network to generate predictions.
  - Each neuron calculates a weighted sum of its inputs, adds a bias term, and applies an activation function.
  - The process continues through all layers to produce the final output.

- **Backward Propagation**: Adjusts weights and biases to minimize error using gradients.
  - The gradient of the loss function is computed with respect to each weight and bias using the chain rule.
  - Weights and biases are updated to reduce the loss, usually through optimization algorithms like gradient descent.

## Implementation

### Perceptron

A perceptron is a simple neural network model for binary classification. Here’s a detailed implementation:

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation_function(linear_output)
                update = self.learning_rate * (target - y_pred)
                self.weights += update * xi
                self.bias += update
    
    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)
```

### Multilayer Neural Network

A more complex neural network with multiple layers. Here’s a detailed implementation:

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights_input_hidden))
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output))
        return self.output

    def backward(self, X, y, output):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden)

        self.weights_hidden_output += self.hidden.T.dot(d_output) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
```

## Usage

To use the perceptron or neural network classes, follow these steps:

1. Clone the repository.
2. Import the necessary classes.
3. Prepare your dataset.
4. Initialize and train the model.

Example:

```python
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR problem

    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
    nn.train(X, y)
    predictions = nn.forward(X)
    print(f"Predictions: {predictions}")
```

## Datasets

Include a section on datasets, explaining where to find sample datasets or how to create your own for training the models. Examples might include:

- **MNIST**: Handwritten digit classification.
- **CIFAR-10**: Image classification.
- **IMDB**: Sentiment analysis.

## Results and Evaluation

Describe how to evaluate the models and interpret the results. Include information on metrics such as accuracy, precision, recall, and F1 score. Provide examples of visualizations (e.g., confusion matrix, ROC curve) to help understand the model's performance.

## Contributing

Contributions are welcome! Whether it's bug fixes, new features, or improvements to the documentation, feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---
## Contact
- Ritik Patel - [ritik.patel129@gmail.com]
- Project Link: [https://github.com/thatritikpatel/Deep-Learning-Theory-and-Implementation/tree/main]"
---"# Deep-Learning-Theory-and-Implementation" 
