import numpy as np
from numpy.typing import ArrayLike


class Perceptron:
    def __init__(self, input_dim: int, weights = [], lambda_reg: float = 0.00001):
        """
        Initialize the perceptron.

        Args:
            input_dim (int): The dimension of the input vector.
        """
        # Initialize the weights to small random values
        if len(weights) == 0:
            self.weights = [1 / input_dim] * input_dim
        else:
            self.weights = weights
            
        # Regularization parameter
        self.lambda_reg = lambda_reg

    def forward(self, input_vec: ArrayLike):
        """
        Perform the forward pass of the perceptron.

        Args:
            input_vec (np.array): The input vector.

        Returns:
            float: The output of the perceptron.
        """
        # Compute the weighted sum of the inputs
        
        # Zerlegen Sie die Winkel in ihre Sinus- und Kosinus-Komponenten
        sin_components = np.sin(input_vec)
        cos_components = np.cos(input_vec)

        # Gewichten Sie die Komponenten und berechnen Sie das arithmetische Mittel
        weighted_sin = np.sum(self.weights * sin_components)
        weighted_cos = np.sum(self.weights * cos_components)

        # Rechnen Sie den resultierenden Winkel zurück
        average_angle = np.arctan2(weighted_sin, weighted_cos)
        
        # Convert the average angle to the range 0 to 2pi
        if average_angle < 0:
            average_angle += 2 * np.pi
        
        ###weighted_sum = np.dot(self.weights, input_vec)

        # Apply the ReLU activation function
        output = average_angle if average_angle > 0 else average_angle / 1000

        return output

    def update_weights(self, input_vec: ArrayLike, error: float, learning_rate: float):
        """
        Update the weights of the perceptron using gradient descent.

        Args:
            input_vec (np.array): The input vector.
            error (float): The error of the perceptron's output.
            learning_rate (float): The learning rate for gradient descent.
        """
        # Compute the gradient of the error with respect to the weights
        gradient = error * input_vec
        
        # Add the regularization term to the gradient
        # gradient += self.lambda_reg * self.weights

        # Update the weights using gradient descent
        self.weights -= learning_rate * gradient