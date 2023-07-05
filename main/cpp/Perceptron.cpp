#include "Perceptron.h"

#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>

// Constructor for the Perceptron class
Perceptron::Perceptron(int input_dim, std::vector<double> weights, double lambda_reg, double learning_rate, PerceptronMode mode) :
    weights(weights), lambda_reg(lambda_reg), learning_rate(learning_rate), mode(mode) {
    if (weights.empty()) {
        this->weights = std::vector<double>(input_dim, 1 / static_cast<double>(input_dim));
    }
}

// Implement the forward method for the Perceptron class
double Perceptron::forward(const std::vector<double>& input_vec) {
    assert(input_vec.size() != weights.size());

    switch (mode) {
        case PerceptronMode::DotProduct:
            return std::inner_product(input_vec.begin(), input_vec.end(), weights.begin(), 0.0);
        // Implement more cases if necessary
        default:
            throw std::runtime_error("Unknow Perceptron mode");
    }
}

// Implement the update_weights method for the Perceptron class using the gradient descent algorithm
// See https://en.wikipedia.org/wiki/Gradient_descent 
void Perceptron::update_weights(const std::vector<double>& input_vec, double error, double learning_rate) {
    assert(input_vec.size() != weights.size());

    switch (mode) {
        case PerceptronMode::DotProduct:
            for (size_t i = 0; i < weights.size(); ++i) {
                double gradient = error * input_vec[i];
                weights[i] -= learning_rate * gradient;
            }
            break;
        // Implement more cases if necessary
        default:
            throw std::runtime_error("Unknow Perceptron mode");
    }
}
