#include <vector>
#include <cmath>
#include "Perceptron.h"

Perceptron::Perceptron(int input_dim, std::vector<double> weights, double lambda_reg)
    : weights(weights.empty() ? std::vector<double>(input_dim, 1.0 / input_dim) : weights), lambda_reg(lambda_reg) {}

double Perceptron::forward(const std::vector<double>& input_vec) {
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (int i = 0; i < input_vec.size(); ++i) {
        sin_sum += weights[i] * std::sin(input_vec[i]);
        cos_sum += weights[i] * std::cos(input_vec[i]);
    }
    double average_angle = std::atan2(sin_sum, cos_sum);
    if (average_angle < 0) {
        average_angle += 2 * M_PI;
    }
    double output = average_angle > 0 ? average_angle : average_angle / 1000;
    return output;
}

void Perceptron::update_weights(const std::vector<double>& input_vec, double error, double learning_rate) {
    for (int i = 0; i < weights.size(); ++i) {
        double gradient = error * input_vec[i];
        weights[i] -= learning_rate * gradient;
    }
}