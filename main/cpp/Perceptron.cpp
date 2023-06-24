#include <vector>
#include <cmath>
#include "Perceptron.h"

Perceptron::Perceptron(int input_dim, bool ZDimension, std::vector<double> weights, double lambda_reg)
    : lambda_reg(lambda_reg) {
        int dimension_multiplier = ZDimension ? 3 : 2;
        this->weights = weights.empty() ? std::vector<double>(dimension_multiplier * input_dim, 1.0 / input_dim) : weights;
    }

double Perceptron::forward(const std::vector<double>& input_vec, bool ZDimension) {
    // TODO: Respect spherical coordinates in 3D.
    // The sums change when in 3D. The input vector is unchanged.
    int dimension_multiplier = ZDimension ? 3 : 2;
    // Separation in sin and cos within neuron
    double cos_sum = 0.0;
    double sin_sum = 0.0;
    double sinTheta_sum = 0.0;
    int index_break = input_vec.size() / dimension_multiplier;
    for (int i = 0; i < index_break; ++i) {
        cos_sum += weights[i] * input_vec[i];
    }
    for (int i = index_break; i < 2 * index_break; ++i) {
        sin_sum += weights[i] * input_vec[i];
    }
    if (ZDimension) {
        for (int i = 2 * index_break; i < 3 * index_break; ++i) {
            sinTheta_sum += weights[i] * input_vec[i];
        }
    }
    // cos_sum /= index_break;
    // sin_sum /= index_break;
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