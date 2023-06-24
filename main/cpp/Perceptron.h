#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <cmath>

class Perceptron {
public:
    // Constructor
    Perceptron() = default;

    // Constructor
    Perceptron(int input_dim, bool ZDimension, std::vector<double> weights = {}, double lambda_reg = 0.00001);

    // Methods
    double forward(const std::vector<double>& input_vec, bool ZDimension);
    void update_weights(const std::vector<double>& input_vec, double error, double learning_rate);

private:
    std::vector<double> weights;
    double lambda_reg;
};

#endif // PERCEPTRON_H
