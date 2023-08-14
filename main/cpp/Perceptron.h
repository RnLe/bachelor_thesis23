#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <cmath>

/*
DEPRECATED: This class isn't used anymore. It was used to test the perceptron algorithm on the Vicsek model.
*/

enum class PerceptronMode {
    DotProduct
    // Add further modes if necessary
};

class Perceptron {
public:
    // Constructor
                                                    Perceptron          () = default;
                                                    Perceptron          (int input_dim, std::vector<double> weights = {}, double lambda_reg = 0.1, double learning_rate = 0.00001, PerceptronMode mode = PerceptronMode::DotProduct);
                                                    ~Perceptron         () = default;

    // Methods
    double                                          forward             (const std::vector<double>& input_vec);
    void                                            update_weights      (const std::vector<double>& input_vec, double error, double learning_rate);

public:
    std::vector<double>                             weights;
    double                                          lambda_reg, learning_rate;
    PerceptronMode                                  mode;
};

#endif // PERCEPTRON_H
