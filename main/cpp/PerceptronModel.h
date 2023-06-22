#ifndef PERCEPTRONMODEL_H
#define PERCEPTRONMODEL_H

#include "VicsekModel.h"
#include "Perceptron.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <numeric>

class PerceptronModel : public VicsekModel {
public:
    enum LearningMode { UNIFORM, IMITATEVICSEK, MAXIMIZEORDER };

    // Parameterized Constructor
    PerceptronModel(int N, double L, double v, double noise, double r, Mode mode, int k_neighbors, double learning_rate, std::vector<double> weights);

    // Methods
    void learn();
    void update() override;
    std::vector<double> neighbors_to_input_vec(std::vector<Particle*> neighbors, std::vector<double> distances);
    double compute_error(Particle& particle, std::vector<Particle*> neighbors, std::vector<double> input_vec);
    double get_target(Particle& particle, std::vector<Particle*> neighbors);
    double get_prediction(std::vector<double> input_vec);

private:
    double learning_rate;
    Perceptron perceptron;
};

#endif // PERCEPTRONMODEL_H
