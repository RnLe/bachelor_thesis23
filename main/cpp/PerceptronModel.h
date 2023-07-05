#ifndef PERCEPTRONMODEL_H
#define PERCEPTRONMODEL_H

#include "VicsekModel.h"
#include "Perceptron.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <numeric>

// A class to differentiate between different neural network models
enum class NeuralNetwork {
    UnitVector
    // Add further modes if necessary
};

class PerceptronModel : public VicsekModel {
public:
    enum LearningMode { UNIFORM, IMITATEVICSEK, MAXIMIZEORDER };

    // Parameterized Constructor
                                                    PerceptronModel             (int N, double L, double v, double noise, double r, Mode mode, int k_neighbors, bool ZDimension = false,
                                                                                bool seed = false, std::vector<double> weights = {}, double lambda_reg = 0.1, double learning_rate = 0.00001, NeuralNetwork neural_network = NeuralNetwork::UnitVector);

    // Methods
    void                                            writeToFile                 (int timesteps, std::string filetype, int N, double L, double v, double r, Mode mode, int k, double noise);
    void                                            learn                       ();
    void                                            update                      () override;
    void                                            update_unitVector           ();
    std::vector<double>                             neighbors_to_x              (std::vector<Particle*> neighbors, std::vector<double> distances);
    std::vector<double>                             neighbors_to_y              (std::vector<Particle*> neighbors, std::vector<double> distances);
    std::vector<double>                             neighbors_to_z              (std::vector<Particle*> neighbors, std::vector<double> distances);

    std::vector<Perceptron>                         perceptrons;
    NeuralNetwork                                   neural_network;
};

#endif // PERCEPTRONMODEL_H
