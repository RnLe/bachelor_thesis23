#ifndef NEURAL_SWARM_MODEL_H
#define NEURAL_SWARM_MODEL_H

#include "SwarmModel.h"
#include "Particle.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <numeric>

// A class for use with Tensorflow in Python
class NeuralSwarmModel : public SwarmModel {
    public:
        // Constructor
        NeuralSwarmModel(int N, double L, double v, double noise, double r, Mode mode,
                         int k_neighbors, bool ZDimension = false, bool seed = false)
                        : SwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed) {}
        ~NeuralSwarmModel() = default;

        // Methods
        // Collect the input data for the neural network
        std::vector<std::vector<Particle*>> get_all_neighbors();

        // Update the angles of the particles
        void update_angles(std::vector<double> angles);

        // Execute one timestep
        void update() override;
};

#endif // NEURAL_SWARM_MODEL_H