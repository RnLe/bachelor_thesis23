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
        // Return all neighbors of all (or one) particle/s
        std::vector<std::vector<Particle*>> get_all_neighbors();
        std::vector<Particle*> get_neighbors_neural(int index);

        // Same as above, but returns an array of angles
        std::vector<std::vector<double>> get_all_angles();
        std::vector<double> get_angles(int index);

        // Update the angles of the particles
        void update_angles(std::vector<double> angles);
        void update_angle(int index, double angle);

        // Get local order parameter for one particle
        double get_local_order_parameter(int index);

        // Execute one timestep
        void update() override;
};

#endif // NEURAL_SWARM_MODEL_H