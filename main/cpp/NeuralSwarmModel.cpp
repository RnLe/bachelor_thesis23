#include "NeuralSwarmModel.h"
#include "Particle.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <numeric>

// Method to collect the input data for the neural network
std::vector<std::vector<Particle*>> NeuralSwarmModel::get_all_neighbors() {
    std::vector<std::vector<Particle*>> all_neighbors;
    for (Particle& particle : particles) {
        std::vector<Particle*> neighbors;
        std::vector<double> distances;
        std::tie(neighbors, distances) = get_neighbors(particle, &particle - &particles[0]);
        all_neighbors.push_back(neighbors);
    }
    return all_neighbors;
}

// Method to update the particles
// Contains the new angles for the particles
void NeuralSwarmModel::update_angles(std::vector<double> angles) {
    // Loop over all particles
    for (int i = 0; i < particles.size(); i++) {
        // Update the angle of the particle
        particles[i].angle = angles[i];
    }
}

// Method to execute one timestep
// Assumes that the angles of the particles have been updated
// Only the positions of the particles are updated
void NeuralSwarmModel::update() {
    // Loop over all particles
    for (Particle& particle : particles) {
        // Update the position of the particle
        particle.x += v * cos(particle.angle);
        particle.y += v * sin(particle.angle);

        // If the particle is outside the box, move it to the other side
        if (particle.x < 0) {
            particle.x += L;
        }
        else if (particle.x > L) {
            particle.x -= L;
        }
        if (particle.y < 0) {
            particle.y += L;
        }
        else if (particle.y > L) {
            particle.y -= L;
        }
    }
    // Update the cells
    update_cells();
}