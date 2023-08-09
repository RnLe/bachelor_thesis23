#include "NeuralSwarmModel.h"
#include "Particle.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>

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

// Method to collect the input data for the neural network, but just for one particle
std::vector<Particle*> NeuralSwarmModel::get_neighbors_neural(int index) {
    std::vector<Particle*> neighbors;
    std::vector<double> distances;
    std::tie(neighbors, distances) = get_neighbors(particles[index], index);
    return neighbors;
}

std::vector<std::vector<double>> NeuralSwarmModel::get_all_angles() {
    std::vector<std::vector<double>> all_angles;
    for (int i = 0; i < particles.size(); i++) {
        std::vector<double> angles(particles[i].k_neighbors.size());
        angles = get_angles(i);
        all_angles.push_back(angles);
    }
    return all_angles;
}

std::vector<double> NeuralSwarmModel::get_angles(int index) {
    // Log
    // TODO: For the neural network, the angles should be relative to the particle
    Particle& particle = particles[index];
    // Determine the neighbors of the particle and save them in the particle
    std::vector<Particle*> neighbors;
    std::vector<double> distances;
    std::tie(neighbors, distances) = get_neighbors(particles[index], index);
    particle.k_neighbors = neighbors;
    particle.distances = distances;

    // Loop over k neighbors and save the angles
    // If less neighbors than k, fill the rest with 0
    // Also, make sure that the angles are positive (are 0 if negative)
    std::vector<double> angles(k_neighbors + 1);
    for (int i = 0; i < k_neighbors; i++) {
        if (i < neighbors.size()) {
            angles[i] = neighbors[i]->angle;
            if (angles[i] < 0.) {
                angles[i] += 2. * M_PI;
            }
            if (angles[i] > 2. * M_PI) {
                angles[i] = 2. * M_PI;
            }
        }
        else {
            angles[i] = 0.0;
        }
    }
    return angles;
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

// Method to update the angle of one particle
void NeuralSwarmModel::update_angle(int index, double angle) {
    particles[index].angle = angle;
}

// Method to get the local order parameter for one particle
double NeuralSwarmModel::get_local_order_parameter(int index) {
    // Get the neighbors of the particle
    std::vector<Particle*> neighbors = particles[index].k_neighbors;

    // Calculate the average angle of the neighbors
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (const Particle* p : neighbors) {
        sin_sum += std::sin(p->angle);
        cos_sum += std::cos(p->angle);
    }
    return std::hypot(cos_sum / neighbors.size(), sin_sum / neighbors.size());
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