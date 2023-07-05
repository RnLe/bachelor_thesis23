#include "VicsekModel.h"
#include "PerceptronModel.h"
#include "Perceptron.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <numeric>

    // This class only operates in 2D. Dummy variables are introduced for the code to work.

    // Constructor
    // The constructor is the same as the one in VicsekModel, except that the perceptrons are initialized,
    // depending on the NeuralNetwork mode.
    PerceptronModel::PerceptronModel(int N, double L, double v, double noise, double r, Mode mode, int k_neighbors, bool ZDimension, bool seed, std::vector<double> weights, double lambda_reg, double learning_rate, NeuralNetwork neural_network) :
        VicsekModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed), neural_network(neural_network) {
        
        // For NeuralNetwork::UnitVector, 5 perceptrons are needed, one for each dimension, the distance and the final average.
        if (neural_network == NeuralNetwork::UnitVector) {
            // First layer perceptrons
            for (int i = 0; i < 4; ++i) {
                perceptrons.push_back(Perceptron(k_neighbors, weights, lambda_reg, learning_rate, PerceptronMode::DotProduct));
            }
        }
    }

    void PerceptronModel::writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, Mode mode, int k, double noise) {
        SwarmModel::writeToFile(timesteps, filetype, N, L, v, r, mode, k, noise, "Perceptron");
    }

    // learn method
    void PerceptronModel::learn() {
        
    }

    // Update method
    // This method differentiats between the different NeuralNetwork modes.
    // Each mode is wrapped in a separate update method.
    void PerceptronModel::update() {
        switch (neural_network) {
        case NeuralNetwork::UnitVector:
            update_unitVector();
            break;
        }
    }

    // Update method for NeuralNetwork::UnitVector
    void PerceptronModel::update_unitVector() {
        std::vector<Particle> new_particles;
        update_cells();
        for (int i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            std::vector<Particle*> neighbors;
            std::vector<double> distances;
            std::tie(neighbors, distances) = get_neighbors(particle, i);

            // Reserve 4 vectors for the 4 perceptrons
            std::vector<double> input_vec_x, input_vec_y, input_vec_z, input_vec_distance;

            // Get the input vectors for each perceptron
            input_vec_x = neighbors_to_x(neighbors, distances);
            input_vec_y = neighbors_to_y(neighbors, distances);
            input_vec_z = neighbors_to_z(neighbors, distances);
            input_vec_distance = distances;

            // First neural layer
            double sum_x = perceptrons[0].forward(input_vec_x);
            double sum_y = perceptrons[1].forward(input_vec_y);
            double sum_z = perceptrons[2].forward(input_vec_z);
            double sum_distance = perceptrons[3].forward(input_vec_distance);

            // no second neural layer for now

            // Calculate angle and new positions of the particle (for now, only 2D)
            double new_angle = std::atan2(sum_y / neighbors.size(), sum_x/ neighbors.size());
            double new_polarAngle = M_PI / 2;
            if (new_angle < 0) {
                new_angle += 2 * M_PI;
            }
            if (noise != 0.0) {
                new_angle = fmod(new_angle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen1), 2 * M_PI);
            }
            
            double new_x = fmod(particle.x + v * std::cos(new_angle), L);
            if (new_x < 0) new_x += L;
            if (new_x > L) new_x -= L;
            double new_y = fmod(particle.y + v * std::sin(new_angle), L);
            if (new_y < 0) new_y += L;
            if (new_y > L) new_y -= L;
            double new_z = 0.0;
            new_particles.push_back(Particle(new_x, new_y, new_z, new_angle, new_polarAngle, neighbors));
        }
        particles = new_particles;
    }


    // These methods create three (four with z) vectors of the neighbors' angles and distances.
    // For 2D, polar coordinates are used and two vectors contain the sin and cos of the angles.
    // For 3D, spherical coordinates are used and three vectors contain the cos, sin (for x), sin, sin (for y) and cos (for z) of the angles.
    // The last vector contains the distances.
    // In neighbors_to_x, neighbors_to_y and neighbors_to_z, a case differentiation for 2D and 3D is made.
    std::vector<double> PerceptronModel::neighbors_to_x(std::vector<Particle*> neighbors, std::vector<double> distances) {
        std::vector<double> x;

        if (ZDimension) {
            for (int i = 0; i < neighbors.size(); ++i) {
                x.push_back(std::cos(neighbors[i]->angle) * std::sin(neighbors[i]->polarAngle));
            }
        } else {
            for (int i = 0; i < neighbors.size(); ++i) {
                x.push_back(std::cos(neighbors[i]->angle));
            }
        }
        
        return x;
    }

    // neighbors_to_y method
    std::vector<double> PerceptronModel::neighbors_to_y(std::vector<Particle*> neighbors, std::vector<double> distances) {
        std::vector<double> y;

        if (ZDimension) {
            for (int i = 0; i < neighbors.size(); ++i) {
                y.push_back(std::sin(neighbors[i]->angle) * std::sin(neighbors[i]->polarAngle));
            }
        } else {
            for (int i = 0; i < neighbors.size(); ++i) {
                y.push_back(std::sin(neighbors[i]->angle));
            }
        }
        
        return y;
    }

    // neighbors_to_z method
    std::vector<double> PerceptronModel::neighbors_to_z(std::vector<Particle*> neighbors, std::vector<double> distances) {
        std::vector<double> z;

        if (ZDimension) {
            for (int i = 0; i < neighbors.size(); ++i) {
                z.push_back(std::cos(neighbors[i]->polarAngle));
            }
        }
        
        return z;
    } 
