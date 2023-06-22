#include "VicsekModel.h"
#include "PerceptronModel.h"
#include "Perceptron.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <numeric>

    // Constructor
    PerceptronModel::PerceptronModel(int N, double L, double v, double noise, double r, Mode mode, int k_neighbors, double learning_rate, std::vector<double> weights)
        : VicsekModel(N, L, v, noise, r, mode, k_neighbors), learning_rate(learning_rate) {
        if (weights.empty()) {
            perceptron = Perceptron(k_neighbors + 1);
        } else {
            perceptron = Perceptron(k_neighbors + 1, weights);
        }
    }

    // learn method
    void PerceptronModel::learn() {
        std::vector<Particle> new_particles;
        update_cells();
        for (int i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            std::vector<Particle*> neighbors;
            std::vector<double> distances;
            std::tie(neighbors, distances) = get_neighbors(particle, i);
            std::vector<double> input_vec = neighbors_to_input_vec(neighbors, distances);
            double error = compute_error(particle, neighbors, input_vec);
            perceptron.update_weights(input_vec, error, learning_rate);
            double new_angle = perceptron.forward(input_vec);
            double new_x = fmod(particle.x + v * std::cos(new_angle), L);
            double new_y = fmod(particle.y + v * std::sin(new_angle), L);
            new_particles.push_back(Particle(new_x, new_y, new_angle, neighbors));
        }
        particles = new_particles;
    }

    // update method
    void PerceptronModel::update() {
        std::vector<Particle> new_particles;
        update_cells();
        for (int i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            std::vector<Particle*> neighbors;
            std::vector<double> distances;
            std::tie(neighbors, distances) = get_neighbors(particle, i);
            std::vector<double> input_vec = neighbors_to_input_vec(neighbors, distances);
            double new_angle = fmod(perceptron.forward(input_vec) + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen), 2 * M_PI);
            double new_x = fmod(particle.x + v * std::cos(new_angle), L);
            double new_y = fmod(particle.y + v * std::sin(new_angle), L);
            new_particles.push_back(Particle(new_x, new_y, new_angle, neighbors));
        }
        particles = new_particles;
    }

    // neighbors_to_input_vec method
    std::vector<double> PerceptronModel::neighbors_to_input_vec(std::vector<Particle*> neighbors, std::vector<double> distances) {
        std::vector<double> input_vec;
        for (Particle* p : neighbors) {
            input_vec.push_back(p->angle);
        }
        return input_vec;
    }

    // compute_error method
    double PerceptronModel::compute_error(Particle& particle, std::vector<Particle*> neighbors, std::vector<double> input_vec) {
        double target = get_target(particle, neighbors);
        double prediction = get_prediction(input_vec);
        
        // Compute the error
        double error = fmod(abs(target - M_PI) - abs(prediction - M_PI), 2 * M_PI);
        
        return error * error;  // Return the squared error
    }


    // get_target method
    double PerceptronModel::get_target(Particle& particle, std::vector<Particle*> neighbors) {
        double new_x, new_y, new_angle;
        std::tie(new_x, new_y, new_angle) = get_new_particle_vicsek(particle, neighbors);
        return new_angle;
    }

    // get_prediction method
    double PerceptronModel::get_prediction(std::vector<double> input_vec) {
        return perceptron.forward(input_vec);
    }