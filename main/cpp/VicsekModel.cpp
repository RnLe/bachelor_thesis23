#include "VicsekModel.h"
#include "SwarmModel.h"
#include "Perceptron.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <random>

    void VicsekModel::update() {
        update_cells();
        for (int i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            std::vector<Particle*> neighbors;
            std::vector<double> distances;
            std::tie(neighbors, distances) = get_neighbors(particle, i);
            double new_x, new_y, new_angle;
            std::tie(new_x, new_y, new_angle) = get_new_particle_vicsek(particle, neighbors);
            particle.x = new_x;
            particle.y = new_y;
            particle.angle = new_angle;
            particle.k_neighbors = neighbors;
        }
    }

    std::tuple<double, double, double> VicsekModel::get_new_particle_vicsek(Particle& particle, std::vector<Particle*> neighbors) {
        double sin_sum = 0.0;
        double cos_sum = 0.0;
        for (Particle* p : neighbors) {
            sin_sum += std::sin(p->angle);
            cos_sum += std::cos(p->angle);
        }
        double avg_angle = std::atan2(sin_sum / neighbors.size(), cos_sum / neighbors.size());
        if (avg_angle < 0) {
            avg_angle += 2 * M_PI;
        }
        double new_angle = fmod(avg_angle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen), 2 * M_PI);
        // PBC
        double new_x = fmod(particle.x + v * std::cos(new_angle), L);
        if (new_x < 0) new_x += L;
        double new_y = fmod(particle.y + v * std::sin(new_angle), L);
        if (new_y < 0) new_y += L;

        return std::make_tuple(new_x, new_y, new_angle);
    }
