#include "VicsekModel.h"
#include "SwarmModel.h"
#include "Perceptron.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <omp.h>

    void VicsekModel::update() {
        update_cells();
        #pragma omp parallel for
        for (int i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            std::vector<Particle*> neighbors;
            std::vector<double> distances;
            std::tie(neighbors, distances) = get_neighbors(particle, i);
            double new_x, new_y, new_z, new_angle, new_polarAngle;
            std::tie(new_x, new_y, new_z, new_angle, new_polarAngle) = get_new_particle_vicsek(particle, neighbors);
            particle.x = new_x;
            particle.y = new_y;
            particle.z = ZDimension ? new_z : 0;
            particle.angle = new_angle;
            particle.polarAngle = ZDimension ? new_polarAngle : M_PI / 2;
            particle.k_neighbors = neighbors;
        }
    }

    void VicsekModel::writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, SwarmModel::Mode mode, int k, double noise) {
        SwarmModel::writeToFile(timesteps, filetype, N, L, v, r, mode, k, noise, "Vicsek");
    }

    std::tuple<double, double, double, double, double> VicsekModel::get_new_particle_vicsek(Particle& particle, std::vector<Particle*> neighbors) {
        double sin_sum_azimuth = 0.0;
        double cos_sum_azimuth = 0.0;
        double sin_sum_polar = 0.0;
        double cos_sum_polar = 0.0;
        
        for (Particle* p : neighbors) {
            sin_sum_azimuth += std::sin(p->angle);
            cos_sum_azimuth += std::cos(p->angle);
            sin_sum_polar += std::sin(p->polarAngle);
            cos_sum_polar += std::cos(p->polarAngle);
        }

        double avg_angle = std::atan2(sin_sum_azimuth / neighbors.size(), cos_sum_azimuth / neighbors.size());
        double avg_polarAngle = std::atan2(sin_sum_polar / neighbors.size(), cos_sum_polar / neighbors.size());
        
        if (avg_angle < 0) {
            avg_angle += 2 * M_PI;
        }
        if (avg_polarAngle < 0) {
            avg_polarAngle += M_PI;
        }

        double new_angle = avg_angle;
        double new_polarAngle = avg_polarAngle;
        
        if (noise != 0.0) {
            double new_angle = fmod(avg_angle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(fastgen1), 2 * M_PI);
            double new_polarAngle = fmod(avg_polarAngle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(fastgen2), M_PI);
        }

        
        // PBC
        double new_x = fmod(particle.x + v * std::sin(new_polarAngle) * std::cos(new_angle), L);
        if (new_x < 0) new_x += L;
        
        double new_y = fmod(particle.y + v * std::sin(new_polarAngle) * std::sin(new_angle), L);
        if (new_y < 0) new_y += L;
        
        double new_z = fmod(particle.z + v * std::cos(new_polarAngle), L);
        if (new_z < 0) new_z += L;

        return std::make_tuple(new_x, new_y, new_z, new_angle, new_polarAngle);
    }