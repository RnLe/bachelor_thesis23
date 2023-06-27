#include "VicsekModel.h"
#include "SwarmModel.h"
#include "Perceptron.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <omp.h>
#include <iostream>
#include <algorithm>    // for sort() and find_if()

    void VicsekModel::update() {
        // TODO: The particles are updated while being used to update other particles.
        // Change it to save the new particles in a new, temporary particle vector and overwrite the particles at the end of the for loop
        update_cells();
        #pragma omp parallel for
        for (int i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            std::vector<Particle*> neighbors;
            std::vector<double> distances;
            std::tie(neighbors, distances) = get_neighbors(particle, i);

            double new_x, new_y, new_z, new_angle, new_polarAngle;
            if (mode == SwarmModel::Mode::QUANTILE) {
                std::tie(new_x, new_y, new_z, new_angle, new_polarAngle) = get_new_particle_quantile(particle, neighbors);
            } else {

                std::tie(new_x, new_y, new_z, new_angle, new_polarAngle) = get_new_particle_vicsek(particle, neighbors);
            }
            particle.x = new_x;
            particle.y = new_y;
            particle.z = ZDimension ? new_z : 0;
            particle.angle = new_angle;
            particle.polarAngle = ZDimension ? new_polarAngle : M_PI / 2;
            particle.k_neighbors = neighbors;
        }
    }

    // Method to mean over the angle of a particle list (Only 2D right now)
    double VicsekModel::average_angle_particles(const std::vector<Particle*>& particles) {
        double sin_sum = 0.0;
        double cos_sum = 0.0;
        
        for (const Particle* p : particles) {
            sin_sum += std::sin(p->angle);
            cos_sum += std::cos(p->angle);
        }

        double avg_angle = std::atan2(sin_sum / particles.size(), cos_sum / particles.size());
        if (avg_angle < 0) {
            avg_angle += 2 * M_PI;
        }

        return avg_angle;
    }

    // Method to mean over a list of angles (Only 2D right now)
    // Averages without applying noise
    double VicsekModel::average_angle(const std::vector<double> angles) {
        double sin_sum = 0.0;
        double cos_sum = 0.0;
        
        for (double angle : angles) {
            sin_sum += std::sin(angle);
            cos_sum += std::cos(angle);
        }

        double avg_angle = std::atan2(sin_sum / angles.size(), cos_sum / angles.size());
        if (avg_angle < 0) {
            avg_angle += 2 * M_PI;
        }

        return avg_angle;
    }

    // Method to handle the quantile mode
    std::tuple<double, double, double, double, double> VicsekModel::get_new_particle_quantile(Particle& particle, std::vector<Particle*> neighbors) {
        std::vector<Particle*> quantile_neighbors;
        std::vector<double> quantile_angles;
        
        if (neighbors.size() > k_neighbors) {
            // Sort neighbors by angle
            std::sort(neighbors.begin(), neighbors.end(), [](const Particle* a, const Particle* b) {
                return a->angle < b->angle;
            });
            
            // Separate particles in dynamic bins
            int bin_size = neighbors.size() / k_neighbors;
            for (int i = 0; i < k_neighbors; ++i) {
                int start = i * bin_size;
                int end = (i+1) * bin_size;
                if (i == k_neighbors - 1) {
                    end = neighbors.size();
                }
                
                std::vector<Particle*> bin_neighbors(neighbors.begin() + start, neighbors.begin() + end);
                double bin_avg_angle = average_angle_particles(bin_neighbors);
                
                quantile_neighbors.insert(quantile_neighbors.end(), bin_neighbors.begin(), bin_neighbors.end());
                quantile_angles.push_back(bin_avg_angle);
            }
        } else {
            quantile_neighbors = neighbors;
            for (const Particle* p : neighbors) {
                quantile_angles.push_back(p->angle);
            }
        }
        
        // Use the angles list to average the new angle and determine the new position
        // (Only 2D right now)
        double new_angle = average_angle(quantile_angles);
        double new_polarAngle = 0.0;
        
        if (noise != 0.0) {
            new_angle = fmod(new_angle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen1), 2 * M_PI);
        }

        double new_x;
        double new_y;
        double new_z;

        // PBC
        if (ZDimension) {
            new_x = fmod(particle.x + v * std::sin(new_polarAngle) * std::cos(new_angle), L);
            new_y = fmod(particle.y + v * std::sin(new_polarAngle) * std::sin(new_angle), L);
            new_z = fmod(particle.z + v * std::cos(new_polarAngle), L);
            if (new_z < 0) new_z += L;
        } else {
            new_x = fmod(particle.x + v * std::cos(new_angle), L);
            new_y = fmod(particle.y + v * std::sin(new_angle), L);            
            new_z = 0.;
        }
        if (new_x < 0) new_x += L;
        if (new_y < 0) new_y += L;

        return std::make_tuple(new_x, new_y, new_z, new_angle, new_polarAngle);
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
            new_angle = fmod(avg_angle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen1), 2 * M_PI);
            new_polarAngle = fmod(avg_polarAngle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen2), M_PI);
        }

        double new_x;
        double new_y;
        double new_z;

        // PBC
        if (ZDimension) {
            new_x = fmod(particle.x + v * std::sin(new_polarAngle) * std::cos(new_angle), L);
            new_y = fmod(particle.y + v * std::sin(new_polarAngle) * std::sin(new_angle), L);
            new_z = fmod(particle.z + v * std::cos(new_polarAngle), L);
            if (new_z < 0) new_z += L;
        } else {
            new_x = fmod(particle.x + v * std::cos(new_angle), L);
            new_y = fmod(particle.y + v * std::sin(new_angle), L);            
            new_z = 0.;
        }
        if (new_x < 0) new_x += L;
        if (new_y < 0) new_y += L;


        return std::make_tuple(new_x, new_y, new_z, new_angle, new_polarAngle);
    }

    void VicsekModel::writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, SwarmModel::Mode mode, int k, double noise) {
        SwarmModel::writeToFile(timesteps, filetype, N, L, v, r, mode, k, noise, "Vicsek");
    }