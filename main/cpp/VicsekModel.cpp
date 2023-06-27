#include "VicsekModel.h"
#include "SwarmModel.h"
#include "Perceptron.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <omp.h>
#include <iostream>

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

            // Create temporary neighbor lists for the Quantile Method
            // This is inherently unoptimized. Urgently review the code. Its written to work for a specific case.
            std::vector<Particle*> neighbors_temp;
            std::vector<double> distances_temp;
            if (mode == SwarmModel::Mode::QUANTILE) std::tie(neighbors_temp, distances_temp) = reduceQuantileNeighbors(particle, i);

            double new_x, new_y, new_z, new_angle, new_polarAngle, new_quantileAngle;
            if (mode == SwarmModel::Mode::QUANTILE) {
                std::tie(new_x, new_y, new_z, new_quantileAngle, new_polarAngle) = get_new_particle_vicsek(particle, neighbors_temp);
            }

            std::tie(new_x, new_y, new_z, new_angle, new_polarAngle) = get_new_particle_vicsek(particle, neighbors);
            particle.x = new_x;
            particle.y = new_y;
            particle.z = ZDimension ? new_z : 0;
            particle.angle = (mode == SwarmModel::Mode::QUANTILE) ? new_quantileAngle : new_angle;
            particle.polarAngle = ZDimension ? new_polarAngle : M_PI / 2;
            particle.k_neighbors = neighbors;
        }
    }

    std::pair<std::vector<Particle*>, std::vector<double>> VicsekModel::reduceQuantileNeighbors(Particle& particle, int index) {
        std::vector<Particle*> neighbors;
        std::vector<double> distances;
        std::tie(neighbors, distances) = get_neighbors(particle, index);

        if (neighbors.size() <= k_neighbors) {
            return {neighbors, distances};
        }

        std::vector<Particle*> reduced_neighbors;
        std::vector<double> reduced_distances;
        int bin_size = neighbors.size() / k_neighbors;

        for (int i = 0; i < k_neighbors; ++i) {
            int start = i * bin_size;
            int end = (i+1) * bin_size;
            if (i == k_neighbors - 1) end = neighbors.size();  // Take the remaining neighbors into the last bin if they are not evenly divisible

            std::vector<Particle*> bin_neighbors(neighbors.begin() + start, neighbors.begin() + end);
            
            double new_x, new_y, new_z, new_angle, new_polarAngle;
            std::tie(new_x, new_y, new_z, new_angle, new_polarAngle) = wrapper_new_particle_vicsek(bin_neighbors);

            Particle* avg_particle = new Particle(new_x, new_y, new_z, new_angle, new_polarAngle);
            reduced_neighbors.push_back(avg_particle);
            // Assuming the distance to the averaged particle is the average of the distances to the particles in the bin
            reduced_distances.push_back(std::accumulate(distances.begin() + start, distances.begin() + end, 0.0) / bin_size);
        }

        return {reduced_neighbors, reduced_distances};
    }

    // This is a wrapper for the get_new_particle_vicsek() method which means over a particle list only, instead of requiring a particle explicitly.
    std::tuple<double, double, double, double, double> VicsekModel::wrapper_new_particle_vicsek(std::vector<Particle*> neighbors) {
        if (neighbors.size() == 0) {std::cerr << "Empty list in wrapper_new_particle_vicsek()!"; }
        // Get the last particle in the list and reduce the list by one, removing the last element.
        Particle* particle = neighbors.back();
        neighbors.pop_back();
        get_new_particle_vicsek(*particle, neighbors);
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