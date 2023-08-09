#include "VicsekModel.h"
#include "SwarmModel.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <omp.h>
#include <iostream>
#include <algorithm>    // for sort() and find_if()

// Method to update the particles in the model according to the Vicsek model
void VicsekModel::update() {
    update_cells();
    // Make a deep copy of the particles vector
    // This ensures that the order of the particles does not change while updating
    std::vector<Particle> new_particles = particles;

    // Loop over all particles
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        // Get neighbors of particle i and calculate new position and angle
        Particle& particle = particles[i];
        std::vector<Particle*> neighbors;
        std::vector<double> distances;
        std::tie(neighbors, distances) = get_neighbors(particle, i);    // In this method, the particle.cellRange is changed

        double new_x, new_y, new_z, new_angle, new_polarAngle;
        std::tie(new_x, new_y, new_z, new_angle, new_polarAngle) = get_new_particle_vicsek(particle, neighbors);

        // Update the new particle
        new_particles[i].x = new_x;
        new_particles[i].y = new_y;
        new_particles[i].z = new_z;
        new_particles[i].angle = new_angle;
        new_particles[i].polarAngle = new_polarAngle;
        new_particles[i].k_neighbors = neighbors;
        new_particles[i].distances = distances;
        new_particles[i].cellRange = particle.cellRange;
    }

    // Update the particles with the new particles (to avoid updating the particles in the model while using them)
    particles = new_particles;
}

// Method to mean over the angle of a particle list (Only 2D right now)
double VicsekModel::average_angle_particles(const std::vector<Particle*>& particles) {
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    
    // Loop over all particles and sum the sin and cos of the angle
    for (const Particle* p : particles) {
        sin_sum += std::sin(p->angle);
        cos_sum += std::cos(p->angle);
    }

    // Calculate the average angle and make sure it is between 0 and 2pi (for visualization purposes)
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

// Method to handle the Vicsek mode
std::tuple<double, double, double, double, double> VicsekModel::get_new_particle_vicsek(Particle& particle, std::vector<Particle*> neighbors) {
    double sin_sum_azimuth = 0.0;
    double cos_sum_azimuth = 0.0;
    double sin_sum_polar = 0.0;
    double cos_sum_polar = 0.0;
    
    for (Particle* p : neighbors) {
        // Check whether p is nullpointer
        if (p == nullptr) {
            continue;
        }
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
        if (lcg) {
            new_angle = fmod(avg_angle + lcg1.random_f() * noise - noise / 2, 2 * M_PI);
            // new_polarAngle = fmod(avg_polarAngle + lcg1.random_f() * noise - noise / 2, M_PI);
        } else {
            new_angle = fmod(avg_angle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen1), 2 * M_PI);
            // new_polarAngle = fmod(avg_polarAngle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen2), M_PI);
        }
    }

    double new_x;
    double new_y;
    double new_z;

    // PBC (Periodic Boundary Conditions)
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

void VicsekModel::writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, Mode mode, int k, double noise) {
    SwarmModel::writeToFile(timesteps, filetype, N, L, v, r, mode, k, noise, "Vicsek");
}