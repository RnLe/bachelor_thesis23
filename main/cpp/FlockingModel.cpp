#include "FlockingModel.h"
#include "SwarmModel.h"
#include "helperFunctions.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <omp.h>
#include <iostream>
#include <algorithm>    // for sort() and find_if()
#include <iomanip>
#include <fstream>

// Method to update the particles in the model according to the Vicsek model
void FlockingModel::update() {
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
        std::tie(new_x, new_y, new_z, new_angle, new_polarAngle) = get_new_particle_flocking(particle, neighbors, distances);

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

// Get new particle according to the flocking order parameter
std::tuple<double, double, double, double, double> FlockingModel::get_new_particle_flocking(Particle& particle, std::vector<Particle*> neighbors, std::vector<double> distances) {

    double A_sum = std::cos(particle.angle);
    double B_sum = std::sin(particle.angle);

    double distance_bias = this->distance_bias;
    double alpha = this->alpha;

    // Loop over all neighbors
    for (int i = 1; i < neighbors.size(); i++) {
        // Modify distance between the particles
        double distance = distances[i];
        // Is the distance smaller than the bias? If yes, set it to the bias
        distance = std::max(distance, distance_bias * distance_bias);
        // Power of alpha / 2
        distance = std::pow(distance, alpha / 2.0);

        // Calculate the A and B sum
        A_sum += std::cos(neighbors[i]->angle)  / distances[i];
        B_sum += std::sin(neighbors[i]->angle)  / distances[i];
    }
    double avg_angle = std::atan2(B_sum / neighbors.size(), A_sum / neighbors.size());

    // Calculate the new angle of the particle
    double new_angle;

    if (noise != 0.0) {
        if (lcg) {
            new_angle = fmod(avg_angle + lcg1.random_f() * noise - noise / 2, 2 * M_PI);
            // new_polarAngle = fmod(avg_polarAngle + lcg1.random_f() * noise - noise / 2, M_PI);
        } else {
            new_angle = fmod(avg_angle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen1), 2 * M_PI);
            // new_polarAngle = fmod(avg_polarAngle + std::uniform_real_distribution<>(-noise / 2, noise / 2)(gen2), M_PI);
        }
    }

    // Make sure the angle is between 0 and 2pi
    if (new_angle < 0) {
        new_angle += 2 * M_PI;
    }
    else if (new_angle > 2 * M_PI) {
        new_angle -= 2 * M_PI;
    }

    // Calculate the new position of the particle
    double new_x = particle.x + v * cos(new_angle);
    double new_y = particle.y + v * sin(new_angle);
    double new_z = particle.z;

    // If the particle is outside the box, move it to the other side
    if (new_x < 0) {
        new_x += L;
    }
    else if (new_x > L) {
        new_x -= L;
    }
    if (new_y < 0) {
        new_y += L;
    }
    else if (new_y > L) {
        new_y -= L;
    }

    // Calculate the new polar angle of the particle
    double new_polarAngle = std::atan2(new_y - L / 2, new_x - L / 2);

    return std::make_tuple(new_x, new_y, new_z, new_angle, new_polarAngle);
}

// Method to write the particles to a file
void FlockingModel::writeToFileFlocking(int timesteps, std::string filetype, int N, double L, double v, double r, Mode mode, int k, double noise, std::string model) {
        if (filetype == "xyz") {
            std::string modus;
            if (mode == Mode::FIXED) {
                modus = "kNeighbors";
            }
            else if (mode == Mode::RADIUS) {
                modus = "rRadius";
            }
            else if (mode == Mode::QUANTILE) {
                modus = "kQuantiles";
            }
            else if (mode == Mode::FIXEDRADIUS) {
                modus = "kFixedRadius";
            }
            std::string base = "../../data/particles_";
            std::string radiusOrK;
            switch (mode)
            {
            case Mode::FIXED:
                radiusOrK = "_k" + std::to_string(k);
                break;
            case Mode::RADIUS:
                radiusOrK = "_r" + helperFunctions::format_float(r);
                break;
            case Mode::QUANTILE:
                radiusOrK = "_k" + std::to_string(k);
                break;
            case Mode::FIXEDRADIUS:
                radiusOrK = "_r" + helperFunctions::format_float(r) + "_k" + std::to_string(k);
                break;
            default:
                radiusOrK = "_k" + std::to_string(k);
                break;
            }
            std::string parameters = "t" + std::to_string(timesteps) + "_N" + std::to_string(N) + "_L" + helperFunctions::format_float(L) + "_v" + helperFunctions::format_float(v) + "_n" + helperFunctions::format_float(noise)
            + radiusOrK + "_mode_" + modus + "_model_" + model+ "_" + (ZDimension ? "3D" : "2D") + "_alpha" + helperFunctions::format_float(alpha) + "_distanceBias" + helperFunctions::format_float(distance_bias);
            std::string filename = base + parameters + ".xyz";
            std::ofstream file(filename);
            for (int i = 0; i < timesteps; ++i) {
                file << particles.size() << "\n\n";
                for (Particle& particle : particles) {
                    file << particle.x << " " << particle.y << " " << particle.z << "\n";
                }
                // Print progress
                std::cout << "\033[1;32mProgress: " << std::fixed << std::setprecision(2) << (double)i / timesteps * 100 << "%\033[0m\r";
                std::cout.flush();
                update();
            }
            // Clear last line in console
            std::cout << "\033[2K";
            // Print 100% progress
            std::cout << "\033[1;32mProgress: 100%\033[0m\n";
            file.close();
        }
    }