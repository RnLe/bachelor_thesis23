#ifndef SWARM_MODEL_H
#define SWARM_MODEL_H

#include <vector>
#include <random>
#include <string>
#include "Particle.h"

class SwarmModel {
public:
    enum Mode { RADIUS, FIXED };

    // Constructor
    SwarmModel(int N, double L, double v, double noise, double r, Mode mode, int k_neighbors, bool ZDimension = false, bool seed = false);
    virtual ~SwarmModel() = default;

    // Methods
    void update_cells();
    std::vector<double> get_density_hist3D();
    std::vector<double> get_dynamic_radius();
    std::pair<std::vector<Particle*>, std::vector<double>> get_neighbors(Particle& particle, int index);
    virtual void update() = 0;
    double mean_direction2D();
    std::pair<double, double> mean_direction3D();
    void writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, SwarmModel::Mode mode, int k, double noise, std::string model);
    std::string format_float(float number);

protected:
    // Member variables
    int N, mode, k_neighbors, num_cells, cellSpan = 0;
    double L, v, noise, r, density2D, density3D;
    std::vector<Particle> particles;
    std::vector<std::vector<std::vector<std::vector<int>>>> cells3D;
    std::vector<std::vector<std::vector<int>>> cells2D;
    std::vector<int> mode1_cells;

    unsigned int seed1 = 123; // Seed for 1st generator
    unsigned int seed2 = 456; // Seed for 2nd generator
    unsigned int seed3 = 789; // Seed for 3rd generator

    // Toggle seed manually - seed flag
    bool seed, ZDimension;

    std::mt19937 gen1;
    std::mt19937 gen2;
    std::mt19937 gen3;
    std::minstd_rand fastgen1;
    std::minstd_rand fastgen2;
    std::minstd_rand fastgen3;     
    std::random_device rd;
};

#endif // SWARM_MODEL_H
