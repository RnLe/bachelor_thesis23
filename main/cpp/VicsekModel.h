#ifndef VICSEKMODEL_H
#define VICSEKMODEL_H

#include "SwarmModel.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <random>

class VicsekModel : public SwarmModel {
public:
    VicsekModel(int N, double L, double v, double noise, double r, SwarmModel::Mode mode = RADIUS, int k_neighbors = 5, bool ZDimension = false, bool seed = true)
        : SwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed) {}
        
    virtual ~VicsekModel() = default;

    void update() override;

    std::pair<std::vector<Particle*>, std::vector<double>> reduceQuantileNeighbors(Particle& particle, int index);
    std::tuple<double, double, double, double, double> wrapper_new_particle_vicsek(std::vector<Particle*> neighbors);
    std::tuple<double, double, double, double, double> get_new_particle_vicsek(Particle& particle, std::vector<Particle*> neighbors);

    void writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, SwarmModel::Mode mode, int k, double noise);
};

#endif // VICSEKMODEL_H
