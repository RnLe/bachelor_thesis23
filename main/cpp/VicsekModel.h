#ifndef VICSEKMODEL_H
#define VICSEKMODEL_H

#include "SwarmModel.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <random>

class VicsekModel : public SwarmModel {
public:
    VicsekModel(int N, double L, double v, double noise, double r, SwarmModel::Mode mode = RADIUS, int k_neighbors = 5)
        : SwarmModel(N, L, v, noise, r, mode, k_neighbors) {}
        
    virtual ~VicsekModel() = default;


    void update() override;

    std::tuple<double, double, double> get_new_particle_vicsek(Particle& particle, std::vector<Particle*> neighbors);

protected:
    std::default_random_engine gen;
};

#endif // VICSEKMODEL_H
