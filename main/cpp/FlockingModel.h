#ifndef FLOCKING_MODEL_H
#define FLOCKING_MODEL_H

#include "SwarmModel.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>

class FlockingModel : public SwarmModel {
public:
                                                        FlockingModel               (int N, double L, double v, double noise, double r, Mode mode = Mode::RADIUS, int k_neighbors = 5, bool ZDimension = false, bool seed = true, double alpha = 1.0, double distance_bias = 1.0)
                                                                                    : SwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed), distance_bias(distance_bias), alpha(alpha) {}
    virtual                                             ~FlockingModel              () = default;

    void                                                update                      () override;

    std::tuple<double, double, double, double, double>  get_new_particle_flocking   (Particle& particle, std::vector<Particle*> neighbors, std::vector<double> distances);

    void                                                writeToFileFlocking         (int timesteps, std::string filetype, int N, double L, double v, double r, Mode mode, int k, double noise, std::string model);

    double alpha = 1.0;
    double distance_bias = 1.0;
};

#endif // FLOCKING_MODEL_H
