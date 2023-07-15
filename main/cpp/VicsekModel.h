#ifndef VICSEKMODEL_H
#define VICSEKMODEL_H

#include "SwarmModel.h"

#include <vector>
#include <tuple>
#include <cmath>
#include <random>

class VicsekModel : public SwarmModel {
public:
                                                        VicsekModel                 (int N, double L, double v, double noise, double r, Mode mode = Mode::RADIUS, int k_neighbors = 5, bool ZDimension = false, bool seed = true)
                                                                                    : SwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed) {}
    virtual                                             ~VicsekModel                () = default;

    void                                                update                      () override;

    double                                              average_angle_particles     (const std::vector<Particle*>& particles);
    double                                              average_angle               (std::vector<double> angles);
    std::tuple<double, double, double, double, double>  get_new_particle_vicsek     (Particle& particle, std::vector<Particle*> neighbors);

    void                                                writeToFile                 (int timesteps, std::string filetype, int N, double L, double v, double r, Mode mode, int k, double noise);
};

#endif // VICSEKMODEL_H
