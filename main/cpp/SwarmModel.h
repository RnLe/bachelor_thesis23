#ifndef SWARM_MODEL_H
#define SWARM_MODEL_H

#include <vector>
#include <string>
#include "Particle.h"

class SwarmModel {
public:
    enum Mode { RADIUS, FIXED };

    // Constructor
    SwarmModel(int N, double L, double v, double noise, double r, Mode mode, int k_neighbors);
    virtual ~SwarmModel() = default;

    // Methods
    void update_cells();
    std::vector<double> get_density_hist();
    std::vector<double> get_dynamic_radius();
    std::pair<std::vector<Particle*>, std::vector<double>> get_neighbors(Particle& particle, int index);
    virtual void update() = 0;
    double va();
    void writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, SwarmModel::Mode mode);
    std::string format_float(float number);

protected:
    // Member variables
    int N, mode, k_neighbors, num_cells, cellSpan = 0;
    double L, v, noise, r, density;
    std::vector<Particle> particles;
    std::vector<std::vector<std::vector<int>>> cells;
    std::vector<int> mode1_cells;
};

#endif // SWARM_MODEL_H
