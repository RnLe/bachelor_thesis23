#ifndef SWARM_MODEL_H
#define SWARM_MODEL_H

#include "Particle.h"
#include "LCG.h"

#include <vector>
#include <random>
#include <string>

enum class Mode {
    RADIUS, FIXED, QUANTILE, FIXEDRADIUS
};
class SwarmModel {
public:

    // Constructor
                                                            SwarmModel                  (int N, double L, double v, double noise, double r, Mode mode,
                                                                                        int k_neighbors, bool ZDimension = false, bool seed = false);
    virtual                                                 ~SwarmModel                 () = default;

    // Methods  
    void                                                    update_cells                ();
    std::vector<double>                                     get_density_hist3D          ();
    std::vector<double>                                     get_dynamic_radius          ();
    std::pair<std::vector<Particle*>, std::vector<double>>  get_neighbors               (Particle& particle, int index);
    virtual void                                            update                      () {};
    double                                                  mean_direction2D            ();
    std::pair<double, double>                               mean_direction3D            ();
    double                                                  density_weighted_op();
    std::pair<int, std::pair<double, double>>               density_weighted_op_watcher(int timeLimit = -1, double tolerance = 1e-6);
    std::pair<int, std::pair<double, double>>               mean_direction_watcher      (int timeLimit = -1, double tolerance = 1e-6);
    void                                                    sweep                       (int timesteps);
    void                                                    writeToFile                 (int timesteps, std::string filetype, int N, double L, double v, double r,
                                                                                        Mode mode, int k, double noise, std::string model);

public:
    // Member variables
    int                                                     N, k_neighbors, num_cells, cellSpan = 0, k_reserve;
    Mode                                                    mode;
    double                                                  L, v, noise, r, density2D, density3D;
    std::vector<Particle>                                   particles;
    std::vector<std::vector<std::vector<std::vector<int>>>> cells3D;
    std::vector<std::vector<std::vector<int>>>              cells2D;
    std::vector<int>                                        mode1_cells;

    unsigned int                                            seed1 = 123456789; // Seed for 1st generator
    unsigned int                                            seed2 = 456789123; // Seed for 2nd generator
    unsigned int                                            seed3 = 789123456; // Seed for 3rd generator

    // Toggle seed and random generator manually - seed flag
    bool                                                    seed, ZDimension, lcg = true;

    // Random generators
    LCG                                                     lcg1;

    std::mt19937                                            gen1;
    std::mt19937                                            gen2;
    std::mt19937                                            gen3;
    std::minstd_rand                                        fastgen1;
    std::minstd_rand                                        fastgen2;
    std::minstd_rand                                        fastgen3;     
    std::random_device                                      rd;
};

#endif // SWARM_MODEL_H