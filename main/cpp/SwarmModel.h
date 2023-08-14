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

/*
Virtual base class for swarm models. Provides critical functionality which is universal between models.
Critical components are: 
    -   The update_cells() method, which updates the cells for the swarm model. This is used to speed up the neighbor search.
    -   The get_neighbors() method, which returns the neighbors of a particle.
    -   The virtual update() method, which updates the swarm model. 
    -   The sweep() method, which sweeps the swarm model for a given number of timesteps.
    -   The writeToFile() method, which writes the swarm model to a file. This is used to write the swarm model to a file.
    -   Order parameter methods, which calculate the order parameter of the swarm model.
*/
class SwarmModel {
public:

    // Constructor
                                                            SwarmModel                  (int N, double L, double v, double noise, double r, Mode mode,
                                                                                        int k_neighbors, bool ZDimension = false, bool seed = false);
    virtual                                                 ~SwarmModel                 () = default;

    // Methods  
    void                                                    update_cells                ();
    std::vector<double>                                     get_density_hist3D          ();     // Not used
    std::vector<double>                                     get_dynamic_radius          ();     // Not used
    std::pair<std::vector<Particle*>, std::vector<double>>  get_neighbors               (Particle& particle, int index);
    virtual void                                            update                      () {};
    double                                                  mean_direction2D            ();
    std::pair<double, double>                               mean_direction3D            ();
    double                                                  density_weighted_op         ();
    // Watchers run a simulation and calculate/mean an order parameter, considering correlations between timesteps by measuring the OP only at each sweep.
    // DEPRECATED: While these watchers work, it is way more efficient to write the simulations (pos, angles, ...) to a file (sweep by sweep)
    // and then calculate the order parameters from the files. This is done using python notebooks.
    std::pair<int, std::pair<double, double>>               density_weighted_op_watcher (int timeLimit = -1, double tolerance = 1e-6);
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