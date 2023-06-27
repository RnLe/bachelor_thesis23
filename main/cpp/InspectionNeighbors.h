#ifndef INSPECTION_NEIGHBORS_H
#define INSPECTION_NEIGHBORS_H

#include "VicsekModel.h"
#include "helperFunctions.h"

#include <vector>
#include <map>
#include <string>
#include <iostream>

// Custom namespaces in use:
// helperFunctions::

/*
This file is an aggregation of tests to determine the quality and behavior of the Vicsek Model,
limiting the particles to see k neighbors instead of neighbors within a specific radius.

The goal of this file is to produce data for visualization and comparison with the original Vicsek Model.
The converging behavior is then used to make suitable predictions and adjustments to the neural network.

For small systems, the  number of neighbors is reduced to 4.
*/

class Inspector {
public:
    // Constructor
                                                    Inspector                                      ();

    // Wrapper for the densitiesVicsekValues() method
    void                                            runForAllNoiseLevels_Fig2a                     (bool writeToFile, int timesteps = 5000);
    void                                            runForAllNoiseLevels_Fig2a_quantile            (bool writeToFile, int timesteps = 5000);
    void                                            runForAllNoiseLevels_Fig2b                     (bool writeToFile, int timesteps = 5000);
    void                                            runForAllNoiseLevels_Fig2b_quantile            (bool writeToFile, int timesteps = 5000);
    // Function to check the order parameter
    void                                            equilibrate_va_VicsekValues_2a                 (bool writeToFile = false, std::string fileNamePrefix = "Fig2a", std::vector<double> noises = {});
    void                                            equilibrate_va_VicsekValues_2b                 (bool writeToFile = false, std::string fileNamePrefix = "Fig2a");

private:
    bool                                            densityNeighbors, ZDimension, seed;
    int                                             N = 100, k_neighbors = 5, timesteps;
    double                                          L = 20, v = 0.03, noise = 0.2, r = 1.0;
    std::vector<int>                                N_densities;
    std::vector<double>                             chosen_settings;
    std::map<std::string, std::vector<double>>      settings;
    SwarmModel::Mode                                mode;
};

#endif // INSPECTION_NEIGHBORS_H