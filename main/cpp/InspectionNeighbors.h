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
This class grew into a helper class to run and write mutliple simulations.
It handles wide ranges of parameters, creates uniform file names (containing all relevant parameters) and writes the simulation data to files in the data/ folder.
*/

class Inspector {
public:
    // Constructor
                                                    Inspector                                      ();

    // Wrapper for the densitiesVicsekValues() method
    void                                            runForAllNoiseLevelsAndModes_Fig2b             (bool writeToFile, int timesteps = 5000);
    void                                            runForAllNoiseLevelsAndModes_Fig2a             (bool writeToFile, int timesteps = 5000);
    void                                            runForAllMultipleInitialConditions_density_weighted(bool writeToFile, int timesteps = 5000, int runs = 5);
    void                                            runForAllNoiseLevels_Fig2a                     (bool writeToFile, int timesteps = 5000);
    void                                            runForAllNoiseLevels_density_weighted          (bool writeToFile, int timesteps = 5000, bool random = false);
    void                                            runForAllNoiseLevels_Fig2b                     (bool writeToFile, int timesteps = 5000);
    void                                            runForAllNoiseLevels                           (bool writeToFile, int timesteps = 10000);
    // Function to check the order parameter
    void                                            equilibrate_va_VicsekValues_2a                 (bool writeToFile = false, std::string fileNamePrefix = "Fig2a", std::vector<double> noises = {});
    void                                            equilibrate_va_VicsekValues_2b                 (bool writeToFile = false, std::string fileNamePrefix = "Fig2a");
    void                                            equilibrate_density_weighted_op                (bool writeToFile = false, std::string fileNamePrefix = "density_weighted_op", std::vector<double> noises = {}, bool random = false);

    void                                            writeSimulationFile                            (std::string fileName, std::vector<double> noises);
private:
    bool                                            densityNeighbors, ZDimension, seed;
    int                                             N = 100, k_neighbors = 5, timesteps;
    double                                          L = 20, v = 0.03, noise = 0.2, r = 1.0;
    std::vector<int>                                N_densities;
    std::vector<double>                             chosen_settings;
    std::map<std::string, std::vector<double>>      settings;
    Mode                                            mode;
};

#endif // INSPECTION_NEIGHBORS_H