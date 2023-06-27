#include "InspectionNeighbors.h"

#include <fstream>
#include <string>

// Constructor implementation
Inspector::Inspector() {
    std::vector<double> densities = helperFunctions::logspace(-1, 1, 100);
    for (int i = 0; i < densities.size(); i++) {
        N_densities.push_back(densities[i] * L * L);
    }

    // Hyperparameters
    // Choose between RADIUS and FIXED
    mode = SwarmModel::Mode::FIXED;

    // Flags
    densityNeighbors = true;   // Figure 2b of Vicsek model
    ZDimension = false;        // 2D or 3D
    seed = true;               // Whether to use a seed for reproducability
}

// Wrapper for the densitiesVicsekValues() method
void Inspector::runForAllNoiseLevels_Fig2b(bool writeToFile, int timesteps) {
    this->timesteps = timesteps;

    settings = {
        // Figure 2b in Vicsek model has L=20, n=2.0 fixed.
        //                          N,      L,      v,      noise,  r,  k
        {"Densities0.1", {          20,     20,     0.03,   0.1,    1,  4}},
        {"Densities0.2", {          20,     20,     0.03,   0.2,    1,  4}},
        {"Densities0.3", {          20,     20,     0.03,   0.3,    1,  4}},
        {"Densities0.5", {          20,     20,     0.03,   0.5,    1,  4}},
        {"Densities0.75", {         20,     20,     0.03,   0.75,    1,  4}},
        {"Densities1.0", {          20,     20,     0.03,   1.0,    1,  4}},
        {"Densities1.25", {         20,     20,     0.03,   1.25,    1,  4}},
        {"Densities1.5", {          20,     20,     0.03,   1.5,    1,  4}},
        {"Densities2.0", {          20,     20,     0.03,   2.0,    1,  4}}
    };

    std::vector<int> ks = {3, 4, 5, 6, 7, 8, 9, 10};

    for (int k : ks){
        for (auto& setting : settings) {
            chosen_settings = setting.second;
            N = chosen_settings[0];
            L = chosen_settings[1];
            v = chosen_settings[2];
            noise = chosen_settings[3];
            r = chosen_settings[4];
            k_neighbors = k;
        
            std::string modus = (mode == SwarmModel::Mode::FIXED ? "kNeighbors" : "rRadius");
            std::cout << "\033[1;32m\nWriting file for " << modus << " n=" << helperFunctions::format_float(noise) << " k=" << std::to_string(k_neighbors) << " t=" << std::to_string(timesteps) << "\033[0m\n";
            equilibrate_va_VicsekValues_2b(writeToFile, "Fig2b");
        }
    }
}

// Wrapper for the densitiesVicsekValues() method
void Inspector::runForAllNoiseLevels_Fig2a(bool writeToFile, int timesteps) {
    this->timesteps = timesteps;

    // Density is fixed here, rho = 4.
    settings = {
        // Figure 2a in Vicsek model has rho=4 fixed.
        //                          N,      L,      v,      noise,  r,  k
        {"BaseNoiseAnalysis", {     40,     3.1,    0.03,   0.1,    1,  4}},
        {"BaseNoiseAnalysis", {     100,    5,      0.03,   0.1,    1,  4}},
        {"BaseNoiseAnalysis", {     400,    1,      0.03,   0.1,    1,  4}},
        {"BaseNoiseAnalysis", {     1600,   20,     0.03,   0.1,    1,  4}},
        {"BaseNoiseAnalysis", {     4000,   31.6,   0.03,   0.1,    1,  4}},
    };

    std::vector<int> ks = {3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> noises = helperFunctions::linspace(0., 5., 50);

    for (int k : ks){
        k_neighbors = k;
        
        std::string modus = (mode == SwarmModel::Mode::FIXED ? "kNeighbors" : "rRadius");
        std::cout << "\033[1;32m\nWriting file for " << modus << " N=" << std::to_string(N) << " k=" << std::to_string(k_neighbors) << " t=" << std::to_string(timesteps) << "\033[0m\n";
        equilibrate_va_VicsekValues_2a(writeToFile, "Fig2a", noises);
    }
}

// densitiesVicsekValues function implementation. Fig2b.
void Inspector::equilibrate_va_VicsekValues_2b(bool writeToFile, std::string fileNamePrefix) {
    // Open file if writeToFile is true
    std::ofstream outFile;
    if (writeToFile) {
        std::string modus = (mode == SwarmModel::Mode::FIXED ? "kNeighbors_" : "rRadius_");
        std::string filename = "../../data/VicsekModel/" + fileNamePrefix + "_" + modus + "n" + helperFunctions::format_float(noise) + "_k" + std::to_string(k_neighbors) + "_t" + std::to_string(timesteps) + ".txt";
        outFile.open(filename);
        if (!outFile) {
            std::cerr << "Failed to open output file. Continuing without writing to file." << std::endl;
            writeToFile = false;
        } else {
            outFile << "n,t,va,prev\n";
        }
    }

    for (int n : N_densities) {
        VicsekModel model(n, L, v, noise, r, static_cast<SwarmModel::Mode>(mode), k_neighbors, ZDimension, seed);
        auto time_va = model.mean_direction_watcher(timesteps);
        std::cout << "Equilibrated for " << n << " particles after " << time_va.first << " timesteps; va = " << time_va.second.first << " , difference to previous = " << std::abs(time_va.second.second - time_va.second.first) << "\n";
        
        // Write to file if flag is set
        if (writeToFile) {
            outFile << n << "," << time_va.first << "," << time_va.second.first << "," << std::abs(time_va.second.second - time_va.second.first) << "\n";
        }
    }

    // Close file if it was opened
    if (writeToFile) {
        outFile.close();
    }
}

// densitiesVicsekValues function implementation. Fig2a.
void Inspector::equilibrate_va_VicsekValues_2a(bool writeToFile, std::string fileNamePrefix, std::vector<double> noises) {
    // Open file if writeToFile is true
    std::ofstream outFile;
    if (writeToFile) {
        std::string modus = (mode == SwarmModel::Mode::FIXED ? "kNeighbors_" : "rRadius_");
        std::string filename = "../../data/VicsekModel/" + fileNamePrefix + "_" + modus + "_k" + std::to_string(k_neighbors) + "_t" + std::to_string(timesteps) + ".txt";
        outFile.open(filename);
        if (!outFile) {
            std::cerr << "Failed to open output file. Continuing without writing to file." << std::endl;
            writeToFile = false;
        } else {
            outFile << "N,n,t,va,prev\n";
        }
    }

    for (auto& setting : settings) {
        chosen_settings = setting.second;
        N = chosen_settings[0];
        L = chosen_settings[1];
        v = chosen_settings[2];
        r = chosen_settings[4];
        for (double noise_ : noises) {
            VicsekModel model(N, L, v, noise_, r, static_cast<SwarmModel::Mode>(mode), k_neighbors, ZDimension, seed);
            auto time_va = model.mean_direction_watcher(timesteps);
            std::cout << "Equilibrated for " << N << " particles with n = " << helperFunctions::format_float(noise_) << "after " << time_va.first << " timesteps; va = " << time_va.second.first << " , difference to previous = " << std::abs(time_va.second.second - time_va.second.first) << "\n";
            
            // Write to file if flag is set
            if (writeToFile) {
                outFile << N << "," << noise << "," << time_va.first << "," << time_va.second.first << "," << std::abs(time_va.second.second - time_va.second.first) << "\n";
            }
        }
    }

    // Close file if it was opened
    if (writeToFile) {
        outFile.close();
    }
}