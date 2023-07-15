#include "InspectionNeighbors.h"

#include <fstream>
#include <string>

// Constructor implementation
Inspector::Inspector() {
    std::vector<double> densities = helperFunctions::logspace(-1, 1, 25);
    for (int i = 0; i < densities.size(); i++) {
        N_densities.push_back(densities[i] * L * L);
    }

    // Hyperparameters
    // Choose between RADIUS and FIXED
    mode = Mode::FIXED;

    // Flags
    densityNeighbors = false;   // Figure 2b of Vicsek model
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
        {"Densities1.0", {          20,     20,     0.03,   1.0,    1,  4}},
        {"Densities2.0", {          20,     20,     0.03,   2.0,    1,  4}}
    };

    std::vector<int> ks = {1, 2, 3, 4, 5, 6};

    for (auto& setting : settings) {
        chosen_settings = setting.second;
        N = chosen_settings[0];
        L = chosen_settings[1];
        v = chosen_settings[2];
        noise = chosen_settings[3];
        r = chosen_settings[4];
    
        std::string modus;   
    
        if (mode == Mode::FIXED) {
            modus = "kNeighbors";
            for (int k : ks) {
                k_neighbors = k;
                
                std::cout << "\033[1;32m\nWriting file for " << modus << " k=" << std::to_string(k_neighbors) << " t=" << std::to_string(timesteps) << "\033[0m\n";
                equilibrate_va_VicsekValues_2b(writeToFile, "Fig2b");
            }
        }
        else if (mode == Mode::RADIUS) {
            modus = "rRadius";
            std::cout << "\033[1;32m\nWriting file for " << modus << " r=" << std::to_string(int(r)) << " t=" << std::to_string(timesteps) << "\033[0m\n";
            equilibrate_va_VicsekValues_2b(writeToFile, "Fig2b");
        }
        else if (mode == Mode::FIXEDRADIUS) {
            modus = "rkFixedRadius";
            for (int k : ks) {
                k_neighbors = k;
                
                std::cout << "\033[1;32m\nWriting file for " << modus << " k=" << std::to_string(k_neighbors) << " t=" << std::to_string(timesteps) << "\033[0m\n";
                equilibrate_va_VicsekValues_2b(writeToFile, "Fig2b");
            }
        }
    }
}

// Wrapper for runForAllNoiseLevels_Fig2a() to run all three modes
void Inspector::runForAllNoiseLevelsAndModes_Fig2a(bool writeToFile, int timesteps) {
    // Run runForAllNoiseLevels_Fig2a() for all three modes
    mode = Mode::FIXED;
    runForAllNoiseLevels_Fig2a(writeToFile, timesteps);

    mode = Mode::RADIUS;
    runForAllNoiseLevels_Fig2a(writeToFile, timesteps);

    mode = Mode::FIXEDRADIUS;
    runForAllNoiseLevels_Fig2a(writeToFile, timesteps);
}

// Wrapper for runForAllNoiseLevels_Fig2a() to run all three modes
void Inspector::runForAllNoiseLevelsAndModes_Fig2b(bool writeToFile, int timesteps) {
    // Run runForAllNoiseLevels_Fig2a() for all three modes
    mode = Mode::FIXED;
    runForAllNoiseLevels_Fig2b(writeToFile, timesteps);

    mode = Mode::RADIUS;
    runForAllNoiseLevels_Fig2b(writeToFile, timesteps);

    mode = Mode::FIXEDRADIUS;
    runForAllNoiseLevels_Fig2b(writeToFile, timesteps);
}

// Wrapper for the densitiesVicsekValues() method
void Inspector::runForAllNoiseLevels_Fig2a(bool writeToFile, int timesteps) {
    this->timesteps = timesteps;

    // Density is fixed here, rho = 4.
    this->settings = {
        // Figure 2a in Vicsek model has rho=4 fixed.
        //                              N,      L,      v,      noise,  r,  k
        {"BaseNoiseAnalysis40", {       40,     3.1,    0.03,   0.1,    1,  4}},
        {"BaseNoiseAnalysis100", {      100,    5,      0.03,   0.1,    1,  4}},
        {"BaseNoiseAnalysis400", {      400,    10,     0.03,   0.1,    1,  4}},
        {"BaseNoiseAnalysis1600", {     1600,   20,     0.03,   0.1,    1,  4}}
        //{"BaseNoiseAnalysis4000", {     4000,   31.6,   0.03,   0.1,    1,  4}}
    };

    // 50 k values
    std::vector<int> ks = {3, 4, 5, 6, 7, 8, 9, 10, 
                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                           31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                           41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
    std::vector<double> noises = helperFunctions::linspace(0., 5., 25);

    std::string modus;   
        
    if (mode == Mode::FIXED) {
        modus = "kNeighbors";
        for (int k : ks) {
            k_neighbors = k;
            
            std::cout << "\033[1;32m\nWriting file for " << modus << " k=" << std::to_string(k_neighbors) << " t=" << std::to_string(timesteps) << "\033[0m\n";
            equilibrate_va_VicsekValues_2a(writeToFile, "Fig2a", noises);
        }
    }
    else if (mode == Mode::RADIUS) {
        modus = "rRadius";
        std::cout << "\033[1;32m\nWriting file for " << modus << " r=" << std::to_string(int(r)) << " t=" << std::to_string(timesteps) << "\033[0m\n";
        equilibrate_va_VicsekValues_2a(writeToFile, "Fig2a", noises);
    }
    else if (mode == Mode::FIXEDRADIUS) {
        modus = "rkFixedRadius";
        for (int k : ks) {
            k_neighbors = k;
            
            std::cout << "\033[1;32m\nWriting file for " << modus << " k=" << std::to_string(k_neighbors) << " t=" << std::to_string(timesteps) << "\033[0m\n";
            equilibrate_va_VicsekValues_2a(writeToFile, "Fig2a", noises);
        }
    }

}

// densitiesVicsekValues function implementation. Fig2b.
void Inspector::equilibrate_va_VicsekValues_2b(bool writeToFile, std::string fileNamePrefix) {
    // Open file if writeToFile is true
    std::ofstream outFile;
    std::string modus;   
    if (mode == Mode::FIXED) {
        modus = "kNeighbors";
        modus += "_k" + std::to_string(k_neighbors);
    }
    else if (mode == Mode::RADIUS) {
        modus = "rRadius";
        modus += "_r" + std::to_string(int(r));
    }
    else if (mode == Mode::FIXEDRADIUS) {
        modus = "rkFixedRadius";
        modus += "_k" + std::to_string(k_neighbors);
        modus += "_r" + std::to_string(int(r));
    }

    std::string filename = "../../data/VicsekModel/Fig2b/" + fileNamePrefix + "_" + modus + "n" + helperFunctions::format_float(noise) + "_t" + std::to_string(timesteps) + ".txt";
    outFile.open(filename);
    if (!outFile) {
        std::cerr << "Failed to open " << filename << ". Continuing without writing to file." << std::endl;
        writeToFile = false;
    } else {
        outFile << "n,t,va,prev\n";
    }

    for (int n : N_densities) {
        VicsekModel model(n, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed);
        auto time_va = model.mean_direction_watcher(timesteps);
        std::cout << "Equilibrated for " << n << " particles after " << time_va.first << " timesteps; va = " << time_va.second.first << "\n";
        
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
    std::string modus;   
    if (mode == Mode::FIXED) {
        modus = "kNeighbors";
        modus += "_k" + std::to_string(k_neighbors);
    }
    else if (mode == Mode::RADIUS) {
        modus = "rRadius";
        modus += "_r" + std::to_string(int(r));
    }
    else if (mode == Mode::FIXEDRADIUS) {
        modus = "rkFixedRadius";
        modus += "_k" + std::to_string(k_neighbors);
        modus += "_r" + std::to_string(int(r));
    }
    if (writeToFile) {
        std::string filename = "../../data/VicsekModel/Fig2a/" + fileNamePrefix + "_" + modus + "_t" + std::to_string(timesteps) + ".txt";
        outFile.open(filename);
        if (!outFile) {
            std::cerr << "Failed to open " << filename << ". Continuing without writing to file." << std::endl;
            writeToFile = false;
        } else {
            outFile << "N,n,t,va,prev\n";
        }
    }

    for (auto& setting : this->settings) {
        chosen_settings = setting.second;
        N = chosen_settings[0];
        L = chosen_settings[1];
        v = chosen_settings[2];
        r = chosen_settings[4];
        int timesteps_ = (N == 1600)? timesteps*2 : timesteps;
        std::cout << "\033[1;32m\nSetting: " << N << " particles, L=" << L << "\033[0m\n";
        for (double noise_ : noises) {
            VicsekModel model(N, L, v, noise_, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed);
            auto time_va = model.mean_direction_watcher(timesteps_);
            std::cout << "Equilibrated for " << N << " particles, L=" << L << " with n = " << helperFunctions::format_float(noise_) << " after " << time_va.first << " timesteps; va = " << time_va.second.first << "\n";
            
            // Write to file if flag is set
            if (writeToFile) {
                outFile << N << "," << noise_ << "," << time_va.first << "," << time_va.second.first << "," << std::abs(time_va.second.second - time_va.second.first) << "\n";
            }
        }
    }

    // Close file if it was opened
    if (writeToFile) {
        outFile.close();
    }
}