#include "VicsekModel.h"
#include "PerceptronModel.h"
#include <map>
#include <vector>

int main() {
    // Hyperparameters
    std::map<std::string, std::vector<double>> settings = {
        {"Xsmall", {20, 20, 0.03, 0.1, 1}},
        {"small", {100, 30, 0.03, 0.1, 1}},
        {"a", {300, 7, 0.03, 2.0, 1}},
        {"b", {300, 25, 0.03, 0.5, 1}},
        {"d", {300, 5, 0.03, 0.1, 1}},
        {"plot1_N40", {40, 3.1, 0.03, 0.1, 1}},
        {"large", {2000, 60, 0.03, 0.1, 1}},
        {"Xlarge", {5000, 60, 0.03, 0.1, 1}},
        {"XlargeR2", {5000, 60, 0.03, 0.1, 2}},
        {"XXlarge", {10000, 60, 0.03, 0.1, 1}},
        {"XXlargeR2", {10000, 60, 0.03, 0.1, 2}},
        {"XXlargeR5", {10000, 60, 0.03, 0.1, 5}},
        {"XXlargeR5n0", {10000, 60, 0.03, 0., 5}},
        {"XXlargeR20", {10000, 60, 0.03, 0.1, 20}},
        {"XXXlarge", {20000, 60, 0.03, 0.1, 1}},
        {"XXlargefast", {10000, 60, 0.1, 0.1, 1}}
    };

    // Choose between RADIUS and FIXED
    SwarmModel::Mode mode = SwarmModel::Mode::FIXED;

    // Flags
    bool ZDimension = false;     // 2D or 3D
    bool seed = true;            // Whether to use a seed for reproducability

    // Duration of simulation
    int timesteps = 1000;

    // Choose settings
    std::vector<double> chosen_settings = settings["small"];
    int N = chosen_settings[0];
    double L = chosen_settings[1];
    double v = chosen_settings[2];
    double noise = chosen_settings[3];
    double r = chosen_settings[4];
    int k_neighbors = 2;


    // Create model
    // VicsekModel model(N, L, v, noise, r, static_cast<SwarmModel::Mode>(mode), k_neighbors, ZDimension, seed);
    PerceptronModel model(N, L, v, noise, r, static_cast<SwarmModel::Mode>(mode), k_neighbors, ZDimension, seed);

    // Write to file
    model.writeToFile(timesteps, "xyz", N=N, L=L, v=v, r=r, mode=mode, k_neighbors, noise);

    return 0;
}
