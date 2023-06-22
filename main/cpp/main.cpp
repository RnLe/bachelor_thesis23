#include "VicsekModel.h"
#include "PerceptronModel.h"
#include <map>
#include <vector>

int main() {
    // Hyperparameters
    std::map<std::string, std::vector<double>> settings = {
        {"a", {300, 7, 0.03, 2.0, 1}},
        {"b", {300, 25, 0.03, 0.5, 1}},
        {"d", {300, 5, 0.03, 0.1, 1}},
        {"plot1_N40", {40, 3.1, 0.03, 0.1, 1}},
        {"large", {2000, 50, 0.03, 0.1, 1}}
    };

    // Choose between RADIUS and FIXED
    SwarmModel::Mode mode = SwarmModel::Mode::FIXED;

    // Choose settings
    std::vector<double> chosen_settings = settings["large"];
    int N = chosen_settings[0];
    double L = chosen_settings[1];
    double v = chosen_settings[2];
    double noise = chosen_settings[3];
    double r = chosen_settings[4];
    int k_neighbors = 5;

    // Create model
    VicsekModel model(N, L, v, noise, r, static_cast<SwarmModel::Mode>(mode), k_neighbors);

    int timesteps = 2000;
    // Write to file
    model.writeToFile(timesteps, "xyz");

    return 0;
}
