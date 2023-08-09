#include "VicsekModel.h"
#include "PerceptronModel.h"
#include "InspectionNeighbors.h"
#include "FlockingModel.h"

#include <map>
#include <vector>

int main() {
    // Hyperparameters
    std::map<std::string, std::vector<double>> settings = {
        //                  N,      L,      v,      noise,  r
        {"XXsmall", {       5,      4,      0.03,   0.1,    1}},
        {"Xsmall", {        20,     6,      0.03,   0.1,    1}},
        {"small", {         100,    30,     0.03,   0.1,    1}},
        {"dense", {         100,    4,      0.03,   0.1,    1}},
        {"a", {             300,    7,      0.03,   2.0,    1}},
        {"b", {             300,    25,     0.03,   0.5,    1}},
        {"d", {             300,    5,      0.03,   0.1,    1}},
        {"plot1_N40", {     40,     3.1,    0.03,   0.1,    1}},
        {"large", {         2000,   60,     0.03,   0.3,    1}},
        {"Xlarge", {        5000,   60,     0.03,   0.5,    3}},
        {"XlargeR3", {      5000,   60,     0.03,   0.2,    3}},
        {"XXlarge", {       10000,  60,     0.03,   0.1,    1}},
        {"XXlargeR2", {     10000,  60,     0.03,   0.1,    2}},
        {"XXlargeR5", {     10000,  60,     0.03,   0.1,    5}},
        {"XXlargeR5n0", {   10000,  60,     0.03,   0.,     5}},
        {"XXlargeR20", {    10000,  60,     0.03,   0.1,    20}},
        {"XXlargefast", {   10000,  60,     0.1,    0.1,    1}},
        {"XXXlarge", {      20000,  60,     0.03,   0.1,    1}},
        {"Ultralarge", {    200000, 60,     0.03,   0.1,    1}},
        {"Coherence", {     400,    10,     0.03,   1.5,    1}},

        {"BaseNoiseAnalysis40", {       40,     3.1,    0.03,   0.1,    1}},
        {"BaseNoiseAnalysis100", {      100,    5,      0.03,   0.1,    1}},
        {"BaseNoiseAnalysis400", {      400,    10,     0.03,   0.1,    1}},
        {"BaseNoiseAnalysis1600", {     1600,   20,     0.03,   0.1,    1}}

    };

    // Choose between RADIUS, FIXED and FIXEDRADIUS
    Mode mode = Mode::FIXEDRADIUS;

    // Flags
    bool ZDimension = false;     // 2D or 3D
    bool seed = true;            // Whether to use a seed for reproducability

    // Duration of simulation
    int timesteps = 50000;

    // Choose settings
    std::vector<double> chosen_settings = settings["small"];
    int     N       = chosen_settings[0];
    double  L       = chosen_settings[1];
    double  v       = chosen_settings[2];
    double  noise   = chosen_settings[3];
    double  r       = chosen_settings[4];
    // Calculate exchange radius from density; (N / L^2) * r^2
    // Example for N = 5000, L = 60, r = 1; 
    double  k       = (N * r * r) / (L * L);
    int     k_neighbors = 5;


    mode = Mode::RADIUS;

    VicsekModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed);
    model.writeToFile(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise);

    // mode = Mode::FIXED;
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 1.0);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.8);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.5);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.1);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.01);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }

    // mode = Mode::FIXEDRADIUS;
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 1.0);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.8);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.5);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.1);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }
    // {
    //     FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.01);
    //     model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    // }

    mode = Mode::RADIUS;
    {
        FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 1.0);
        model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    }
    {
        FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.8);
        model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    }
    {
        FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.5);
        model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    }
    {
        FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.1);
        model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    }
    {
        FlockingModel model(N, L, v, noise, r, static_cast<Mode>(mode), k_neighbors, ZDimension, seed, 1.0, 0.01);
        model.writeToFileFlocking(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise, "Flocking");
    }


    // Write to file

    Inspector inspector;
    // inspector.runForAllNoiseLevels_Fig2a(true, 150000);
    // inspector.runForAllNoiseLevelsAndModes_Fig2b(true, 20000);
    // inspector.runForAllMultipleInitialConditions_density_weighted(true, 20000, 5);
    // inspector.runForAllNoiseLevels(true);

    return 0;
}