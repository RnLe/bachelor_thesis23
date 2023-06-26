#ifndef INSPECTION_NEIGHBORS_H
#define INSPECTION_NEIGHBORS_H

#include <vector>
#include <map>
#include <string>
#include "VicsekModel.h"

/*
This file is an aggregation of tests to determine the quality and behavior of the Vicsek Model,
limiting the particles to see k neighbors instead of neighbors within a specific radius.

The goal of this file is to produce data for visualization and comparison with the original Vicsek Model.
The converging behavior is then used to make suitable predictions and adjustments to the neural network.
*/

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



#endif // INSPECTION_NEIGHBORS_H