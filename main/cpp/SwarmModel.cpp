#include "helperFunctions.h"
#include "Particle.h"
#include "SwarmModel.h"

#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <algorithm>    // for sort() and find_if()
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <omp.h>

    // Constructors
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    SwarmModel::SwarmModel(int N, double L, double v, double noise, double r, Mode mode, int k_neighbors, bool ZDimension, bool seed)
        : N(N), L(L), v(v), noise(noise), r(r), mode(mode), k_neighbors(k_neighbors), density3D(N / (L * L * L)), density2D(N / (L * L)), num_cells(int(ceil(L / (2 * r)))),
        seed(seed), ZDimension(ZDimension) {

        std::uniform_real_distribution<> dis_x(0, L);
        std::uniform_real_distribution<> dis_y(0, L);
        std::uniform_real_distribution<> dis_z(0, ZDimension ? L : 0);
        std::uniform_real_distribution<> angle_dis(0, 2 * M_PI);
        std::uniform_real_distribution<> polarAngle_dis(0, ZDimension ? M_PI : 0);

        // Make a case differentiation here once, instead of a differentiation later each time a random number is used.
        // Use cases: Noise
        // Random noise calculation takes up most of the computing time. Faster generator is introduced.
        if (seed) {
            gen1 = std::mt19937(seed1);
            gen2 = std::mt19937(seed2);
            gen3 = std::mt19937(seed3);

            fastgen1 = std::minstd_rand(seed1);
            fastgen2 = std::minstd_rand(seed2);
            fastgen3 = std::minstd_rand(seed3);
        }
        else {
            gen1 = std::mt19937(rd());
            gen2 = std::mt19937(rd());
            gen3 = std::mt19937(rd());

            fastgen1 = std::minstd_rand(rd());
            fastgen2 = std::minstd_rand(rd());
            fastgen3 = std::minstd_rand(rd());
        }

        for (int i = 0; i < N; ++i) {
            particles.push_back(Particle(dis_x(gen1), dis_y(gen2), dis_z(gen3), angle_dis(gen2), polarAngle_dis(gen1)));
        }

        // Initialize cells
        // A 3-Dimensional vector-array, containing a vector for storing the ID of the particles
        // Used for neighbor lists, effectively just evaluating the size of the cells
        cells3D = std::vector<std::vector<std::vector<std::vector<int>>>>(num_cells, std::vector<std::vector<std::vector<int>>>(num_cells, std::vector<std::vector<int>>(num_cells, std::vector<int>())));
        cells2D = std::vector<std::vector<std::vector<int>>>(num_cells, std::vector<std::vector<int>>(num_cells, std::vector<int>()));

        // Initialize mode1_cells
        for (int i = -cellSpan; i <= cellSpan; ++i) {
            mode1_cells.push_back(i);
        }

        k_reserve = int(mode == Mode::FIXED ? 2 * k_neighbors : 2 * (N * r * r) / (L * L));
    }
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    // Methods
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    void SwarmModel::update_cells() {
        /*
        This method iterates through all particles and checks in what cell it is. The corresponding is updated with the ID of the particle.
        The .size() of a cell can be used to determine how many particles it contains.
        */
        // Reset cells
        cells3D = std::vector<std::vector<std::vector<std::vector<int>>>>(num_cells, std::vector<std::vector<std::vector<int>>>(num_cells, std::vector<std::vector<int>>(num_cells, std::vector<int>())));
        cells2D = std::vector<std::vector<std::vector<int>>>(num_cells, std::vector<std::vector<int>>(num_cells, std::vector<int>()));

        if (ZDimension) {
            for (int i = 0; i < particles.size(); ++i) {
                int cell_x = int(particles[i].x / (2*r)) % num_cells;
                int cell_y = int(particles[i].y / (2*r)) % num_cells;
                int cell_z = int(particles[i].z / (2*r)) % num_cells;
                cells3D[cell_x][cell_y][cell_z].push_back(i);
            }
        } else {
            for (int i = 0; i < particles.size(); ++i) {
                int cell_x = int(particles[i].x / (2*r)) % num_cells;
                int cell_y = int(particles[i].y / (2*r)) % num_cells;
                cells2D[cell_x][cell_y].push_back(i);
            }
        }
    }

    // TODO Implement 2D hist
    std::vector<double> SwarmModel::get_density_hist3D() {
        std::vector<double> densities(num_cells * num_cells * num_cells, 0.0);
        int i = 0;
        for (auto& cell_row : cells3D) {
            for (auto& cell_column : cell_row) {
                for (auto& cell : cell_column) {
                    densities[i] = static_cast<double>(cell.size()) / N;
                    ++i;
                }
            }
        }
        return densities;
    }

    std::vector<double> SwarmModel::get_dynamic_radius() {
        std::vector<double> effective_radii(N, 0.0);
        for (int i = 0; i < particles.size(); ++i) {
            effective_radii[i] = particles[i].cellRange;
        }
        return effective_radii;
    }

    std::pair<std::vector<Particle*>, std::vector<double>> SwarmModel::get_neighbors(Particle& particle, int index) {
        // Case differentiation between radius and fixed number of neighbors
        // If fixed numbers is chosen, effectively let the while loop iterate over every cells (and abort if it has found k neighbors)
        int rangeOfCells = num_cells;
        if (mode == Mode::RADIUS || mode == Mode::FIXEDRADIUS) {
            // Cells are cubes with the side length of a radius. So we only want to iterate over a maximum of 2 cell shell around the radius.
            // Consider that the cell size increases with the radius
            rangeOfCells = 2;
        }

        int cell_x = int(particle.x / (2*r)) % num_cells;
        int cell_y = int(particle.y / (2*r)) % num_cells;
        int cell_z = int(particle.z / (2*r)) % num_cells;
        std::vector<Particle*> neighbors;
        std::vector<double> distances;
        // Allocate space for the maximum number of neighbors (case differentiation between modes)
        // If mode==FIXED, reserve 2k neighbors
        // If mode==RADIUS or mode==FIXEDRADIUS, reserve k = 2 * (N * r * r) / (L * L) (Twice the average number of neighbors)
        neighbors.reserve(k_reserve);
        distances.reserve(k_reserve);
        std::vector<int> mode1_cells(cellSpan * 2 + 1);
        std::iota(mode1_cells.begin(), mode1_cells.end(), -cellSpan);
        int boundary = 0;
        bool breakFlag = false;

        if (ZDimension) {
            while (boundary < rangeOfCells) {
                for (int dx = -boundary; dx <= boundary; ++dx) {
                    for (int dy = -boundary; dy <= boundary; ++dy) {
                        for (int dz = -boundary; dz <= boundary; ++dz) {
                            if (std::abs(dx) != boundary && std::abs(dy) != boundary && std::abs(dz) != boundary) continue;
                            int neighbor_cell_x = (cell_x + dx + num_cells);
                            bool neighbor_cell_x_shift = (neighbor_cell_x == neighbor_cell_x % num_cells);
                            neighbor_cell_x %= num_cells;
                            int neighbor_cell_y = (cell_y + dy + num_cells);
                            bool neighbor_cell_y_shift = (neighbor_cell_y == neighbor_cell_y % num_cells);
                            neighbor_cell_y %= num_cells;
                            int neighbor_cell_z = (cell_z + dz + num_cells);
                            bool neighbor_cell_z_shift = (neighbor_cell_z == neighbor_cell_z % num_cells);
                            neighbor_cell_z %= num_cells;

                            for (int j : cells3D[neighbor_cell_x][neighbor_cell_y][neighbor_cell_z]) {
                                if (index != j) {
                                    float px = neighbor_cell_x_shift ? (dx < 0 ? particles[j].x - L : particles[j].x + L) : particles[j].x;
                                    float py = neighbor_cell_y_shift ? (dy < 0 ? particles[j].y - L : particles[j].y + L) : particles[j].y;
                                    float pz = neighbor_cell_z_shift ? (dz < 0 ? particles[j].z - L : particles[j].z + L) : particles[j].z;
                                    double distance = std::pow((particle.x - px) - L * std::round((particle.x - px) / L), 2) +
                                    std::pow((particle.y - py) - L * std::round((particle.y - py) / L), 2) +
                                    std::pow((particle.z - pz) - L * std::round((particle.z - pz) / L), 2);
                                    // If mode==RADIUS or mode==FIXEDRADIUS and distance smaller that radius OR if mode==FIXED (count all neighbors in last layer and sort by distance later)
                                    if (((mode == Mode::RADIUS || mode == Mode::FIXEDRADIUS) && distance < (r * r)) || mode == Mode::FIXED) {
                                        neighbors.push_back(&particles[j]);
                                        distances.push_back(distance);
                                    }
                                }
                            }
                        }
                    }
                }
                boundary++;
                // Use a break flag here to let the loop iterate one more time.
                // A particle at a boundary of a cell needs to check the neighboring cells, regardless of how many particles it has found in its own cell
                if (breakFlag == true) break;
                if (mode == Mode::FIXED && neighbors.size() > k_neighbors) breakFlag = true;
            }
        }
        else {
            while (boundary < rangeOfCells) {
                for (int dx = -boundary; dx <= boundary; ++dx) {
                    for (int dy = -boundary; dy <= boundary; ++dy) {
                        if (std::abs(dx) != boundary && std::abs(dy) != boundary) continue;
                        // For the correct distance measurement it's important to detect a cell shift
                        int neighbor_cell_x = (cell_x + dx + num_cells);
                        bool neighbor_cell_x_shift = (neighbor_cell_x == neighbor_cell_x % num_cells);
                        neighbor_cell_x %= num_cells;
                        int neighbor_cell_y = (cell_y + dy + num_cells);
                        bool neighbor_cell_y_shift = (neighbor_cell_y == neighbor_cell_y % num_cells);
                        neighbor_cell_y %= num_cells;

                        for (int j : cells2D[neighbor_cell_x][neighbor_cell_y]) {
                            if (index != j) {
                                // Check whether a cell shift was made or not
                                float px = neighbor_cell_x_shift ? (dx < 0 ? particles[j].x - L : particles[j].x + L) : particles[j].x;
                                float py = neighbor_cell_y_shift ? (dy < 0 ? particles[j].y - L : particles[j].y + L) : particles[j].y;
                                double distance = std::pow((particle.x - px) - L * std::round((particle.x - px) / L), 2) +
                                std::pow((particle.y - py) - L * std::round((particle.y - py) / L), 2);
                                // If mode==RADIUS or mode==RADIUSFIXED and distance smaller that radius OR if mode==FIXED (count all neighbors in last layer and sort by distance later)
                                if (((mode == Mode::RADIUS || mode == Mode::FIXEDRADIUS) && distance < (r * r)) || mode == Mode::FIXED) {
                                    neighbors.push_back(&particles[j]);
                                    distances.push_back(distance);
                                }
                            }
                        }
                    }
                }
                boundary++;
                // Use a break flag here to let the loop iterate one more time.
                // A particle at a boundary of a cell needs to check the neighboring cells, regardless of how many particles it has found in its own cell
                if (breakFlag == true) break;
                if (mode == Mode::FIXED && neighbors.size() > k_neighbors) breakFlag = true;
            }
        }

        particle.cellRange = boundary - 1;

        neighbors.push_back(&particle);
        distances.push_back(0);

        if (neighbors.size() > 1) {
            std::vector<std::pair<Particle*, double>> pairs;
            for (int i = 0; i < neighbors.size(); ++i) pairs.push_back(std::make_pair(neighbors[i], distances[i]));
            // std::sort(pairs.begin(), pairs.end(), [](auto& left, auto& right) { return left.second < right.second; });
            // Rewrite this for use with std=c++11
            std::sort(pairs.begin(), pairs.end(), [](std::pair<Particle*, double> left, std::pair<Particle*, double> right) { return left.second < right.second; });

            neighbors.clear();
            distances.clear();
            for (auto& pair : pairs) {
                neighbors.push_back(pair.first);
                distances.push_back(pair.second);
            }
            switch (mode) {
                case Mode::FIXED: {
                    neighbors.resize(k_neighbors + 1);
                    break;
                }
                case Mode::RADIUS: {
                    auto cut_off = std::find_if(distances.begin(), distances.end(), [](double value) { return value >= 1; });
                    if (cut_off != distances.end()) neighbors.resize(std::distance(distances.begin(), cut_off));
                    break;
                }
                case Mode::FIXEDRADIUS: {
                    // Shuffle only if the number of neighbors is larger than k_neighbors
                    if (neighbors.size() > k_neighbors + 1) {
                        std::vector<Particle*> neighbors_copy(k_neighbors + 1);
                        std::vector<double> distances_copy(k_neighbors + 1 );
                        // Add self to neighbors and distances
                        neighbors_copy[0] = neighbors[0];
                        distances_copy[0] = distances[0];
                        // Remove self from neighbors and distances
                        neighbors.erase(neighbors.begin());
                        distances.erase(distances.begin());
                        // Shuffle neighbors and distances in the same manner
                        std::vector<int> indices(neighbors.size());
                        // Fill indices with 0, 1, 2, ..., neighbors.size() - 1
                        std::iota(indices.begin(), indices.end(), 0);
                        std::shuffle(indices.begin(), indices.end(), gen3);
                        // Consequently, the first element of indices is 1
                        for (int i = 1; i < k_neighbors + 1; ++i) {
                            neighbors_copy[i] = neighbors[indices[i]];
                            distances_copy[i] = distances[indices[i]];
                        }
                        neighbors = neighbors_copy;
                        distances = distances_copy;

                        neighbors.resize(k_neighbors + 1);
                        distances.resize(k_neighbors + 1);
                    }
 
                    break;
                }
            }

        }

        return std::make_pair(neighbors, distances);
    }

    double SwarmModel::mean_direction2D() {
        double cos_sum = 0.0;
        double sin_sum = 0.0;
        for (Particle& p : particles) {
            cos_sum += std::cos(p.angle);
            sin_sum += std::sin(p.angle);
        }
        return std::hypot(cos_sum / particles.size(), sin_sum / particles.size());
    }

    std::pair<double, double> SwarmModel::mean_direction3D() {
        double cos_sum_azimuth = 0.0;
        double sin_sum_azimuth = 0.0;
        double cos_sum_polar = 0.0;
        double sin_sum_polar = 0.0;
        for (Particle& p : particles) {
            cos_sum_azimuth += std::cos(p.angle);
            sin_sum_azimuth += std::sin(p.angle);
            cos_sum_polar += std::cos(p.polarAngle);
            sin_sum_polar += std::sin(p.polarAngle);
        }
        double mean_azimuth = std::atan2(sin_sum_azimuth / particles.size(), cos_sum_azimuth / particles.size());
        double mean_polar = std::atan2(sin_sum_polar / particles.size(), cos_sum_polar / particles.size());

        // normalize to [0, 2pi] and [0, pi]
        if (mean_azimuth < 0) mean_azimuth += 2 * M_PI;
        if (mean_polar < 0) mean_polar += M_PI;

        return {mean_azimuth, mean_polar};
    }

    // Density weighted order parameter
    // \rho_r=\sum_{i<k}^N \vec{v}_i \vec{v}_k \frac{1}{\left|\vec{r}_i-\vec{r}_k\right|}=\sum_{k=0}^N \sum_{i=k+1}^N \vec{v}_i \vec{v}_k \frac{1}{\left|\vec{r}_i-\overrightarrow{r_k}\right|}
    // =\sum_{k=0}^N \sum_{i=k+1}^N v_0^2 \cos \left(\varphi_i-\varphi_k\right) d_{i k}^{-1}
    // Implementing the last equation
    double SwarmModel::density_weighted_op() {
        double order_parameter = 0.0;
        double diff_sum = 0.0;
        double temp_diff = 0.0;
        for (int k = 0; k < particles.size(); ++k) {
            for (int i = k + 1; i < particles.size(); ++i) {
                temp_diff = std::hypot(particles[i].x - particles[k].x, particles[i].y - particles[k].y);
                order_parameter += this->v * this->v * std::cos(particles[i].angle - particles[k].angle) / temp_diff;
                diff_sum += 1 / temp_diff;
            }
        }
        return order_parameter;
    }

    // This method is 2D for now
    std::pair<int, std::pair<double, double>> SwarmModel::density_weighted_op_watcher(int timeLimit, double tolerance) {
        // Hyperparameters

        // Maximum number of steps to equilibrate
        int maxEquilibrationSteps = timeLimit * 0.5;
        // Minimum number of steps to average over after equilibration
        int numberStepsAveraging = 40000;
        // Relative tolerance for standard deviation
        double relativeTolerance = 0.01;    // 1%
        // Number of steps to average over for std and mean
        int numberStepsStdMean = 1000;

        // Variables
        int timeStep = 0;
        double mean = 0.01;
        double std = 0.01;
        double last_std = 0.01;
        double relative_diff = 0.01;
        std::vector<double> mean_directions(timeLimit);
        // This is a lot of memory, but it's the fastest way to do it. The memory is freed after the method is finished.

        // Count the number of consecutive times the relative difference is smaller than the tolerance
        int consecutive = 0;
        int consecutive_limit = 10;

        // In this loop, the model is equilibrated
        // The model is considered equilibrated if the standard deviation of the last numberStepsStdMean steps is relatively the same
        // compared to the standard deviation of the last numberStepsStdMean steps before that.

        while (timeStep < maxEquilibrationSteps) {
            // Update the model
            update();
            mean_directions[timeStep] = density_weighted_op();

            // Calculate everything only every numberStepsStdMean steps
            if (timeStep % numberStepsStdMean == 0 and timeStep > 0) {
                // Calculate the new mean
                // Consider to calculate the mean only for the last numberStepsStdMean steps
                mean = std::accumulate(mean_directions.begin() + timeStep - numberStepsStdMean, mean_directions.begin() + mean_directions.size(), 0.0) / numberStepsStdMean;
                // Calculate the new standard deviation
                std = 0.0;
                for (int i = 0; i < numberStepsStdMean; ++i) {
                    std += std::pow(mean_directions[i] - mean, 2);
                }
                std /= numberStepsStdMean;
                std = std::sqrt(std);

                // Check if the standard deviation is relatively the same
                if (timeStep > numberStepsStdMean) {
                    relative_diff = std::abs(std / last_std - 1);
                    if (relative_diff < relativeTolerance) {
                        last_std = std;
                        ++consecutive;
                        if (consecutive == consecutive_limit) {
                            break;
                        }
                    }
                    else {
                        consecutive = 0;
                    }
                }
                last_std = std;
            // Print the progress
            std::cout << std::fixed << std::setprecision(3) << "\033[1;32mEquilibrating: " << N << " particles at t = " << timeStep << ", rho = " << mean_directions[timeStep] << ", latest std: " << 
            last_std << ", mean: " << mean << ", rel diff: " << relative_diff << "                               \033[0m\r";
            std::cout.flush();
            }

            
            ++timeStep;
        }

        int equilibrationSteps = timeStep;
        numberStepsAveraging = std::max(numberStepsAveraging, equilibrationSteps);

        // After equilibration, the model is averaged over numberStepsAveraging steps
        for (int i = 0; i < numberStepsAveraging; ++i) {
            update();
            mean_directions[++timeStep] = mean_direction2D();
            // Print the progress every 1000 steps
            if (timeStep % 1000 == 0) {
                std::cout << std::fixed << std::setprecision(3) << "\033[1;32mAveraging: " << N << " particles at t = " << timeStep << ", va = " << mean_directions[timeStep] << "                                                                                \033[0m\r";
                std::cout.flush();
            }
        }

        // Calculate the mean for the last numberStepsAveraging steps
        mean = std::accumulate(mean_directions.begin() + timeStep - numberStepsAveraging, mean_directions.begin() + mean_directions.size(), 0.0) / numberStepsAveraging;
        // Print number of averaging steps
        // std::cout << "Number of averaging steps: " << numberStepsAveraging << "                                                                                                        \n";

        // Calculate the standard deviation for the last numberStepsAveraging steps
        double sum_sq = 0.0;
        for (int i = timeStep - numberStepsAveraging; i < timeStep; ++i) {
            sum_sq += std::pow(mean_directions[i] - mean, 2);
        }
        sum_sq = std::sqrt(sum_sq / numberStepsAveraging);

        return std::make_pair(timeStep, std::make_pair(mean, sum_sq));
    }

    // This method is 2D for now
    std::pair<int, std::pair<double, double>> SwarmModel::mean_direction_watcher(int timeLimit, double tolerance) {
        // Hyperparameters

        // Maximum number of steps to equilibrate
        int maxEquilibrationSteps = timeLimit * 0.5;
        // Minimum number of steps to average over after equilibration
        int numberStepsAveraging = 40000;
        // Relative tolerance for standard deviation
        double relativeTolerance = 0.01;    // 1%
        // Number of steps to average over for std and mean
        int numberStepsStdMean = 1000;

        // Variables
        int timeStep = 0;
        double mean = 0.01;
        double std = 0.01;
        double last_std = 0.01;
        double relative_diff = 0.01;
        std::vector<double> mean_directions(timeLimit);
        // This is a lot of memory, but it's the fastest way to do it. The memory is freed after the method is finished.

        // Count the number of consecutive times the relative difference is smaller than the tolerance
        int consecutive = 0;
        int consecutive_limit = 10;

        // In this loop, the model is equilibrated
        // The model is considered equilibrated if the standard deviation of the last numberStepsStdMean steps is relatively the same
        // compared to the standard deviation of the last numberStepsStdMean steps before that.

        while (timeStep < maxEquilibrationSteps) {
            // Update the model
            update();
            mean_directions[timeStep] = mean_direction2D();

            // Calculate everything only every numberStepsStdMean steps
            if (timeStep % numberStepsStdMean == 0 and timeStep > 0) {
                // Calculate the new mean
                // Consider to calculate the mean only for the last numberStepsStdMean steps
                mean = std::accumulate(mean_directions.begin() + timeStep - numberStepsStdMean, mean_directions.begin() + mean_directions.size(), 0.0) / numberStepsStdMean;
                // Calculate the new standard deviation
                std = 0.0;
                for (int i = 0; i < numberStepsStdMean; ++i) {
                    std += std::pow(mean_directions[i] - mean, 2);
                }
                std /= numberStepsStdMean;
                std = std::sqrt(std);

                // Check if the standard deviation is relatively the same
                if (timeStep > numberStepsStdMean) {
                    relative_diff = std::abs(std / last_std - 1);
                    if (relative_diff < relativeTolerance) {
                        last_std = std;
                        ++consecutive;
                        if (consecutive == consecutive_limit) {
                            break;
                        }
                    }
                    else {
                        consecutive = 0;
                    }
                }
                last_std = std;
            // Print the progress
            std::cout << std::fixed << std::setprecision(3) << "\033[1;32mEquilibrating: " << N << " particles at t = " << timeStep << ", va = " << mean_directions[timeStep] << ", latest std: " << 
            last_std << ", mean: " << mean << ", rel diff: " << relative_diff << "                               \033[0m\r";
            std::cout.flush();
            }

            
            ++timeStep;
        }

        int equilibrationSteps = timeStep;
        numberStepsAveraging = std::max(numberStepsAveraging, equilibrationSteps);

        // After equilibration, the model is averaged over numberStepsAveraging steps
        for (int i = 0; i < numberStepsAveraging; ++i) {
            update();
            mean_directions[++timeStep] = mean_direction2D();
            // Print the progress every 1000 steps
            if (timeStep % 1000 == 0) {
                std::cout << std::fixed << std::setprecision(3) << "\033[1;32mAveraging: " << N << " particles at t = " << timeStep << ", va = " << mean_directions[timeStep] << "                                                                                \033[0m\r";
                std::cout.flush();
            }
        }

        // Calculate the mean for the last numberStepsAveraging steps
        mean = std::accumulate(mean_directions.begin() + timeStep - numberStepsAveraging, mean_directions.begin() + mean_directions.size(), 0.0) / numberStepsAveraging;
        // Print number of averaging steps
        // std::cout << "Number of averaging steps: " << numberStepsAveraging << "                                                                                                        \n";

        // Calculate the standard deviation for the last numberStepsAveraging steps
        double sum_sq = 0.0;
        for (int i = timeStep - numberStepsAveraging; i < timeStep; ++i) {
            sum_sq += std::pow(mean_directions[i] - mean, 2);
        }
        sum_sq = std::sqrt(sum_sq / numberStepsAveraging);

        return std::make_pair(timeStep, std::make_pair(mean, sum_sq));
    }



    void SwarmModel::writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, Mode mode, int k, double noise, std::string model) {
        if (filetype == "xyz") {
            std::string modus;
            if (mode == Mode::FIXED) {
                modus = "kNeighbors";
            }
            else if (mode == Mode::RADIUS) {
                modus = "rRadius";
            }
            else if (mode == Mode::QUANTILE) {
                modus = "kQuantiles";
            }
            else if (mode == Mode::FIXEDRADIUS) {
                modus = "kFixedRadius";
            }
            std::string base = "../../data/particles_";
            std::string radiusOrK;
            switch (mode)
            {
            case Mode::FIXED:
                radiusOrK = "_k" + std::to_string(k);
                break;
            case Mode::RADIUS:
                radiusOrK = "_r" + helperFunctions::format_float(r);
                break;
            case Mode::QUANTILE:
                radiusOrK = "_k" + std::to_string(k);
                break;
            case Mode::FIXEDRADIUS:
                radiusOrK = "_r" + helperFunctions::format_float(r) + "_k" + std::to_string(k);
                break;
            default:
                radiusOrK = "_k" + std::to_string(k);
                break;
            }
            std::string parameters = "t" + std::to_string(timesteps) + "_N" + std::to_string(N) + "_L" + helperFunctions::format_float(L) + "_v" + helperFunctions::format_float(v) + "_n" + helperFunctions::format_float(noise)
            + radiusOrK + "_mode_" + modus + "_model_" + model+ "_" + (ZDimension ? "3D" : "2D");
            std::string filename = base + parameters + ".xyz";
            std::ofstream file(filename);
            for (int i = 0; i < timesteps; ++i) {
                file << particles.size() << "\n\n";
                for (Particle& particle : particles) {
                    file << particle.x << " " << particle.y << " " << particle.z << "\n";
                }
                // Print progress
                std::cout << "\033[1;32mProgress: " << std::fixed << std::setprecision(2) << (double)i / timesteps * 100 << "%\033[0m\r";
                std::cout.flush();
                update();
            }
            // Clear last line in console
            std::cout << "\033[2K";
            // Print 100% progress
            std::cout << "\033[1;32mProgress: 100%\033[0m\n";
            file.close();
        }
    }
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
