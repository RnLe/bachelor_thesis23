#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <algorithm>    // for sort() and find_if()
#include <iostream>
#include <iomanip>
#include <fstream>
#include "Particle.h"
#include "SwarmModel.h"
#include <sstream>
#include <omp.h>

    // Constructor
    SwarmModel::SwarmModel(int N, double L, double v, double noise, double r, Mode mode, int k_neighbors, bool ZDimension, bool seed)
        : N(N), L(L), v(v), noise(noise), r(r), mode(mode), k_neighbors(k_neighbors), density3D(N / (L * L * L)), density2D(N / (L * L)), num_cells(int(L / (2 * r))),
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
    }

    // Methods
    void SwarmModel::update_cells() {
        // Reset cells
        cells3D = std::vector<std::vector<std::vector<std::vector<int>>>>(num_cells, std::vector<std::vector<std::vector<int>>>(num_cells, std::vector<std::vector<int>>(num_cells, std::vector<int>())));
        cells2D = std::vector<std::vector<std::vector<int>>>(num_cells, std::vector<std::vector<int>>(num_cells, std::vector<int>()));

        if (ZDimension) {
            for (int i = 0; i < particles.size(); ++i) {
                int cell_x = int(particles[i].x / r) % num_cells;
                int cell_y = int(particles[i].y / r) % num_cells;
                int cell_z = int(particles[i].z / r) % num_cells;
                cells3D[cell_x][cell_y][cell_z].push_back(i);
            }
        } else {
            for (int i = 0; i < particles.size(); ++i) {
                int cell_x = int(particles[i].x / r) % num_cells;
                int cell_y = int(particles[i].y / r) % num_cells;
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
        if (mode == Mode(RADIUS)) {
            // Cells are cubes with the side length of a radius. So we only want to iterate over a maximum of 1 cell shell around the radius.
            // OPTIMIZATION: Detect whether r is close to a whole number. Iterating over the next shell of neighbor cells is very computing intensive.
            double tolerance = 0.1;
            double fractional_part = r - int(r);
            if (fractional_part < tolerance) {
                rangeOfCells = int(r) + 1;
            } else {
                rangeOfCells = int(r) + 2;
            }
        }

        int cell_x = int(particle.x / r) % num_cells;;
        int cell_y = int(particle.y / r) % num_cells;;
        int cell_z = int(particle.z / r) % num_cells;;
        std::vector<Particle*> neighbors;
        std::vector<double> distances;
        std::vector<int> mode1_cells(cellSpan * 2 + 1);
        std::iota(mode1_cells.begin(), mode1_cells.end(), -cellSpan);
        int boundary = 0;

        if (ZDimension) {
            while (boundary < rangeOfCells) {
                for (int dx = -boundary; dx <= boundary; ++dx) {
                    for (int dy = -boundary; dy <= boundary; ++dy) {
                        for (int dz = -boundary; dz <= boundary; ++dz) {
                            if (std::abs(dx) != boundary && std::abs(dy) != boundary && std::abs(dz) != boundary) continue;
                            int neighbor_cell_x = (cell_x + dx + num_cells) % num_cells;
                            int neighbor_cell_y = (cell_y + dy + num_cells) % num_cells;
                            int neighbor_cell_z = (cell_y + dz + num_cells) % num_cells;

                            for (int j : cells3D[neighbor_cell_x][neighbor_cell_y][neighbor_cell_z]) {
                                if (index != j) {
                                    double distance = std::pow((particle.x - particles[j].x) - L * std::round((particle.x - particles[j].x) / L), 2) +
                                    std::pow((particle.y - particles[j].y) - L * std::round((particle.y - particles[j].y) / L), 2) +
                                    std::pow((particle.z - particles[j].z) - L * std::round((particle.z - particles[j].z) / L), 2);
                                    neighbors.push_back(&particles[j]);
                                    distances.push_back(distance);
                                }
                            }
                        }
                    }
                }
                boundary++;
                if (mode == Mode(FIXED) && neighbors.size() > k_neighbors) break;
            }
        } else {
            while (boundary < rangeOfCells) {
                for (int dx = -boundary; dx <= boundary; ++dx) {
                    for (int dy = -boundary; dy <= boundary; ++dy) {
                        if (std::abs(dx) != boundary && std::abs(dy) != boundary) continue;
                        int neighbor_cell_x = (cell_x + dx + num_cells) % num_cells;
                        int neighbor_cell_y = (cell_y + dy + num_cells) % num_cells;
                        
                        for (int j : cells2D[neighbor_cell_x][neighbor_cell_y]) {
                            if (index != j) {
                                double distance = std::pow((particle.x - particles[j].x) - L * std::round((particle.x - particles[j].x) / L), 2) +
                                std::pow((particle.y - particles[j].y) - L * std::round((particle.y - particles[j].y) / L), 2);
                                // If mode==radius and distance smaller that radius OR if mode==fixed (count all neighbors in last layer and sort by distance later)
                                if ((mode == 0 && distance < (r * r)) || mode == 1) {
                                    neighbors.push_back(&particles[j]);
                                    distances.push_back(distance);
                                }       
                            }
                        }
                    }
                }
                boundary++;
                if (mode == Mode(FIXED) && neighbors.size() > k_neighbors) break;
            }
        }
        
        

        // // Parallelization might lead to more neighbors in the list than desired.
        // // Additional check to truncate the neighors.
        // if(neighbors.size() > k_neighbors) {
        //     // Get the indices of the k_neighbors smallest elements in 'distances'
        //     std::vector<size_t> indices(distances.size());
        //     std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, ..., distances.size() - 1
        //     std::partial_sort(indices.begin(), indices.begin() + k_neighbors, indices.end(),
        //                     [&distances](size_t i1, size_t i2) { return distances[i1] < distances[i2]; });
        //     indices.resize(k_neighbors);  // Only keep the first k_neighbors indices

        //     // Construct new 'neighbors' and 'distances' vectors with only the closest neighbors
        //     std::vector<Particle*> new_neighbors(k_neighbors);
        //     std::vector<double> new_distances(k_neighbors);
        //     for(size_t i = 0; i < k_neighbors; ++i) {
        //         new_neighbors[i] = neighbors[indices[i]];
        //         new_distances[i] = distances[indices[i]];
        //     }

        //     // Replace 'neighbors' and 'distances' with their new versions
        //     neighbors = std::move(new_neighbors);
        //     distances = std::move(new_distances);
        // }

        particle.cellRange = boundary - 1;

        neighbors.push_back(&particle);
        distances.push_back(0);

        if (neighbors.size() > 1) {
            std::vector<std::pair<Particle*, double>> pairs;
            for (int i = 0; i < neighbors.size(); ++i) pairs.push_back(std::make_pair(neighbors[i], distances[i]));
            std::sort(pairs.begin(), pairs.end(), [](auto& left, auto& right) { return left.second < right.second; });

            neighbors.clear();
            distances.clear();
            for (auto& pair : pairs) {
                neighbors.push_back(pair.first);
                distances.push_back(pair.second);
            }

            if (mode == 1) {
                neighbors.resize(k_neighbors + 1);
            } else if (mode == 0) {
                auto cut_off = std::find_if(distances.begin(), distances.end(), [](double value) { return value >= 1; });
                if (cut_off != distances.end()) neighbors.resize(std::distance(distances.begin(), cut_off));
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


    void SwarmModel::writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, SwarmModel::Mode mode, int k, double noise, std::string model) {
        if (filetype == "xyz") {
            std::string base = "../../data/particles_";
            std::string radiusOrK = mode == SwarmModel::Mode::FIXED ? "_k" + format_float(k) : "_r" + format_float(r);
            std::string parameters = "t" + std::to_string(timesteps) + "_N" + std::to_string(N) + "_L" + format_float(L) + "_v" + format_float(v) + "_n" + format_float(noise)
            + radiusOrK + "_mode_" + (mode == SwarmModel::Mode::FIXED ? "fixed" : "radius") + "_model_" + model+ "_" + (ZDimension ? "3D" : "2D");
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
            std::cout << std::endl;
            file.close();
        }
    }



    std::string SwarmModel::format_float(float number) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(std::numeric_limits<float>::digits10);

        out << number;

        std::string str = out.str();
        size_t end = str.find_last_not_of('0') + 1;

        if (str[end - 1] == '.') {
            end--;
        }

        str.erase(end, std::string::npos);
        return str;
    }
