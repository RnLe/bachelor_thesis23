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

    // Member variables
    int N, mode, k_neighbors, num_cells, cellSpan = 0;
    double L, v, noise, r, density;
    std::vector<Particle> particles;
    std::vector<std::vector<std::vector<int>>> cells;
    std::vector<int> mode1_cells;
    enum Mode { RADIUS, FIXED };

    // Constructor
    SwarmModel::SwarmModel(int N, double L, double v, double noise, double r, Mode mode, int k_neighbors)
        : N(N), L(L), v(v), noise(noise), r(r), mode(mode), k_neighbors(k_neighbors), density(N / (L * L)), num_cells(int(L / r)) {
        // Initialize particles
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, L);
        std::uniform_real_distribution<> angle_dis(0, 2 * M_PI);
        for (int i = 0; i < N; ++i) {
            particles.push_back(Particle(dis(gen), dis(gen), angle_dis(gen)));
        }

        // Initialize cells
        cells = std::vector<std::vector<std::vector<int>>>(num_cells, std::vector<std::vector<int>>(num_cells, std::vector<int>()));

        // Initialize mode1_cells
        for (int i = -cellSpan; i <= cellSpan; ++i) {
            mode1_cells.push_back(i);
        }
    }

    // Methods
    void SwarmModel::update_cells() {
        // Reset cells
        cells = std::vector<std::vector<std::vector<int>>>(num_cells, std::vector<std::vector<int>>(num_cells, std::vector<int>()));

        for (int i = 0; i < particles.size(); ++i) {
            int cell_x = int(particles[i].x / r) % num_cells;
            int cell_y = int(particles[i].y / r) % num_cells;
            cells[cell_x][cell_y].push_back(i);
        }
    }

    std::vector<double> SwarmModel::get_density_hist() {
        std::vector<double> densities(num_cells * num_cells, 0.0);
        int i = 0;
        for (auto& cell_row : cells) {
            for (auto& cell : cell_row) {
                densities[i] = static_cast<double>(cell.size()) / N;
                ++i;
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
        int cell_x = int(particle.x / r) % num_cells;;
        int cell_y = int(particle.y / r) % num_cells;;
        std::vector<Particle*> neighbors;
        std::vector<double> distances;
        std::vector<int> mode1_cells(cellSpan * 2 + 1);
        std::iota(mode1_cells.begin(), mode1_cells.end(), -cellSpan);
        int boundary = 0;

        while (neighbors.size() < k_neighbors) {
            for (int dx = -boundary; dx <= boundary; ++dx) {
                for (int dy = -boundary; dy <= boundary; ++dy) {
                    if (std::abs(dx) != boundary && std::abs(dy) != boundary) continue;
                    int neighbor_cell_x = (cell_x + dx + num_cells) % num_cells;
                    int neighbor_cell_y = (cell_y + dy + num_cells) % num_cells;

                    for (int j : cells[neighbor_cell_x][neighbor_cell_y]) {
                        if (index != j) {
                            double distance = std::pow((particle.x - particles[j].x) - L * std::round((particle.x - particles[j].x) / L), 2) +
                                            std::pow((particle.y - particles[j].y) - L * std::round((particle.y - particles[j].y) / L), 2);
                            neighbors.push_back(&particles[j]);
                            distances.push_back(distance);
                        }
                    }
                }
            }
            if (mode == Mode(RADIUS) && boundary == 1) break;
            boundary += 1;
        }

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

    double SwarmModel::va() {
        double cos_sum = 0.0;
        double sin_sum = 0.0;
        for (Particle& p : particles) {
            cos_sum += std::cos(p.angle);
            sin_sum += std::sin(p.angle);
        }
        return std::hypot(cos_sum / particles.size(), sin_sum / particles.size());
    }

    void SwarmModel::writeToFile(int timesteps, std::string filetype, int N, double L, double v, double r, SwarmModel::Mode mode) {
        if (filetype == "xyz") {
            std::string base = "../../data/particles_";
            std::string parameters = "N" + std::to_string(N) + "_L" + format_float(L) + "_v" + format_float(v) + "_r" + format_float(r) + "_mode:" + std::to_string(static_cast<int>(mode));
            std::string filename = base + parameters + ".xyz";
            std::ofstream file(filename);
            for (int i = 0; i < timesteps; ++i) {
                update();
                file << particles.size() << "\n\n";
                for (Particle& particle : particles) {
                    file << particle.x << " " << particle.y << "\n";
                }
                // Print progress
                std::cout << "\033[1;32mProgress: " << std::fixed << std::setprecision(2) << (double)i / timesteps * 100 << "%\033[0m\r";
                std::cout.flush();
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
