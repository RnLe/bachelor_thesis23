#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

class RunningAverage {
    double total = 0.0;
    int count = 0;
public:
    void add(double value) {
        total += value;
        count++;
    }
    double average() {
        return count ? total / count : 0.0;
    }
    void clear() {
        total = 0.0;
        count = 0;
    }
};

class Particle {
public:
    double x, y, angle;
    Particle(double x, double y, double angle) : x(x), y(y), angle(angle) {}
};

class VicsekModel {
    int N, num_cells, mode;
    double L, v, noise, r, density;
    std::vector<Particle> particles;
    std::vector<std::vector<std::vector<int>>> cells;
public:
    VicsekModel(int N, double L, double v, double noise, double r, int mode = 0)
        : N(N), L(L), v(v), noise(noise), r(r), mode(mode), density(N / (L * L)),
          num_cells(int(L / r)), cells(num_cells, std::vector<std::vector<int>>(num_cells)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, L), dis2(0, 2 * M_PI);
        for(int i = 0; i < N; ++i)
            particles.emplace_back(dis(gen), dis(gen), dis2(gen));
    }
    void update_cells() {
        for(int i = 0; i < num_cells; ++i)
            for(int j = 0; j < num_cells; ++j)
                cells[i][j].clear();
        for(int i = 0; i < N; ++i) {
            int cell_x = int(particles[i].x / r) % num_cells;
            int cell_y = int(particles[i].y / r) % num_cells;
            cells[cell_x][cell_y].push_back(i);
        }
    }

    void update() {
        std::vector<Particle> new_particles;
        update_cells();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-noise / 2, noise / 2);
        for(int i = 0; i < N; ++i) {
            int cell_x = int(particles[i].x / r);
            int cell_y = int(particles[i].y / r);
            std::vector<Particle> neighbours;
            for(int dx = -1; dx <= 1; ++dx)
                for(int dy = -1; dy <= 1; ++dy) {
                    int neighbour_cell_x = (cell_x + dx + num_cells) % num_cells;
                    int neighbour_cell_y = (cell_y + dy + num_cells) % num_cells;
                    for(int j : cells[neighbour_cell_x][neighbour_cell_y])
                        if(i != j && hypot(particles[i].x - particles[j].x - L * round((particles[i].x - particles[j].x) / L), 
                                        particles[i].y - particles[j].y - L * round((particles[i].y - particles[j].y) / L)) < r)
                            neighbours.push_back(particles[j]);
                }
            double avg_angle = particles[i].angle;
            if(!neighbours.empty()) {
                double sum_sin = 0, sum_cos = 0;
                for(const Particle& p : neighbours) {
                    sum_sin += sin(p.angle);
                    sum_cos += cos(p.angle);
                }
                avg_angle = atan2(sum_sin / neighbours.size(), sum_cos / neighbours.size());
            }
            double new_angle = avg_angle + dis(gen);
            double new_x = (particles[i].x + v * cos(new_angle)) - floor((particles[i].x + v * cos(new_angle)) / L) * L;
            double new_y = (particles[i].y + v * sin(new_angle)) - floor((particles[i].y + v * sin(new_angle)) / L) * L;
            new_particles.emplace_back(new_x, new_y, new_angle);
        }
        particles = new_particles;
    }

    double va() {
        double sum_sin = 0, sum_cos = 0;
        for(const Particle& p : particles) {
            sum_sin += sin(p.angle);
            sum_cos += cos(p.angle);
        }
        return hypot(sum_cos / N, sum_sin / N);
    }

};

int main() {
    // Flags
    bool plot1 = false, plot2 = true;

    // Initialize Model
    std::map<std::string, std::vector<double>> settings = {
        // {"a", {300, 7, 0.03, 2.0, 1, 4}},
        // {"b", {300, 25, 0.03, 0.1, 1, 1}},
        // {"d", {300, 5, 0.03, 0.1, 1, 4}},
        {"plot1_N40", {40, 3.1, 0.03, 0.1, 1, 4}},
        {"plot1_N100", {100, 5, 0.03, 0.1, 1, 4}},
        {"plot1_N400", {400, 10, 0.03, 0.1, 1, 4}},
        {"plot1_N4000", {4000, 31.6, 0.03, 0.1, 1, 4}},
        {"plot1_N10000", {10000, 50, 0.03, 0.1, 1, 4}}
    };

    std::vector<double> noises = {};

    for (int i = 0; i <= 50; i++){
        noises.push_back(5./50. * i);
    }
    
    // std::vector<double> noises = {0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.};

    int num_frames = 200000, mode = 1, N_cutoff = floor(num_frames / 20);
    int progress = 0, total = settings.size() * noises.size();

    std::ofstream file("output2.csv");
    file << "N,L,noise,average_va,last_va\n";

    if (plot1) { 
        for (const auto& setting : settings) {
            if (setting.first.find("plot1") != 0) continue;  // Skip settings not starting with "plot1"

            double N = setting.second[0];
            double L = setting.second[1];
            double v = setting.second[2];
            double r = setting.second[4];
            double scale = setting.second[5];
            
            #pragma omp parallel for
            for (double noise : noises) {
                std::vector<double> va_values;
                RunningAverage avg_va;
                VicsekModel model(N, L, v, noise, r, mode);

                for(int i = 0; i < num_frames; ++i) {
                    model.update();
                    double va = model.va();
                    va_values.push_back(va);
                    if (i >= num_frames - N_cutoff)  // Only add to average for the last 500 frames
                        avg_va.add(va);
                    else if (i == num_frames - N_cutoff)  // Clear the average when we reach the last 500 frames
                        avg_va.clear();
                    std::cout << "\rProgress: " << progress * 100 / total << "%" << ", Noise: " << noise << " , N: " << N << " , Current: " << i * 100 / num_frames<< "%                " << "\e[?25l";
                }
                file << N << "," << L << "," << noise << "," << avg_va.average() << "," << va_values.back() << "\n";
                progress++;
            }
        }
    } if (plot2) {
        std::vector<double> densities = {};
        num_frames = 200000, N_cutoff = floor(num_frames / 20);
        double L = 20, noise = 2.5, v = 0.03, r = 1.;
        for (int i = 0; i < 20; i++) {
            densities.push_back(2./20. * i);
            densities.push_back(2. + 8./20. * i);
        }
        total = densities.size();
        #pragma omp parallel for
        for (double density : densities) {
                std::vector<double> va_values;
                RunningAverage avg_va;
                VicsekModel model(density * L * L, L, v, noise, r, mode);

                for(int i = 0; i < num_frames; ++i) {
                    model.update();
                    double va = model.va();
                    va_values.push_back(va);
                    if (i >= num_frames - N_cutoff)  // Only add to average for the last 500 frames
                        avg_va.add(va);
                    else if (i == num_frames - N_cutoff)  // Clear the average when we reach the last 500 frames
                        avg_va.clear();
                    std::cout << "\rProgress: " << progress * 100 / total << "%" << ", Noise: " << noise << " , N: " << density * L * L << " , Current: " << i * 100 / num_frames<< "%                " << "\e[?25l";
                }
                file << density * L * L << "," << L << "," << noise << "," << avg_va.average() << "," << va_values.back() << "\n";
                progress++;
            }
    }
    std::cout << "\rProgress: 100%\n";
    file.close();
    return 0;
}