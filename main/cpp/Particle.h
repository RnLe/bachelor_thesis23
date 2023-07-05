#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

class Particle {
    public:
                                                    Particle    (float x, float y, float z, float angle, float polarAngle, std::vector<Particle*> k_neighbors = {}, std::vector<double> distances = {}, int cellRange = 0)
                                                                : x(x), y(y), z(z), angle(angle), polarAngle(polarAngle), k_neighbors(k_neighbors), cellRange(cellRange), distances(distances) {};

        int                                         cellRange;
        float                                       x, y, z, angle, polarAngle;
        std::vector<Particle*>                      k_neighbors;
        std::vector<double>                         distances;
};

#endif // PARTICLE_H
