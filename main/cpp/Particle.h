#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

class Particle {
    public:
        Particle(float x, float y, float z, float angle, float polarAngle, std::vector<Particle*> k_neighbors = {}, int cellRange = 0)
        : x(x), y(y), z(z), angle(angle), polarAngle(polarAngle), k_neighbors(k_neighbors), cellRange(cellRange) {};

        float x;
        float y;
        float z;
        float angle;
        float polarAngle;
        std::vector<Particle*> k_neighbors;
        int cellRange;
};

#endif // PARTICLE_H