#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

class Particle {
    public:
        Particle(float x, float y, float angle, std::vector<Particle*> k_neighbors = {}, int cellRange = 0)
        : x(x), y(y), angle(angle), k_neighbors(k_neighbors), cellRange(cellRange) {};

        float x;
        float y;
        float angle;
        std::vector<Particle*> k_neighbors;
        int cellRange;
};

#endif