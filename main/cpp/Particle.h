#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

class Particle {
    public:
                                                    Particle        (float x, float y, float z, float angle, float polarAngle, std::vector<Particle*> k_neighbors = {}, int cellRange = 0);

        int                                         cellRange;
        float                                       x, y, z, angle, polarAngle;
        std::vector<Particle*>                      k_neighbors;
};

#endif // PARTICLE_H
