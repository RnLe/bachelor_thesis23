#include <vector>

class Particle {
public:
    // Konstruktor
    Particle(float x, float y, float angle, std::vector<Particle*> k_neighbors = {}, int cellRange = 0)
        : x(x), y(y), angle(angle), k_neighbors(k_neighbors), cellRange(cellRange) {}

private:
    float x;
    float y;
    float angle;
    std::vector<Particle*> k_neighbors;
    int cellRange;
};
