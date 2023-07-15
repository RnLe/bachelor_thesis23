# ifndef __LCG_H__
# define __LCG_H__

#include <cmath>

typedef unsigned long long int ulli;

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// CLASSES
class LCG {
    public:
                                                LCG(ulli seed, ulli a, ulli c, ulli m) : seed(seed), a(a), c(c), m(m) {};
                                                // If no seed is given, use the IBM default
                                                LCG() : seed(1), a(65539), c(0), m(pow(2, 31)) {};
        float                                   random_f();          
    private:
        ulli                                    seed;
        ulli                                    a;
        ulli                                    c;
        ulli                                    m;

        ulli                                    iteration_step();

};

// This method is used to generate a random float between 0 and 1, uniformly distributed
inline float LCG::random_f() {
    ulli result = iteration_step();
    return static_cast<float>(result) / static_cast<float>(m);
}

inline ulli LCG::iteration_step() {
    seed = (a * seed + c) % m;
    return seed;
}

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// CLASSES END

# endif // __LCG_H__