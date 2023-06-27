#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

/*
This header is a collection of helper functions. Some functions imitate numpy functions. Some other functions are completely custom.
*/

#include <vector>
#include <cmath>
#include <sstream>

// Define a custom namespace to avoid conflicts
namespace helperFunctions
{
    
    // TODO: Make a template for multiple data types
    std::vector<double>                             linspace            (double start, double end, std::size_t num);
    std::vector<double>                             logspace            (double start, double end, std::size_t num);
    std::string                                     format_float        (float number);

}

#endif // HELPER_FUNCTIONS_H