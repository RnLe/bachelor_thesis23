#include "helperFunctions.h"

#include <cmath>
#include <iostream>
#include <iomanip>

std::vector<double> helperFunctions::linspace(double start, double end, std::size_t num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);

    for (std::size_t i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }

    return result;
}

std::vector<double> helperFunctions::logspace(double start, double end, std::size_t num) {
    std::vector<double> exponents = linspace(start, end, num);
    std::vector<double> result(num);

    double base = 10.0;

    for (std::size_t i = 0; i < num; ++i) {
        result[i] = std::pow(base, exponents[i]);
    }

    return result;
}


std::string helperFunctions::format_float(float number) {
    std::ostringstream out;
    // This line is for use with std=c++17
    // out << std::fixed << std::setprecision(std::numeric_limits<float>::digits10);
    // Rewrite this line for use with std=c++11
    out << std::fixed << std::setprecision(6);

    out << number;

    std::string str = out.str();
    size_t end = str.find_last_not_of('0') + 1;
    if (str[end - 1] == '.') {
        end--;
    }

    str.erase(end, std::string::npos);
    return str;
}