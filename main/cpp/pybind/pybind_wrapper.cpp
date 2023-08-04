#include <pybind11/pybind11.h>
#include "../Particle.h"
#include "../LCG.h"
#include "../Perceptron.h"
#include "../helperFunctions.h"
#include "../SwarmModel.h"
#include "../VicsekModel.h"
#include "../PerceptronModel.h"
#include "../InspectionNeighbors.h"
#include "../NeuralSwarmModel.h"

// Automatic conversion headers
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;



PYBIND11_MODULE(Solver, m) {
    // Optional docstring
    m.doc() = "pybind11 example plugin";

    // Add enums for use in classes
    // Mode enum class
    py::enum_<Mode>(m, "Mode")
        .value("RADIUS", Mode::RADIUS)
        .value("FIXED", Mode::FIXED)
        .value("QUANTILE", Mode::QUANTILE)
        .value("FIXEDRADIUS", Mode::FIXEDRADIUS);

    // NeuralNetwork enum class
    py::enum_<NeuralNetwork>(m, "NeuralNetwork")
        .value("UnitVector", NeuralNetwork::UnitVector);

    // PerceptronMode enum class
    py::enum_<PerceptronMode>(m, "PerceptronMode")
        .value("DotProduct", PerceptronMode::DotProduct);

    // Add classes
    // Particle
    py::class_<Particle>(m, "Particle")
        .def(   py::init<float, float, float, float, float, std::vector<Particle*>, std::vector<double>, int>(),
                py::arg("x"), py::arg("y"), py::arg("z"), py::arg("angle"), py::arg("polarAngle"), py::arg("k_neighbors") = py::arg(), py::arg("distances") = py::arg(), py::arg("cellRange") = 0)
        .def_readwrite("cellRange", &Particle::cellRange)
        .def_readwrite("x", &Particle::x)
        .def_readwrite("y", &Particle::y)
        .def_readwrite("z", &Particle::z)
        .def_readwrite("angle", &Particle::angle)
        .def_readwrite("polarAngle", &Particle::polarAngle)
        .def_readwrite("distances", &Particle::distances)
        .def_readwrite("k_neighbors", &Particle::k_neighbors);

    // Perceptron
    py::class_<Perceptron>(m, "Perceptron")
        .def(   py::init<int, std::vector<double>, double, double, PerceptronMode>(),
                py::arg("input_dim"), py::arg("weights") = py::arg(), py::arg("lambda_reg") = 0.1, py::arg("learning_rate") = 0.00001, py::arg("mode") = PerceptronMode::DotProduct)
        .def("forward", &Perceptron::forward)
        .def("update_weights", &Perceptron::update_weights)
        .def_readwrite("weights", &Perceptron::weights)
        .def_readwrite("lambda_reg", &Perceptron::lambda_reg)
        .def_readwrite("learning_rate", &Perceptron::learning_rate)
        .def_readwrite("mode", &Perceptron::mode);

    // SwarmModel
    py::class_<SwarmModel>(m, "SwarmModel")
        .def(   py::init<int, double, double, double, double, Mode, int, bool, bool>(),
                py::arg("N"), py::arg("L"), py::arg("v"), py::arg("noise"), py::arg("r"), py::arg("mode"), py::arg("k_neighbors"), py::arg("ZDimension") = false, py::arg("seed") = false)
        .def("update_cells", &SwarmModel::update_cells)
        .def("get_density_hist3D", &SwarmModel::get_density_hist3D)
        .def("get_dynamic_radius", &SwarmModel::get_dynamic_radius)
        .def("get_neighbors", &SwarmModel::get_neighbors)
        .def("mean_direction2D", &SwarmModel::mean_direction2D)
        .def("mean_direction3D", &SwarmModel::mean_direction3D)
        .def("density_weighted_op", &SwarmModel::density_weighted_op)
        .def("density_weighted_op_watcher", &SwarmModel::density_weighted_op_watcher)
        .def("mean_direction_watcher", &SwarmModel::mean_direction_watcher)
        .def("writeToFile", &SwarmModel::writeToFile)
        .def_readwrite("N", &SwarmModel::N)
        .def_readwrite("mode", &SwarmModel::mode)
        .def_readwrite("k_neighbors", &SwarmModel::k_neighbors)
        .def_readwrite("num_cells", &SwarmModel::num_cells)
        .def_readwrite("cellSpan", &SwarmModel::cellSpan)
        .def_readwrite("L", &SwarmModel::L)
        .def_readwrite("v", &SwarmModel::v)
        .def_readwrite("noise", &SwarmModel::noise)
        .def_readwrite("r", &SwarmModel::r)
        .def_readwrite("density2D", &SwarmModel::density2D)
        .def_readwrite("density3D", &SwarmModel::density3D)
        .def_readwrite("particles", &SwarmModel::particles)
        .def_readwrite("cells3D", &SwarmModel::cells3D)
        .def_readwrite("cells2D", &SwarmModel::cells2D)
        .def_readwrite("mode1_cells", &SwarmModel::mode1_cells)
        .def_readwrite("seed1", &SwarmModel::seed1)
        .def_readwrite("seed2", &SwarmModel::seed2)
        .def_readwrite("seed3", &SwarmModel::seed3)
        .def_readwrite("seed", &SwarmModel::seed)
        .def_readwrite("ZDimension", &SwarmModel::ZDimension);

    // VicsekModel
    py::class_<VicsekModel, SwarmModel>(m, "VicsekModel")
        .def(   py::init<int, double, double, double, double, Mode, int, bool, bool>(),
                py::arg("N"), py::arg("L"), py::arg("v"), py::arg("noise"), py::arg("r"), py::arg("mode"), py::arg("k_neighbors"), py::arg("ZDimension") = false, py::arg("seed") = false)
        .def("update", &VicsekModel::update)
        .def("average_angle_particles", &VicsekModel::average_angle_particles)
        .def("average_angle", &VicsekModel::average_angle)
        .def("get_new_particle_vicsek", &VicsekModel::get_new_particle_vicsek)
        .def("writeToFile", &VicsekModel::writeToFile);

    // PerceptronModel
    py::class_<PerceptronModel, SwarmModel>(m, "PerceptronModel")
        .def(   py::init<int, double, double, double, double, Mode, int, bool, bool>(),
                py::arg("N"), py::arg("L"), py::arg("v"), py::arg("noise"), py::arg("r"), py::arg("mode"), py::arg("k_neighbors"), py::arg("ZDimension") = false, py::arg("seed") = false)
        .def("update", &PerceptronModel::update)
        .def("writeToFile", &PerceptronModel::writeToFile)
        .def("learn", &PerceptronModel::learn)
        .def("update_unitVector", &PerceptronModel::update_unitVector)
        .def("neighbors_to_x", &PerceptronModel::neighbors_to_x)
        .def("neighbors_to_y", &PerceptronModel::neighbors_to_y)
        .def("neighbors_to_z", &PerceptronModel::neighbors_to_z)
        .def_readwrite("perceptrons", &PerceptronModel::perceptrons)
        .def_readwrite("neural_network", &PerceptronModel::neural_network);

    // Inspector
    py::class_<Inspector>(m, "Inspector")
        .def(py::init<>())
        .def("runForAllNoiseLevels_Fig2a", &Inspector::runForAllNoiseLevels_Fig2a)
        .def("runForAllNoiseLevels_Fig2b", &Inspector::runForAllNoiseLevels_Fig2b)
        .def("runForAllNoiseLevels_density_weighted", &Inspector::runForAllNoiseLevels_density_weighted)
        .def("equilibrate_va_VicsekValues_2a", &Inspector::equilibrate_va_VicsekValues_2a)
        .def("equilibrate_va_VicsekValues_2b", &Inspector::equilibrate_va_VicsekValues_2b)
        .def("equilibrate_density_weighted_op", &Inspector::equilibrate_density_weighted_op);

    // NeuralSwarmModel
    py::class_<NeuralSwarmModel, SwarmModel>(m, "NeuralSwarmModel")
        .def(   py::init<int, double, double, double, double, Mode, int, bool, bool>(),
                py::arg("N"), py::arg("L"), py::arg("v"), py::arg("noise"), py::arg("r"), py::arg("mode"), py::arg("k_neighbors"), py::arg("ZDimension") = false, py::arg("seed") = false)
        .def("update", &NeuralSwarmModel::update)
        .def("get_all_neighbors", &NeuralSwarmModel::get_all_neighbors)
        .def("get_neighbors_neural", &NeuralSwarmModel::get_neighbors_neural)
        .def("get_all_angles", &NeuralSwarmModel::get_all_angles)
        .def("get_angles", &NeuralSwarmModel::get_angles)
        .def("update_angles", &NeuralSwarmModel::update_angles)
        .def("update_angle", &NeuralSwarmModel::update_angle)
        .def("get_local_order_parameter", &NeuralSwarmModel::get_local_order_parameter);

    // LCG
    py::class_<LCG>(m, "LCG")
        .def(   py::init<ulli, ulli, ulli, ulli>(),
                py::arg("seed"), py::arg("a"), py::arg("c"), py::arg("m"))
        .def(   py::init<>())
        .def("random_f", &LCG::random_f);
}
