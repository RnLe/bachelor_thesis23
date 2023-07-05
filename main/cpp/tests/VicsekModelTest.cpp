#include "gtest/gtest.h"
#include "../VicsekModel.h"
#include "../SwarmModel.h"
#include "../Particle.h"
#include "../helperFunctions.h"

#include <map>
#include <algorithm>

class VicsekModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up code to be used for all tests
    }

    void TearDown() override {
        // We can add cleanup code here if needed
    }
};

TEST_F(VicsekModelTest, UpdateCells) {
    VicsekModel model(200, 5.0, 0.03, 0.1, 1.0, SwarmModel::RADIUS, 5, false, false);

    // Updating the cells
    model.update_cells();

    // Check if the particles are correctly distributed to the cells
    if (model.ZDimension) {
        for (int i = 0; i < int(model.particles.size()); ++i) {
            int cell_x = int(model.particles[i].x / model.r) % model.num_cells;
            int cell_y = int(model.particles[i].y / model.r) % model.num_cells;
            int cell_z = int(model.particles[i].z / model.r) % model.num_cells;

            // Check if the particles are in the expected cells in cells3D
            EXPECT_NE(std::find(model.cells3D[cell_x][cell_y][cell_z].begin(), model.cells3D[cell_x][cell_y][cell_z].end(), i), model.cells3D[cell_x][cell_y][cell_z].end());
        }
    } else {
        for (int i = 0; i < int(model.particles.size()); ++i) {
            int cell_x = int(model.particles[i].x / model.r) % model.num_cells;
            int cell_y = int(model.particles[i].y / model.r) % model.num_cells;

            // Check if the particles are in the expected cells in cells2D
            EXPECT_NE(std::find(model.cells2D[cell_x][cell_y].begin(), model.cells2D[cell_x][cell_y].end(), i), model.cells2D[cell_x][cell_y].end());
        }
    }
}
