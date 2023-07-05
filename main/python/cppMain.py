from Solver import Particle, Perceptron, PerceptronModel, VicsekModel, NeuralNetwork, PerceptronMode, Mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Slider

if __name__ == "__main__":
    
    # Hyperparameters
    
    settings = {
        #                  N,      L,      v,      noise,  r
        "XXsmall": [       5,      4,      0.03,   0.1,    1],
        "Xsmall": [        20,     6,      0.03,   0.1,    1],
        "small": [         100,    30,     0.03,   0.1,    3],
        "a": [             300,    7,      0.03,   2.0,    1],
        "b": [             300,    25,     0.03,   0.5,    1],
        "d": [             300,    5,      0.03,   0.1,    1],
        "plot1_N40": [     40,     3.1,    0.03,   0.1,    1],
        "large": [         2000,   60,     0.03,   0.3,    1],
        "Xlarge": [        5000,   60,     0.03,   0.5,    1],
        "XlargeR3": [      5000,   60,     0.03,   0.2,    3],
        "XXlarge": [       10000,  60,     0.03,   0.1,    1],
        "XXlargeR2": [     10000,  60,     0.03,   0.1,    2],
        "XXlargeR5": [     10000,  60,     0.03,   0.1,    5],
        "XXlargeR5n0": [   10000,  60,     0.03,   0.,     5],
        "XXlargeR20": [    10000,  60,     0.03,   0.1,    20],
        "XXlargefast": [   10000,  60,     0.1,    0.1,    1],
        "XXXlarge": [      20000,  60,     0.03,   0.1,    1],
        "Ultralarge": [    200000, 60,     0.03,   0.1,    1]
    }
    
    # Choose between RADIUS, FIXED, QUANTILE, FIXEDRADIUS
    mode = Mode.FIXEDRADIUS

    # Flags
    ZDimension = False     # 2D or 3D

    # Duration of simulation
    timesteps = 5000

    # Choose settings
    chosen_settings = settings["small"]
    N       = chosen_settings[0]
    L       = chosen_settings[1]
    v       = chosen_settings[2]
    noise   = chosen_settings[3]
    r       = chosen_settings[4]
    # Calculate exchange radius from density; (N / L^2) * r^2
    # Example for N = 5000, L = 60, r = 1;
    k       = (N * r * r) / (L * L)
    k_neighbors = 5
    
    # Create model
    model = VicsekModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=True)
    # model = PerceptronModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=True)

    # Write to file
    # model.writeToFile(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise)

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create sliders
    axcolor = 'lightgoldenrodyellow'
    axNoise = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)
    axRadius = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
    axNeighbors = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)

    sliderNoise = Slider(axNoise, 'Noise', 0.1, 3.0, valinit=noise)
    sliderRadius = Slider(axRadius, 'Radius', 1.0, 5.0, valinit=r)
    sliderNeighbors = Slider(axNeighbors, 'Neighbors', 1, 20, valinit=k_neighbors, valstep=1)
    
    # Misc figure and axes settings
    ax.set_aspect('equal')

    def update(val):
        # Read slider values and update model
        new_noise = sliderNoise.val
        new_r = sliderRadius.val
        new_k_neighbors = int(sliderNeighbors.val)
        
        model.noise = new_noise
        model.r = new_r
        model.k_neighbors = new_k_neighbors
        
        global r
        r = new_r


    sliderNoise.on_changed(update)
    sliderRadius.on_changed(update)
    sliderNeighbors.on_changed(update)

    def update_animation(timestep):
        # Delete all quivers and circles
        ax.clear()
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        
        model.update()
        
        # Draw new arrows and circles
        for particle in model.particles:
            ax.quiver(particle.x, particle.y, np.cos(particle.angle), np.sin(particle.angle), angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=3, headlength=4, headaxislength=3, color='k')
        
        # Select one particle
        first_particle = model.particles[0]
        
        # Draw circle around first particle
        circle = plt.Circle((first_particle.x, first_particle.y), r, color='b', fill=False)
        ax.add_artist(circle)
        
        # Mark neighbors of first particle
        for i, neighbor in enumerate(first_particle.k_neighbors):
            # Check whether neighbor is empty
            if neighbor is not None and neighbor != first_particle:
                # Draw line to neighbor
                ax.plot([first_particle.x, neighbor.x], [first_particle.y, neighbor.y], color='green')
                # Write distance to neighbor on the line
                distance = first_particle.distances[i]
                ax.text((first_particle.x + neighbor.x) / 2, (first_particle.y + neighbor.y) / 2, str(round(distance, 2)), fontsize=12, verticalalignment='top', horizontalalignment='left')
            
        # To the left of the plot, write the number of total neighbors which are not None
        ax.text(0, L + 2, "Neighbors: " + str(len([neighbor for neighbor in first_particle.k_neighbors if neighbor is not None])), fontsize=12, verticalalignment='top', horizontalalignment='left')
        
        # Draw cells around first particle
        for dx in [-1, 0, 1] if model.mode == Mode.RADIUS else list(range(-first_particle.cellRange, first_particle.cellRange + 1)):
            for dy in [-1, 0, 1] if model.mode == Mode.RADIUS else list(range(-first_particle.cellRange, first_particle.cellRange + 1)):
                # Position of the cell
                cell_x = (int(first_particle.x / (2 * model.r)) + dx) % model.num_cells
                cell_y = (int(first_particle.y / (2 * model.r)) + dy) % model.num_cells

                # Create rectangle
                cell = patches.Rectangle((cell_x * (2 * model.r), cell_y * (2 * model.r)), (2 * model.r), (2 * model.r), linewidth=1, edgecolor='whitesmoke', facecolor='green', alpha=0.6)

                ax.add_patch(cell)
                
        # Draw all cells
        # The cell width and height is 2 * r
        for cell_x in range(model.num_cells):
            for cell_y in range(model.num_cells):
                # Create rectangle
                cell = patches.Rectangle((cell_x * (2 * model.r), cell_y * (2 * model.r)), (2 * model.r), (2 * model.r), linewidth=1, edgecolor='grey', facecolor='blue', alpha=0.1)
                
                # Count the number of particles in the cell
                # :::::::::::::::::::::::::::::::::::::::::
                # std::vector<std::vector<std::vector<int>>> cells2D;
                # Where cells2D[cell_x][cell_y] is a vector of particle indices
                
                cells2D = model.cells2D
                num_particles = len(cells2D[cell_x][cell_y])
                
                # Write number of particles in cell
                ax.text(cell_x * (2 * model.r) + model.r, cell_y * (2 * model.r) + model.r, str(num_particles), horizontalalignment='center', verticalalignment='center', fontsize=8)

                ax.add_patch(cell)
            

    ani = animation.FuncAnimation(fig, update_animation, interval=30)

    plt.show()
