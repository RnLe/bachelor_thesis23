from Solver import Particle, Perceptron, PerceptronModel, VicsekModel, NeuralNetwork, PerceptronMode, Mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec

if __name__ == "__main__":
    
    # Hyperparameters
    
    settings = {
        #                  N,      L,      v,      noise,  r
        "XXsmall": [       5,      4,      0.03,   0.1,    1],
        "Xsmall": [        50,     10,      0.03,   0.1,    1],
        "small": [         200,    30,     0.03,   0.1,    1],
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
        "Ultralarge": [    200000, 60,     0.03,   0.1,    1],
        
        "BaseNoiseAnalysis40": [       40,     3.1,    0.03,   0.1,    1],
        "BaseNoiseAnalysis100": [      100,    5,      0.03,   0.1,    1],
        "BaseNoiseAnalysis400": [      400,    10,     0.03,   0.1,    1],
        "BaseNoiseAnalysis1600": [     1600,   20,     0.03,   0.1,    1]
    }
    
    # Choose between RADIUS, FIXED, QUANTILE, FIXEDRADIUS
    mode = Mode.RADIUS

    # Flags
    ZDimension = False     # 2D or 3D

    # Duration of simulation
    timesteps = 5000

    # Choose settings
    chosen_settings = settings["Xsmall"]
    N       = chosen_settings[0]
    L       = chosen_settings[1]
    v       = chosen_settings[2]
    noise   = chosen_settings[3]
    r       = chosen_settings[4]
    # Calculate exchange radius from density; (N / L^2) * r^2
    # Example for N = 5000, L = 60, r = 1;
    k       = (N * r * r) / (L * L)
    k_neighbors = 2
    
    va_array = []
    density_array = []
    
    # Create model
    model = VicsekModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=True)
    # model = PerceptronModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=True)

    # Write to file
    # model.writeToFile(timesteps, "xyz", N, L, v, r, mode, k_neighbors, noise)

    # Create animation and control areas
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 3) 

    ax = plt.subplot(gs[0,:])
    plt.axis('off')  # hide the axes

    # Create axes for animation
    ax1 = plt.subplot(gs[:,1:])
    ax2 = plt.subplot(gs[2,0])
    
    
    # Misc figure and axes settings
    ax1.set_aspect('equal')

    # Create sliders
    axcolor = 'lightgoldenrodyellow'
    axNoise = plt.axes([0.1, 0.90, 0.2, 0.03], facecolor=axcolor)
    axRadius = plt.axes([0.1, 0.85, 0.2, 0.03], facecolor=axcolor)
    axNeighbors = plt.axes([0.1, 0.80, 0.2, 0.03], facecolor=axcolor)

    sliderNoise = Slider(axNoise, 'Noise', 0.1, 3.0, valinit=noise)
    sliderRadius = Slider(axRadius, 'Radius', 1.0, 5.0, valinit=r)
    sliderNeighbors = Slider(axNeighbors, 'Neighbors', 1, 20, valinit=k_neighbors, valstep=1)
    
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
    
    # Create buttons
    # Buttons for: start/stop (toggle), reset
    
    # Button functions
    def buttonStartPress(event):
        global buttonStartFlag
        buttonStartFlag = not buttonStartFlag
        
        # Change button text
        if buttonStartFlag:
            buttonStart.label.set_text("Stop")
        else:
            buttonStart.label.set_text("Start")
        
    def buttonResetPress(event):
        global buttonStartFlag
        global model
        model = VicsekModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=True)
        # Clear arrays
        va_array.clear()
        density_array.clear()
        update_animation(0)
        buttonStartFlag = False
        buttonStart.label.set_text("Start")
        
    def buttonGhostPress(event):
        global buttonGhostFlag
        buttonGhostFlag = not buttonGhostFlag
        
        # Change button text
        if buttonGhostFlag:
            buttonGhost.label.set_text("Show")
        else:
            buttonGhost.label.set_text("Hide")
    
    axStart = plt.axes([0.1, 0.75, 0.066, 0.04])
    axGhost = plt.axes([0.166, 0.75, 0.066, 0.04])
    axReset = plt.axes([0.232, 0.75, 0.068, 0.04])

    buttonStart = Button(axStart, 'Start', color=axcolor, hovercolor='0.975')
    buttonGhost = Button(axGhost, 'Hide', color=axcolor, hovercolor='0.975')
    buttonReset = Button(axReset, 'Reset', color=axcolor, hovercolor='0.975')

    buttonStart.on_clicked(buttonStartPress)
    buttonGhost.on_clicked(buttonGhostPress)
    buttonReset.on_clicked(buttonResetPress)
    
    # Button flags
    buttonStartFlag = False
    buttonGhostFlag = False
        
    # Create RadioButtons
    # Radiobuttons for
    # Mode: Radius, Fixed, FixedRadius
    # Grid: off, on, dynamic, count, density
    # Distances: off, lines, lines+text
    # Plots: off, va, densities
    
    def update_mode(label):
        global mode
        if label == 'Radius':
            mode = Mode.RADIUS
        elif label == 'Fixed':
            mode = Mode.FIXED
        elif label == 'FixedRadius':
            mode = Mode.FIXEDRADIUS
            
        buttonResetPress(None) 
        
    axMode = plt.axes([0.1, 0.65, 0.2, 0.1], facecolor=axcolor)
    axGrid = plt.axes([0.1, 0.55, 0.2, 0.1], facecolor=axcolor)
    axDistances = plt.axes([0.1, 0.45, 0.2, 0.1], facecolor=axcolor)
    axPlots = plt.axes([0.1, 0.35, 0.2, 0.1], facecolor=axcolor)

    radioMode = RadioButtons(axMode, ('Radius', 'Fixed', 'FixedRadius'), active=0)
    radioGrid = RadioButtons(axGrid, ('Off', 'On', 'Dynamic', 'Count', 'Density'), active=0)
    radioDistances = RadioButtons(axDistances, ('Off', 'Lines', 'Lines+Text'), active=0)
    radioPlots = RadioButtons(axPlots, ('Off', 'Va', 'Densities', 'All'), active=0)
    
    # On radioMode change, update mode
    radioMode.on_clicked(update_mode)

    def update_animation(timestep):
        # If buttonGhostFlag is True, only update model and data, but don't draw anything
        if buttonGhostFlag:
            model.update()
            va_array.append(model.mean_direction2D())
            density_array.append(model.density_weighted_op())
            ax1.clear()
            # Label frame number (from len(va_array))
            ax1.text(6, -2, "Frame: " + str(len(va_array)), fontsize=12, verticalalignment='top', horizontalalignment='left')
            return
        
        # If button is pressed, update animation
        if buttonStartFlag:
            # Delete all quivers and circles
            ax1.clear()
            ax1.set_xlim(0, L)
            ax1.set_ylim(0, L)
            
            model.update()
            
            # Draw new arrows and circles
            for particle in model.particles:
                ax1.quiver(particle.x, particle.y, np.cos(particle.angle), np.sin(particle.angle), angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=3, headlength=4, headaxislength=3, color='k')
            
            # Select one particle
            first_particle = model.particles[0]
            
            # Draw circle around first particle
            circle = plt.Circle((first_particle.x, first_particle.y), r, color='b', fill=False)
            ax1.add_artist(circle)
            
            if radioDistances.value_selected == 'Lines' or radioDistances.value_selected == 'Lines+Text':
                # Mark neighbors of first particle
                for i, neighbor in enumerate(first_particle.k_neighbors):
                    # Check whether neighbor is empty
                    if neighbor is not None and neighbor != first_particle:
                        # Draw line to neighbor
                        ax1.plot([first_particle.x, neighbor.x], [first_particle.y, neighbor.y], color='green')
                        if radioDistances.value_selected == 'Lines+Text':
                            # Write distance to neighbor on the line
                            distance = first_particle.distances[i]
                            ax1.text((first_particle.x + neighbor.x) / 2, (first_particle.y + neighbor.y) / 2, str(round(distance, 2)), fontsize=12, verticalalignment='top', horizontalalignment='left')
                            # Show position of neighbors as text; display the distance behind the coordinates (above the plot)
                            ax1.text(6, L + 0.9*i, str(round(neighbor.x, 2)) + ", " + str(round(neighbor.y, 2)) + ", Distance: " + str(round(distance, 2)), fontsize=12, verticalalignment='top', horizontalalignment='left')
                            
                
            # To the left of the plot, write the number of total neighbors which are not None
            ax1.text(0, L + 2, "Neighbors: " + str(len([neighbor for neighbor in first_particle.k_neighbors if neighbor is not None]) - 1), fontsize=12, verticalalignment='top', horizontalalignment='left')
            
            if radioGrid.value_selected == 'Dynamic':
                # Draw cells around first particle
                for dx in [-1, 0, 1] if model.mode == Mode.RADIUS else list(range(-first_particle.cellRange, first_particle.cellRange + 1)):
                    for dy in [-1, 0, 1] if model.mode == Mode.RADIUS else list(range(-first_particle.cellRange, first_particle.cellRange + 1)):
                        # Position of the cell
                        cell_x = (int(first_particle.x / (2 * model.r)) + dx) % model.num_cells
                        cell_y = (int(first_particle.y / (2 * model.r)) + dy) % model.num_cells

                        # Create rectangle
                        cell = patches.Rectangle((cell_x * (2 * model.r), cell_y * (2 * model.r)), (2 * model.r), (2 * model.r), linewidth=1, edgecolor='whitesmoke', facecolor='green', alpha=0.6)

                        ax1.add_patch(cell)
            
            if radioGrid.value_selected == 'On' or radioGrid.value_selected == 'Count':  
                # Draw all cells
                # The cell width and height is 2 * r
                for cell_x in range(model.num_cells):
                    for cell_y in range(model.num_cells):
                        # Create rectangle
                        cell = patches.Rectangle((cell_x * (2 * model.r), cell_y * (2 * model.r)), (2 * model.r), (2 * model.r), linewidth=1, edgecolor='grey', facecolor='blue', alpha=0.2)
                        
                        # Count the number of particles in the cell
                        # :::::::::::::::::::::::::::::::::::::::::
                        # std::vector<std::vector<std::vector<int>>> cells2D;
                        # Where cells2D[cell_x][cell_y] is a vector of particle indices
                        
                        cells2D = model.cells2D
                        num_particles = len(cells2D[cell_x][cell_y])
                        
                        if radioGrid.value_selected == 'Count':
                            # Write number of particles in cell
                            ax1.text(cell_x * (2 * model.r) + model.r, cell_y * (2 * model.r) + model.r, str(num_particles), horizontalalignment='center', verticalalignment='center', fontsize=8)

                        ax1.add_patch(cell)
                        
                # Add text for total cell count
                ax1.text(0, -2, "Cells: " + str(model.num_cells * model.num_cells), fontsize=12, verticalalignment='top', horizontalalignment='left')
            
            # Add values to arrays
            va_array.append(model.mean_direction2D())
            density_array.append(model.density_weighted_op())
            
            if radioPlots.value_selected == 'Va':
                # Label the average velocity from model.mean_direction2D()
                ax1.text(0, L + 4, "Va: " + str(round(va_array[-1], 2)), fontsize=12, verticalalignment='top', horizontalalignment='left')
                # Plot the average velocity over time
                ax2.clear()
                ax2.plot(range(len(va_array)), va_array, label="Va")
                ax2.legend()
                ax2.set_xlim(0, len(va_array))
            elif radioPlots.value_selected == 'Densities':
                # Label the density from model.density_weighted_op()
                ax1.text(0, L + 4, "Density: " + str(round(density_array[-1], 2)), fontsize=12, verticalalignment='top', horizontalalignment='left')
                ax2.clear()
                ax2.plot(range(len(density_array)), density_array, label="Density")
                ax2.legend()
                ax2.set_xlim(0, len(va_array))
            elif radioPlots.value_selected == 'All':
                # Label all order parameters
                ax1.text(0, L + 4, "Va: " + str(round(va_array[-1], 2)), fontsize=12, verticalalignment='top', horizontalalignment='left')
                ax1.text(0, L + 3, "Density: " + str(round(density_array[-1], 2)), fontsize=12, verticalalignment='top', horizontalalignment='left')
                ax2.clear()
                ax2.plot(range(len(va_array)), va_array, label="Va")
                ax2.plot(range(len(density_array)), density_array, label="Density")
                ax2.legend()
                ax2.set_xlim(0, len(va_array))
                
            # Label frame number (from len(va_array))
            ax1.text(6, -2, "Frame: " + str(len(va_array)), fontsize=12, verticalalignment='top', horizontalalignment='left')
            

    ani = animation.FuncAnimation(fig, update_animation, interval=30)

    plt.show()
