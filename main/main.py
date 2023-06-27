import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from matplotlib.widgets import Button
from customTimer import Timer
from runningAverage import RunningAverage
from perceptron import Perceptron
from swarmModel import SwarmModel, VicsekModel, PerceptronModel

import cProfile

def animate(i):
    """Updates the plot for each frame."""
    if not calcOnly:
        ax1.clear()
        ax1.set_xlim(0, L)
        ax1.set_ylim(0, L)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.quiver([p.x for p in model.particles], [p.y for p in model.particles], np.cos([p.angle for p in model.particles]), np.sin([p.angle for p in model.particles]), scale_units='xy', scale=scale)
        
        # Draw circle around first particle
        first_particle = model.particles[0]
        circle = patches.Circle((first_particle.x, first_particle.y), radius=model.r, fill=False, color="blue")
        ax1.add_patch(circle)
        
        # Mark neighbors of first particle
        for neighbor in first_particle.k_neighbors[1:]:
            ax1.plot([first_particle.x, neighbor.x], [first_particle.y, neighbor.y], color='red')  # draw line to neighbor
            # ax1.scatter(neighbor.x, neighbor.y, color='red')  # mark neighbor
            
        label_text = f'Neighbors: {len(first_particle.k_neighbors) - 1}\n\nMode: {"radius" if mode == 0 else "fixed"}'
        ax1.text(L + 8, L - 1, label_text, fontsize=18, ha='right', va='top')
        
        for dx in [-1, 0, 1] if model.mode == 0 else list(range(-first_particle.cellRange, first_particle.cellRange + 1)):
            for dy in [-1, 0, 1] if model.mode == 0 else list(range(-first_particle.cellRange, first_particle.cellRange + 1)):
                # Position of the cell
                cell_x = (int(first_particle.x / model.r) + dx) % model.num_cells
                cell_y = (int(first_particle.y / model.r) + dy) % model.num_cells

                # Create rectangle
                cell = patches.Rectangle((cell_x * model.r, cell_y * model.r), model.r, model.r, linewidth=1, edgecolor='whitesmoke', facecolor='green', alpha=0.6)

                ax1.add_patch(cell)

    
    model.update()
    
    ax2.set_xlim(0, i+1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Order parameter')
    va_values.append(model.va())
    avg_va.add(va_values[-1])
    avg_va_list.append(avg_va.average(i+1))
    line_va.set_data(range(len(va_values)), va_values)
    line_avg_va.set_data(range(len(avg_va_list)), avg_va_list)

    # Create histograms
    if histograms:
        ax3.clear()
        ax5.clear()
        ax3.hist(model.get_dynamic_radius(), bins=7, alpha=0.5, label='Dynamic Radius')
        ax3.set_xlim((0, 6))
        ax3.set_ylim((0, 200))
        ax5.hist(model.get_density_hist(), bins=20, alpha=0.5, label='Cell Densities', color='orange')
        ax5.set_ylim((0, 500))
        # ax5.set_xlim((0, 1))
        
        ax3.legend(loc='upper right')
        ax5.legend(loc='upper right')
    
    
def update_noise(val):
    """Updates the noise value in the model."""
    model.noise = val
    
def update_neighbors(val):
    """Updates the noise value in the model."""
    model.k_neighbors = val
    
def toggle_mode(event):
        global mode
        mode = not mode  # Toggle the mode
        model.mode = mode

if __name__ == "__main__":
    # Start profiling
    pr = cProfile.Profile()
    pr.enable()
    
    # Flags
    calcOnly = False
    use_perceptron_model = False  # Set this to True to use the PerceptronModel, False to use the VicsekModel
    training = False
    histograms = False

    # Effectively the time steps t
    num_frames = 2000

    # Initialize Model
    modes = {"radius": 0, "fixed": 1}
    mode = modes["fixed"]
    
    settings = {
        "small": [20, 10, 0.03, 0.2, 1, 1],
        "a": [300, 7, 0.03, 2.0, 1, 4],
        "b": [300, 25, 0.03, 0.5, 1, 1],
        "d": [300, 5, 0.03, 0.1, 1, 4],
        "plot1_N40": [40, 3.1, 0.03, 0.1, 1, 4],
        "large": [2000, 50, 0.03, 0.1, 1, 4]      
    }
    N, L, v, noise, r, scale = settings["b"]
    k_neighbors = 8
    cellSpan = 5 if mode == 1 else 1
    va_values = []
    avg_va_list = []
    fluctuations = []
    avg_va = RunningAverage()

    if use_perceptron_model:
        model = PerceptronModel(N, L, v, noise, r, mode=mode, k_neighbors=k_neighbors)
        
        if training:
            iterations = 100
            for i in range(iterations):
                print(f"Weights: {model.perceptron.weights}")
                model.learn()
        
        weights = model.perceptron.weights
        model = PerceptronModel(N, L, v, noise, r, mode=mode, k_neighbors=k_neighbors, weights=weights)
    else:
        model = VicsekModel(N, L, v, noise, r, mode=mode, k_neighbors=k_neighbors)

    fig, axes = plt.subplot_mosaic("AABB;AAEE;CCDD;CCDD", figsize=(10, 12))
    ax1, ax3, ax4, ax2, ax5 = axes['A'], axes['B'], axes['C'], axes['D'], axes['E']
    
    ax4.remove()
    
    # Add legend
    ax3.legend(loc='upper right')
    ax5.legend(loc='upper right')

    # Add labels
    ax3.set_xlabel('Radius')
    ax3.set_ylabel('Count')
    
    ax5.set_xlabel('Density')
    ax5.set_ylabel('Count')
    
    ax1.set_aspect('equal')
    line_va, = ax2.plot(va_values, label="Current order parameter")
    line_avg_va, = ax2.plot(avg_va_list, label="Overall average order parameter")
    plt.subplots_adjust(bottom=0.25)

    # Sliders
    # Add slider for the noise
    ax_noise = plt.axes([0.1, 0.2, 0.25, 0.03])

    slider_noise = Slider(ax_noise, 'Noise', 0.01, 5.0, valinit=noise)
    slider_noise.on_changed(update_noise)
    
    # Add slider for the neighbors
    ax_neighbors = plt.axes([0.1, 0.15, 0.25, 0.03])

    slider_neighbors = Slider(ax_neighbors, 'Neighbors', 0, 50, valinit=k_neighbors, valstep=1)
    slider_neighbors.on_changed(update_neighbors)
    
    # Buttons
    button_ax = plt.axes([0.1, 0.05, 0.1, 0.03])
    button = Button(button_ax, 'Toggle Mode', color='lightgoldenrodyellow', hovercolor='0.975')

    button.on_clicked(toggle_mode)

    # Animate function
    ax2.legend()
    plt.tight_layout()
    if not calcOnly:
        ani = FuncAnimation(fig, animate, frames=num_frames, interval=1)
        plt.show()
    else:
        model.writeToFile(2000, "xyz")
    plt.close()

    pr.disable()
    pr.dump_stats("my_program.prof")