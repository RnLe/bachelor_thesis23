import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.patches as patches
import time

class Timer:
    """A simple timer class."""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def start(self):
        """Starts the timer."""
        self.start_time = time.time()

    def end(self):
        """Ends the timer."""
        self.end_time = time.time()

    def show(self):
        """Prints the time needed."""
        if self.start_time is not None and self.end_time is not None:
            print(f"{self.name}: {self.end_time - self.start_time} seconds")
        else:
            print("Timer has not started and ended properly.")
            
class RunningAverage:
    """A class that calculates a running average."""
    def __init__(self):
        self.total = 0.0

    def add(self, value: float):
        """Adds a value to the total."""
        self.total += value

    def average(self, count: int):
        """Calculates the average."""
        return self.total / count if count else 0.0

class Particle:
    """A class that represents a particle."""
    def __init__(self, x: float, y: float, angle: float):
        self.x = x
        self.y = y
        self.angle = angle

class VicsekModel:
    """A class that represents the Vicsek model."""
    # Determines whether a radius or a fixed number of neighbors should be used for calculation.
    # Radius: Dynamic number of neighbors, fixed radius.
    # Fixed: Fixed number of neighbors, dynamic radius.
    modes = {"radius": 0, "fixed": 1}
    
    def __init__(self, N: int, L: float, v: float, noise: float, r: float, mode: int = 0):
        self.N = N
        self.L = L
        self.v = v
        self.noise = noise
        self.r = r
        self.mode = mode
        self.density = N / L**2
        self.particles = [Particle(np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, 2*np.pi)) for _ in range(N)]
        self.num_cells = int(L / r)
        self.cells = [[[] for _ in range(self.num_cells)] for _ in range(self.num_cells)]

    def update_cells(self):
        """Updates the cells."""
        self.cells = [[[] for _ in range(self.num_cells)] for _ in range(self.num_cells)]
        for i, particle in enumerate(self.particles):
            cell_x = int(particle.x / self.r)
            cell_y = int(particle.y / self.r)
            self.cells[cell_x][cell_y].append(i)

    def update(self):
        """Updates the model."""
        new_particles = []
        self.update_cells()
        for i, particle in enumerate(self.particles):
            cell_x = int(particle.x / self.r)
            cell_y = int(particle.y / self.r)
            neighbours = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbour_cell_x = (cell_x + dx) % self.num_cells
                    neighbour_cell_y = (cell_y + dy) % self.num_cells
                    for j in self.cells[neighbour_cell_x][neighbour_cell_y]:
                        if i != j and np.hypot((particle.x - self.particles[j].x) - self.L * round((particle.x - self.particles[j].x) / self.L), 
                                               (particle.y - self.particles[j].y) - self.L * round((particle.y - self.particles[j].y) / self.L)) < self.r:
                            neighbours.append(self.particles[j])
            if neighbours:  # check if the list is not empty
                neighbours.append(particle)
                avg_angle = np.arctan2(np.mean([np.sin(p.angle) for p in neighbours]), np.mean([np.cos(p.angle) for p in neighbours]))
            else:
                avg_angle = particle.angle  # if no neighbours, keep the current direction
            new_angle = avg_angle + np.random.uniform(-self.noise/2, self.noise/2)
            new_x = (particle.x + self.v * np.cos(new_angle)) % self.L
            new_y = (particle.y + self.v * np.sin(new_angle)) % self.L
            new_particles.append(Particle(new_x, new_y, new_angle))
        self.particles = new_particles

    def va(self) -> float:
        """Calculates the order parameter."""
        return np.hypot(np.mean([np.cos(p.angle) for p in self.particles]), np.mean([np.sin(p.angle) for p in self.particles]))

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
    
    model.update()

    ax2.clear()
    ax2.set_xlim(0, i+1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Order parameter')
    va_values.append(model.va())
    avg_va.add(va_values[-1])
    avg_va_list.append(avg_va.average(i+1))
    ax2.plot(va_values)
    ax2.plot(avg_va_list, color="Green")

def update_noise(val):
    """Updates the noise value in the model."""
    model.noise = val

if __name__ == "__main__":
    # Flags
    calcOnly = False

    # Effectively the time steps t
    num_frames = 2000

    # Initialize Model
    settings = {
        "a": [300, 7, 0.03, 2.0, 1, 4],
        "b": [300, 25, 0.03, 0.1, 1, 1],
        "d": [300, 5, 0.03, 0.1, 1, 4],
        "plot1_N40": [40, 3.1, 0.03, 0.1, 1, 4]
    }
    
    N, L, v, noise, r, scale = settings["b"]
    va_values = []
    avg_va_list = []
    avg_va = RunningAverage()

    model = VicsekModel(N, L, v, noise, r)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 10))
    ax1.set_aspect('equal')
    plt.subplots_adjust(bottom=0.25)

    # Add slider for the noise
    ax_noise = plt.axes([0.25, 0.15, 0.65, 0.03])

    slider_noise = Slider(ax_noise, 'Noise', 0.01, 5.0, valinit=noise)
    slider_noise.on_changed(update_noise)

    # Animate function
    ax2.legend(["Current order parameter", "Overall average order parameter"])
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=1)
    plt.show()
    print(avg_va_list[-1])
