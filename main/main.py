import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from matplotlib.widgets import Button
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
    def __init__(self, x: float, y: float, angle: float, k_neighbors: list = None):
        self.x = x
        self.y = y
        self.angle = angle
        self.k_neighbors = k_neighbors if k_neighbors is not None else []
        

class Perceptron:
    def __init__(self, input_dim: int, weights = [], lambda_reg: float = 0.00001):
        """
        Initialize the perceptron.

        Args:
            input_dim (int): The dimension of the input vector.
        """
        # Initialize the weights to small random values
        if len(weights) == 0:
            self.weights = np.random.randn(input_dim) * 0.0001
        else:
            self.weights = weights
            
        # Regularization parameter
        self.lambda_reg = lambda_reg

    def forward(self, input_vec: ArrayLike):
        """
        Perform the forward pass of the perceptron.

        Args:
            input_vec (np.array): The input vector.

        Returns:
            float: The output of the perceptron.
        """
        # Compute the weighted sum of the inputs
        weighted_sum = np.dot(self.weights, input_vec)

        # Apply the ReLU activation function
        output = weighted_sum if weighted_sum > 0 else weighted_sum / 1000

        return output

    def update_weights(self, input_vec: ArrayLike, error: float, learning_rate: float):
        """
        Update the weights of the perceptron using gradient descent.

        Args:
            input_vec (np.array): The input vector.
            error (float): The error of the perceptron's output.
            learning_rate (float): The learning rate for gradient descent.
        """
        # Compute the gradient of the error with respect to the weights
        gradient = error * input_vec
        
        # Add the regularization term to the gradient
        # gradient += self.lambda_reg * self.weights

        # Update the weights using gradient descent
        self.weights -= learning_rate * gradient
        
                
class SwarmModel:
    # Determines whether a radius or a fixed number of neighbors should be used for calculation.
    # Radius: Dynamic number of neighbors, fixed radius.
    # Fixed: Fixed number of neighbors, dynamic radius.
    modes = {"radius": 0, "fixed": 1}
    
    def __init__(self, N: int, L: float, v: float, noise: float, r: float, mode: int = modes["radius"], k_neighbors: int = 5, cellSpan: int = 5):
        self.N = N
        self.L = L
        self.v = v
        self.noise = noise
        self.r = r
        self.mode = mode
        self.k_neighbors = k_neighbors
        self.density = N / L**2
        self.particles = [Particle(np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, 2*np.pi)) for _ in range(N)]
        self.num_cells = int(L / r)
        self.cells = [[[] for _ in range(self.num_cells)] for _ in range(self.num_cells)]
        self.cellSpan = cellSpan
        self.mode1_cells = list(range(-cellSpan, cellSpan + 1))
        
    def update_cells(self):
        """Updates the cells."""
        self.cells = [[[] for _ in range(self.num_cells)] for _ in range(self.num_cells)]
        for i, particle in enumerate(self.particles):
            cell_x = int(particle.x / self.r)
            cell_y = int(particle.y / self.r)
            self.cells[cell_x][cell_y].append(i)
            
    def get_fluctuations(self):
        """Function to calculate the standard deviation of the cell densities.
        """
        mean = self.N / self.num_cells
        sum = 0
        for cell_row in self.cells:
            for cell in cell_row:
                sum += (len(cell) - mean)**2
        # Normalize the variance
        max_variance = (self.num_cells - 1) * mean**2 + (self.N - mean)**2
        max_variance /= 200
        return (sum / self.N) / max_variance    
    
    def get_neighbors(self, particle: Particle, index: int):
        """Collects all neighbors, calculates the distances and returns both lists, which resemble respective pairs.
        (e.g.:`distances[i]` maps to `neighbors[i]`)
        ----------------
        ### Args:
            `particle` (Particle): The particle whose neighbors are to be determined.
            
            `cellSpan` (int): The number of neighboring cells which have to be considered. This parameter needs to be adjusted for each observation/configuration.
            
            `index` (int): Index of the particle in the overall particles-array.

        ### Returns:
            _type_: `None`
        """
        cell_x = int(particle.x / self.r)
        cell_y = int(particle.y / self.r)
        neighbors = []
        distances = []
        self.mode1_cells = list(range(-self.cellSpan, self.cellSpan + 1))
        # mode1_cells = list(range(-self.num_cells, self.num_cells + 1))    # takes forever. No real-time.
        for dx in [-1, 0, 1] if self.mode == 0 else self.mode1_cells:
            for dy in [-1, 0, 1] if self.mode == 0 else self.mode1_cells:
                neighbor_cell_x = (cell_x + dx) % self.num_cells
                neighbor_cell_y = (cell_y + dy) % self.num_cells
                for j in self.cells[neighbor_cell_x][neighbor_cell_y]:
                    
                    if index != j:
                        # distance = np.hypot((particle.x - self.particles[j].x) - self.L * round((particle.x - self.particles[j].x) / self.L), 
                        #                 (particle.y - self.particles[j].y) - self.L * round((particle.y - self.particles[j].y) / self.L))
                        
                        distance = ((particle.x - self.particles[j].x) - self.L * round((particle.x - self.particles[j].x) / self.L))**2 + \
                                            ((particle.y - self.particles[j].y) - self.L * round((particle.y - self.particles[j].y) / self.L))**2

                        
                        neighbors.append(self.particles[j])
                        distances.append(distance)
                        
        # Append the particle itself
        neighbors.append(particle)
        distances.append(0)
        
        if len(neighbors) > 1:  # Check if there are any other neighbors
            
            # Sort the neighbors and distances based on distances
            sorted_neighbors, sorted_distances = zip(*sorted(zip(neighbors, distances), key=lambda x: x[1]))
            
            # If there is only a fixed number of neighbors, cut the list down to the k nearest
            if self.mode == 1:
                # Select the k nearest neighbors
                neighbors = list(sorted_neighbors[:self.k_neighbors + 1])
            elif self.mode == 0:
                # Find the index of the first distance that is greater or equal to 1
                cut_off = next((index for index, value in enumerate(sorted_distances) if value >= 1), None)
                # If such an index is found, cut the list down to this index
                if cut_off is not None:
                    neighbors = list(sorted_neighbors[:cut_off])
        
        return neighbors, distances
    
    # This method is virtual and characterizes the model which is implemented
    def update(self):
        """Updates the model."""
        new_particles = []
        
        # Dummy
        # Insert logic here
        new_particles = self.particles
        # Insert logic here
        
        self.particles = new_particles
    
    def va(self) -> float:
        """Calculates the order parameter."""
        return np.hypot(np.mean([np.cos(p.angle) for p in self.particles]), np.mean([np.sin(p.angle) for p in self.particles]))   
        
    
class VicsekModel(SwarmModel):
    """A class that represents the Vicsek model."""
    modes = {"radius": 0, "fixed": 1}
    
    def __init__(self, N: int, L: float, v: float, noise: float, r: float, mode: int = modes["radius"], k_neighbors: int = 5):
        super().__init__(N, L, v, noise, r, mode, k_neighbors)  # Call the constructor of the base class
        
    def update(self):
        """Updates the model."""
        new_particles = []
        self.update_cells()
        
        for i, particle in enumerate(self.particles):
            # Get the all neighbors as lists
            neighbors, distances = self.get_neighbors(particle, i)
            
            new_x, new_y, new_angle = self.get_new_particle_vicsek(particle, neighbors)
            
            new_particles.append(Particle(new_x, new_y, new_angle, neighbors))
            
        self.particles = new_particles
    
    def get_new_particle_vicsek(self, particle: Particle, neighbors: list[Particle]):
        avg_angle = np.arctan2(np.mean([np.sin(p.angle) for p in neighbors]), np.mean([np.cos(p.angle) for p in neighbors]))
            
        # Convert the average angle to the range 0 to 2pi
        if avg_angle < 0:
            avg_angle += 2 * np.pi

        new_angle = avg_angle + np.random.uniform(-self.noise / 2, self.noise / 2)
            
        # Ensure the new angle is in the range 0 to 2pi
        new_angle = new_angle % (2 * np.pi)
            
        new_x = (particle.x + self.v * np.cos(new_angle)) % self.L
        new_y = (particle.y + self.v * np.sin(new_angle)) % self.L
        
        return new_x, new_y, new_angle


class PerceptronModel(SwarmModel):
    """A class that represents the Vicsek model."""
    modes = {"radius": 0, "fixed": 1}
    learning_mode = {"uniform": 0, "imitateVicsek": 1, "maximizeOrder": 2}
    
    def __init__(self, N: int, L: float, v: float, noise: float, r: float, mode: int = modes["fixed"], k_neighbors: int = 5, learning_rate: int = 0.00001, weights = []):
        # Call the constructor of the base class
        super().__init__(N, L, v, noise, r, mode, k_neighbors)
        
        self.learning_rate = learning_rate
        if (len(weights) == 0):
            self.perceptron = Perceptron(k_neighbors + 1)
        else:
            self.perceptron = Perceptron(k_neighbors + 1, weights)
        
        
    def learn(self):
        """Trains the model."""
        new_particles = []
        self.update_cells()

        for i, particle in enumerate(self.particles):
            # Get the all neighbors as lists
            neighbors, distances = self.get_neighbors(particle, i)

            # Convert the neighbors and distances to input vectors for the perceptron
            input_vec = self.neighbors_to_input_vec(neighbors, distances)

            # Compute the error of the perceptron's output
            error = self.compute_error(particle, neighbors, input_vec)

            # Update the weights of the perceptron based on the input vector and error
            self.perceptron.update_weights(input_vec, error, self.learning_rate)

            # Compute the new angle of the particle based on the updated weights of the perceptron
            new_angle = self.perceptron.forward(input_vec)

            new_x = (particle.x + self.v * np.cos(new_angle)) % self.L
            new_y = (particle.y + self.v * np.sin(new_angle)) % self.L

            new_particles.append(Particle(new_x, new_y, new_angle, neighbors))

        self.particles = new_particles
        
    def update(self):
        new_particles = []
        self.update_cells()
        
        for i, particle in enumerate(self.particles):
            # Get the all neighbors as lists
            neighbors, distances = self.get_neighbors(particle, i)

            # Convert the neighbors and distances to input vectors for the perceptron
            input_vec = self.neighbors_to_input_vec(neighbors, distances)
            
            # Compute the new angle of the particle based on the weights
            new_angle = self.perceptron.forward(input_vec)

            new_x = (particle.x + self.v * np.cos(new_angle)) % self.L
            new_y = (particle.y + self.v * np.sin(new_angle)) % self.L

            new_particles.append(Particle(new_x, new_y, new_angle, neighbors))

        self.particles = new_particles

    def neighbors_to_input_vec(self, neighbors: list[Particle], distances: list[float]):
        """Generates the input vector for the neurons from the neighbor list.
        
        ----------------
        ### Args:
            neighbors (list): A list of the nearest neighbors, sorted by distance.
            distances (list): Actual distances of the particles. Shares the same index as `neighbors`.

        ### Returns:
            np.darray: None
        """
        input_vec = []
        for p in neighbors:
            input_vec.append(p.angle)
        
        input_vec = np.array(input_vec)
        #input_vec_norm = input_vec / np.linalg.norm(input_vec)
            
        return input_vec
        
    def compute_error(self, particle: Particle, neighbors: list[Particle], input_vec: list):
        """Resembles the loss function.

        Args:
            particle (Particle):
            neighbors (list[Particle]): A list of the nearest neighbors, sorted by distance.

        Returns:
            float: 
        """
        target = self.get_target(neighbors)
        prediction = self.get_prediction(input_vec)
        
        # Mean Squared Error, MSE        
        error = (target - prediction) / len(target)
        return np.sum(error ** 2)
    
    def get_target(self, neighbors: list[Particle]):
        """Generates a target vector, the same shape and unit as the prediction vector.

        Args:
            particle (Particle): _description_
            neighbors (list[Particle]): _description_

        Returns:
            _type_: _description_
        """        
        target = []
        for p in neighbors:
            new_x, new_y, new_angle = VicsekModel.get_new_particle_vicsek(self, p, neighbors)
            target.append(new_angle)
            
        return np.array(target)
    
    def get_prediction(self, input_vec: list):
        return self.perceptron.weights * input_vec

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
        
        for dx in [-1, 0, 1] if model.mode == 0 else model.mode1_cells:
            for dy in [-1, 0, 1] if model.mode == 0 else model.mode1_cells:
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
    fluctuations.append(model.get_fluctuations())
    line_fluctuations.set_data(range(len(fluctuations)), fluctuations)
    line_va.set_data(range(len(va_values)), va_values)
    line_avg_va.set_data(range(len(avg_va_list)), avg_va_list)

def update_noise(val):
    """Updates the noise value in the model."""
    model.noise = val
    
def update_neighbors(val):
    """Updates the noise value in the model."""
    model.k_neighbors = val
    
def update_cellSpan(val):
    """Updates the noise value in the model."""
    model.cellSpan = val
    
def toggle_mode(event):
        global mode
        mode = not mode  # Toggle the mode
        model.mode = mode

if __name__ == "__main__":
    # Flags
    calcOnly = False
    use_perceptron_model = True  # Set this to True to use the PerceptronModel, False to use the VicsekModel

    # Effectively the time steps t
    num_frames = 2000

    # Initialize Model
    modes = {"radius": 0, "fixed": 1}
    mode = modes["fixed"]
    
    settings = {
        "a": [300, 7, 0.03, 2.0, 1, 4],
        "b": [300, 25, 0.03, 0.5, 1, 1],
        "d": [300, 5, 0.03, 0.1, 1, 4],
        "plot1_N40": [40, 3.1, 0.03, 0.1, 1, 4]
    }
    N, L, v, noise, r, scale = settings["b"]
    k_neighbors = 5
    cellSpan = 5 if mode == 1 else 1
    va_values = []
    avg_va_list = []
    fluctuations = []
    avg_va = RunningAverage()

    if use_perceptron_model:
        model = PerceptronModel(N, L, v, noise, r, mode=mode, k_neighbors=k_neighbors)
        iterations = 100
        for i in range(iterations):
            print(f"Weights: {model.perceptron.weights}")
            model.learn()
        weights = model.perceptron.weights
        model = PerceptronModel(N, L, v, noise, r, mode=mode, k_neighbors=k_neighbors, weights=weights)
    else:
        model = VicsekModel(N, L, v, noise, r, mode=mode, k_neighbors=k_neighbors)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 12))
    # fig, axes = plt.subplot_mosaic("AB;CC", figsize=(10, 12))
    # ax1, ax3, ax2 = axes['A'], axes['B'], axes['C']
    
    ax1.set_aspect('equal')
    line_fluctuations, = ax2.plot(fluctuations, label="Current density inhomogeneity")
    line_va, = ax2.plot(va_values, label="Current order parameter")
    line_avg_va, = ax2.plot(avg_va_list, label="Overall average order parameter")
    plt.subplots_adjust(bottom=0.25)

    # Sliders
    # Add slider for the noise
    ax_noise = plt.axes([0.1, 0.9, 0.25, 0.03])

    slider_noise = Slider(ax_noise, 'Noise', 0.01, 5.0, valinit=noise)
    slider_noise.on_changed(update_noise)
    
    # Add slider for the neighbors
    ax_neighbors = plt.axes([0.1, 0.85, 0.25, 0.03])

    slider_neighbors = Slider(ax_neighbors, 'Neighbors', 0, 50, valinit=k_neighbors, valstep=1)
    slider_neighbors.on_changed(update_neighbors)
    
    # Add slider for the neighbor list
    ax_cellSpan = plt.axes([0.1, 0.8, 0.25, 0.03])

    slider_cellSpan = Slider(ax_cellSpan, 'CellSpan', 0, 10, valinit=cellSpan, valstep=1)
    slider_cellSpan.on_changed(update_cellSpan)
    
    # Buttons
    button_ax = plt.axes([0.1, 0.75, 0.1, 0.03])
    button = Button(button_ax, 'Toggle Mode', color='lightgoldenrodyellow', hovercolor='0.975')

    button.on_clicked(toggle_mode)

    # Animate function
    ax2.legend()
    plt.tight_layout()
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=1)
    plt.show()
    print(avg_va_list[-1])
