import numpy as np
from particle import Particle
from perceptron import Perceptron
from tqdm import tqdm


class SwarmModel:
    # Determines whether a radius or a fixed number of neighbors should be used for calculation.
    # Radius: Dynamic number of neighbors, fixed radius.
    # Fixed: Fixed number of neighbors, dynamic radius.
    modes = {"radius": 0, "fixed": 1}
    
    def __init__(self, N: int, L: float, v: float, noise: float, r: float, mode: int = modes["radius"], k_neighbors: int = 5, cellSpan: int = 3):
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
    
    def get_density_hist(self):
        densities = np.zeros(shape=self.num_cells ** 2)
        i = 0
        for cell_row in self.cells:
            for cell in cell_row:
                densities[i] = len(cell) / self.N
                i += 1
        return densities
        
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
        boundary = 0
        while len(neighbors) < self.k_neighbors:
            for dx in range(-boundary, boundary + 1):
                for dy in range(-boundary, boundary + 1):
                    # Skip the cells that are not on the current boundary
                    if abs(dx) != boundary and abs(dy) != boundary:
                        continue
                    
                    neighbor_cell_x = (cell_x + dx) % self.num_cells
                    neighbor_cell_y = (cell_y + dy) % self.num_cells
                    
                    for j in self.cells[neighbor_cell_x][neighbor_cell_y]:
                        
                        if index != j:
                
                            distance = ((particle.x - self.particles[j].x) - self.L * round((particle.x - self.particles[j].x) / self.L))**2 + \
                                                ((particle.y - self.particles[j].y) - self.L * round((particle.y - self.particles[j].y) / self.L))**2
                            
                            neighbors.append(self.particles[j])
                            distances.append(distance)
            # Leave the loop depending on the mode
            if self.mode == self.modes["radius"] and boundary == 1: break
            boundary += 1
            
        particle.cellRange = boundary - 1
                        
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
  
    def get_dynamic_radius(self):
        """Returns an N dimensional array with the effective radius for each particle.

        Returns:
            np.ndarray: Array of effective radii.
        """  
        effective_radii = np.zeros(shape=self.N)   
        for i, p in enumerate(self.particles):
            effective_radii[i] = p.cellRange
        
        return effective_radii
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

    def writeToFile(self, timesteps: int, filetype: str):
        if filetype == "xyz":
            with open("particles.xyz", "w") as file:
                for _ in tqdm(range(timesteps), desc="Writing to file"):
                    self.update()
                    file.write(f"{len(self.particles)}\n")
                    file.write("\n")
                    for particle in self.particles:
                        file.write(f"C {particle.x} {particle.y} 0\n")
        
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
            
            # new_particles.append(Particle(new_x, new_y, new_angle, neighbors, particle.cellRange))    # doesn't work
            particle.x = new_x
            particle.y = new_y
            particle.angle = new_angle
            particle.k_neighbors = neighbors
            
        # self.particles = new_particles
    
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
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

class PerceptronModel(SwarmModel):
    """A class that represents the Vicsek model."""
    modes = {"radius": 0, "fixed": 1}
    learning_mode = {"uniform": 0, "imitateVicsek": 1, "maximizeOrder": 2}
    
    def __init__(self, N: int, L: float, v: float, noise: float, r: float, mode: int = modes["fixed"], k_neighbors: int = 5, learning_rate: int = 0.000001, weights: list = []):
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
            new_angle = (self.perceptron.forward(input_vec)  + np.random.uniform(-self.noise / 2, self.noise / 2)) % (2 * np.pi)

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
        error = (abs(target - np.pi) - abs(prediction - np.pi)) % (2*np.pi)
        error /= len(target)
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
        return self.perceptron.forward(input_vec)