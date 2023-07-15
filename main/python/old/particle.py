class Particle:
    """A class that represents a particle."""
    def __init__(self, x: float, y: float, angle: float, k_neighbors: list = None, cellRange: int = 0):
        """_summary_

        Args:
            x (float): _description_
            y (float): _description_
            angle (float): _description_
            k_neighbors (list, optional): _description_. Defaults to None.
            cellRange (int, optional): The range of the neighboring cells containing a neighbor. Defaults to 0.
        """             
        self.x = x
        self.y = y
        self.angle = angle
        self.k_neighbors = k_neighbors if k_neighbors is not None else []
        self.cellRange = 0