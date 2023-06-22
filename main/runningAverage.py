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