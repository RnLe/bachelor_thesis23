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