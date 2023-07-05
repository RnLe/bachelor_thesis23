from Solver import Particle, Perceptron, PerceptronModel, VicsekModel, NeuralNetwork, PerceptronMode, Mode

import numpy as np
import sys
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5 import QtGui as QtGui5
from pyqtgraph import mkPen, mkBrush, TextItem, PlotDataItem, PlotCurveItem
from PyQt5.QtWidgets import QDockWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QWidget

import cProfile

# Start profiling
pr = cProfile.Profile()
pr.enable()

# Initialise model
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
stop = False           # Stop simulation

# Duration of simulation

timesteps = 5000

# Choose settings
chosen_settings = settings["a"]
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



# Create GUI
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::
app = pg.mkQApp("Plotting Example")

win = pg.GraphicsLayoutWidget(show=True, title="Particle Animation")
win.resize(1000,1000)
# White background
win.setBackground('w')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title="Updating plot")


# Create a dockable widget
dock = QDockWidget("Controls")

# Create a container widget to hold our widgets
container = QWidget()

# Create a layout to hold our widgets
layout = QVBoxLayout()

# Create some control widgets
stop_button = QPushButton('Stop')
reset_button = QPushButton('Reset')

# Add widgets to the layout and set the layout on the container widget 
layout.addWidget(stop_button)
layout.addWidget(reset_button)

# Set the layout on the container widget
container.setLayout(layout)

# Set the container widget on the dock widget
dock.setWidget(container)


def stop_simulation():
    global stop
    stop = not stop
    if stop:
        stop_button.setText('Continue')
    else:
        stop_button.setText('Stop')
        
def reset_simulation():
    global model
    model = VicsekModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=True)
    if not stop: stop_simulation()
    
# Connect widgets to functions
stop_button.clicked.connect(stop_simulation)
reset_button.clicked.connect(reset_simulation)


    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Initialise the curve and draw data points as (dark grey) dots
curve = p1.plot(pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(66, 135, 245, 255))

# Initialise the arrows
arrows = []
tailLength = 5

# Set fixed axis limits (0 to L)
p1.setXRange(0, L)
p1.setYRange(0, L)

# Set x and y ratio to 1:1
#p1.setAspectLocked(True)

# Create a label item for displaying FPS
fps_label = pg.LabelItem(justify='right')
win.addItem(fps_label)

# Initialize time variable
t1 = time.time()

def update():
    
    if stop:
        return
    
    global t1
    
    # Update the model
    model.update()
    
    # Get new particle positions and angles
    x = [particle.x for particle in model.particles]
    y = [particle.y for particle in model.particles]
    angle = [particle.angle for particle in model.particles]

    # Update the scatter plot (draw particles as dots)
    curve.setData(x, y)

    # Remove old arrows
    # for arrow in arrows:
    #     p1.removeItem(arrow)
    # arrows.clear()
        
    # Draw arrows from angles
    # Calculate the end points of the arrows
    # for i in range(len(x)):
    #     arrow = pg.ArrowItem(pos=(x[i] + np.cos(angle[i]) * tailLength/15, y[i] + np.sin(angle[i]) * tailLength/15), angle=np.rad2deg(np.pi - angle[i]),
    #                          tipAngle=30, baseAngle=0, headLen=5, tailLen=5, tailWidth=0.1, pen=pg.mkPen((50, 50, 50), width=1), brush=pg.mkBrush((50, 50, 50)))
    #     p1.addItem(arrow)
    #     arrows.append(arrow)
    
    # Remove previous cell rectangles and text items if they exist
    if hasattr(p1, 'cell_rects'):
        for rect in p1.cell_rects:
            p1.removeItem(rect)
    if hasattr(p1, 'cell_texts'):
        for text in p1.cell_texts:
            p1.removeItem(text)

    p1.cell_rects = []
    p1.cell_texts = []

    pen = mkPen(color='grey')
    brush = mkBrush(color=(0, 0, 255, 25))  # Blue color with 10% opacity

    # Draw all cells
    for cell_x in range(model.num_cells):
        for cell_y in range(model.num_cells):
            # Create rectangle
            rect = PlotCurveItem(pen=pen, brush=brush)
            rect.setData([cell_x * (2 * model.r), cell_x * (2 * model.r) + 2 * model.r, cell_x * (2 * model.r) + 2 * model.r, cell_x * (2 * model.r)],
                        [cell_y * (2 * model.r), cell_y * (2 * model.r), cell_y * (2 * model.r) + 2 * model.r, cell_y * (2 * model.r) + 2 * model.r])
            
            # Count the number of particles in the cell
            cells2D = model.cells2D
            num_particles = len(cells2D[cell_x][cell_y])
            
            # Write number of particles in cell
            text = TextItem(text=str(num_particles), anchor=(0.5, 0.5))
            text.setPos(cell_x * (2 * model.r) + model.r, cell_y * (2 * model.r) + model.r)

            p1.addItem(rect)
            p1.addItem(text)

            p1.cell_rects.append(rect)
            p1.cell_texts.append(text)
        
    # Calculate and display FPS
    t2 = time.time()
    fps = 1.0 / (t2-t1)
    t1 = t2
    fps_label.setText("FPS: %0.2f" % fps)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

if __name__ == '__main__':
    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()
        
    pr.disable()
    pr.dump_stats("my_program.prof")