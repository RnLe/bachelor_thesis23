import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import customtkinter as ctk
from customtkinter import *

# ctk.set_appearance_mode("dark")

window = Tk()
window.geometry("1200x800")
window.option_add("*Font", "helvetica")

# Create frames
left_frame = Frame(window, width=600, height=800)
right_frame = Frame(window, width=600, height=800)

left_frame.grid(row=0, column=0, padx=10, pady=5)
right_frame.grid(row=0, column=1, padx=10, pady=5)


# Labels
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
lbl1 = Label(left_frame, text=f'Model')
lbl2 = Label(left_frame, text=f'Action')
lbl3 = Label(left_frame, text=f'Mode')
lbl4 = Label(left_frame, text=f'Dimension')

lbl5 = Label(left_frame, text=f'Radius r')
lbl6 = Label(left_frame, text=f'Timesteps t')
lbl7 = Label(left_frame, text=f'Number Particles N')
lbl8 = Label(left_frame, text=f'Noise n')
lbl9 = Label(left_frame, text=f'System Size L')
lbl10 = Label(left_frame, text=f'Speed v')
lbl11 = Label(left_frame, text=f'Neighbors k')

lbl1.grid(row=0, column=0, padx=5, pady=5)
lbl2.grid(row=1, column=0, padx=5, pady=5)
lbl3.grid(row=2, column=0, padx=5, pady=5)
lbl4.grid(row=3, column=0, padx=5, pady=5)

lbl5.grid(row=4, column=0, padx=5, pady=5)
lbl6.grid(row=4, column=2, padx=5, pady=5)
lbl7.grid(row=5, column=0, padx=5, pady=5)
lbl8.grid(row=5, column=2, padx=5, pady=5)
lbl9.grid(row=6, column=0, padx=5, pady=5)
lbl10.grid(row=6, column=2, padx=5, pady=5)
# lbl11.grid(row=4, column=0, padx=5, pady=5)
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Radio buttons
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
radio_var = IntVar()
radio_buttons = {}  # Is this necessary? Maybe later

rb1 = Radiobutton(left_frame, text=f'Vicsek', variable=radio_var, value=1)
rb2 = Radiobutton(left_frame, text=f'Neural Network', variable=radio_var, value=2)
rb3 = Radiobutton(left_frame, text=f'Generate', variable=radio_var, value=3)
rb4 = Radiobutton(left_frame, text=f'Visualize', variable=radio_var, value=4)
rb5 = Radiobutton(left_frame, text=f'Radius', variable=radio_var, value=5)
rb6 = Radiobutton(left_frame, text=f'k Neighbors', variable=radio_var, value=6)
rb7 = Radiobutton(left_frame, text=f'2D', variable=radio_var, value=7)
rb8 = Radiobutton(left_frame, text=f'3D', variable=radio_var, value=8)

rb1.grid(row=0, column=1, padx=5, pady=5)
rb2.grid(row=0, column=2, padx=5, pady=5)
rb3.grid(row=1, column=1, padx=5, pady=5)
rb4.grid(row=1, column=2, padx=5, pady=5)
rb5.grid(row=2, column=1, padx=5, pady=5)
rb6.grid(row=2, column=2, padx=5, pady=5)
rb7.grid(row=3, column=1, padx=5, pady=5)
rb8.grid(row=3, column=2, padx=5, pady=5)

radio_buttons["Vicsek"] = rb1
radio_buttons["Neural Network"] = rb2
radio_buttons["Generate"] = rb3
radio_buttons["Visualize"] = rb4
radio_buttons["Radius"] = rb5
radio_buttons["k Neighbors"] = rb6
radio_buttons["2D"] = rb7
radio_buttons["3D"] = rb8
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Spinboxes
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
spinboxes = {}

spn1 = tk.Spinbox(left_frame, from_=0, to=10000, increment=0.1)
spn2 = tk.Spinbox(left_frame, from_=0, to=100000, increment=100)
spn3 = tk.Spinbox(left_frame, from_=0, to=100000, increment=100)
spn4 = tk.Spinbox(left_frame, from_=0, to=6.8, increment=0.1)
spn5 = tk.Spinbox(left_frame, from_=0, to=1000, increment=1)
spn6 = tk.Spinbox(left_frame, from_=0, to=1, increment=0.01)

spn1.grid(row=4, column=1, padx=5, pady=5)
spn2.grid(row=4, column=3, padx=5, pady=5)
spn3.grid(row=5, column=1, padx=5, pady=5)
spn4.grid(row=5, column=3, padx=5, pady=5)
spn5.grid(row=6, column=1, padx=5, pady=5)
spn6.grid(row=6, column=3, padx=5, pady=5)

spinboxes["radius"] = spn1
spinboxes["timesteps"] = spn2
spinboxes["number particles"] = spn3
spinboxes["noise"] = spn4
spinboxes["system size"] = spn5
spinboxes["speed"] = spn6

# Set default values
spn1.delete(0, 'end')
spn1.insert(0, 1)
spn2.delete(0, 'end')
spn2.insert(0, 1000)
spn3.delete(0, 'end')
spn3.insert(0, 2000)
spn4.delete(0, 'end')
spn4.insert(0, 0.1)
spn5.delete(0, 'end')
spn5.insert(0, 60)
spn6.delete(0, 'end')
spn6.insert(0, 0.03)
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



window.mainloop()