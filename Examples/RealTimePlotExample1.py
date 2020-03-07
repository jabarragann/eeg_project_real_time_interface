import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from tkinter import Tk, Label, Button
import numpy as np
import matplotlib.animation as animation

class LivePlots:
    def __init__(self, master):
        #Font
        LARGE_FONT = ("Verdana", 12)

        #Set root component
        self.master = master
        self.master.title("Live Plot of Power bands")

        #Set label
        self.label1 = Label(self.master, text="Workload detector", font=LARGE_FONT)
        self.label1.pack(pady=10, padx=10)

        #Set init button
        self.button1 = Button(self.master, text="Start Animation", command=self.stop_start_animation)
        self.button1.pack()

        #Create matplotlib graph
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.graph = FigureCanvasTkAgg(self.figure, self.master)
        self.graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
        self.ax.set_title('Live Plot')
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, 4)

        #Create data line
        self.line = Line2D([0,1,2,3,4,5], [1,2,4,5,6,8])
        self.line.set_linestyle('--')
        self.line.set_marker('*')
        self.line.set_markersize(2)
        self.line.set_color('red')

        self.ax.add_line(self.line)

        #Create animation
        self.ani = animation.FuncAnimation(self.figure,
                                           func=self.update_animation,
                                           frames=400,
                                           init_func=self.init_animation,
                                           interval=10, blit=True)

        self.animationRunning = True

    def stop_start_animation(self):
        if self.animationRunning:
            self.ani.event_source.stop()
            self.animationRunning = False
        else:
            self.ani.event_source.start()
            self.animationRunning = True
            
    def init_animation(self):
        self.line.set_data([], [])
        return self.line,

    def update_animation(self,i):
        # print(i)
        x = np.linspace(0, 4, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        self.line.set_data(x, y)
        return self.line,

root = Tk()
my_gui = LivePlots(root)
root.mainloop()