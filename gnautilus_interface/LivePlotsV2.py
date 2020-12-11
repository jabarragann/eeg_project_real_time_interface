import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from tkinter import  Label
import matplotlib.animation as animation
import time
import numpy as np

class LivePlotLogic:
    def __init__(self, controller, axe = None, line = None, maxt=10, dt=0.02, workload_label = None):
        #Controller reference will allow to access to the data queue of the producer
        self.controller = controller

        #Artist and axe that is going to be updated
        self.prediction_line = line[0]
        self.smoothed_line = line[1]
        self.threshold_line = line[2]

        self.ax = axe
        self.workload_label = workload_label

        #Logic variables
        self.dt = dt
        self.maxt = maxt
        self.tdata  = [0]
        self.ydata  = [0]
        self.ydata2 = [0]

        size = 80
        self.y1 = np.zeros(size)
        self.y2 = np.zeros(size)
        self.y3 = np.ones(size) * self.controller.detection_threshold
        self.x1 = np.linspace(-10.0, 0.0, size)

        self.init_time = time.time()
        self.init_time2 = time.time()
        self.animation_init_time = time.time()

        self.total_time = 0

        self.start_removing = False

    def update_animation(self, frame):

        isThereNewData = frame[3]
        if isThereNewData:
            #Print update rate
            d = time.time() - self.animation_init_time
            print("Update rate: {:08.4f}ms".format(d * 1000))
            self.animation_init_time = time.time()

            # Get Last Sample
            lastt = self.tdata[-1]

            # Calculate current time
            self.dt = time.time() - self.init_time
            self.total_time += self.dt
            # print(self.dt)

            # Get new sample
            new_y1 = frame[1]
            new_y2 = frame[2]

            self.y1 = np.roll(self.y1, -1)
            self.y2 = np.roll(self.y2, -1)
            self.y1[-1] = new_y1
            self.y2[-1] = new_y2

            # Get and update label color
            new_color = self.calculate_new_color(new_y2)
            self.workload_label.configure(bg=new_color)
            self.workload_label.update()

            # Update artists data
            self.prediction_line.set_data(self.x1, self.y1)
            self.smoothed_line.set_data(self.x1, self.y2)

            # Reset init time
            self.init_time = time.time()

        else:
            pass

        return self.prediction_line, self.smoothed_line

    def init_animation(self):

        self.ax.set_xlim(self.x1[0],self.x1[-1])

        self.init_time = time.time()
        self.animation_init_time = time.time()
        self.prediction_line.set_data(self.x1,self.y1)
        self.smoothed_line.set_data(self.x1,self.y2)
        self.threshold_line.set_data(self.x1,self.y3)
        return self.prediction_line, self.smoothed_line

    def check_data_queue(self):
        # Check producer data queue
        while True:
            if self.controller.data_queue.empty():
                yield [0, 0, 0, False]
            else:
                data = self.controller.data_queue.get_nowait()
                yield [data['time'], data['data'], data['smooth'], True]

    @staticmethod
    def calculate_new_color(smooth_value):
        red_val = hex(int(smooth_value * 4094))[2:].zfill(3)
        green_val = hex(int((1 - smooth_value) * 4094))[2:].zfill(3)

        new_color = "#{:}{:}000".format(red_val, green_val)

        return new_color

class LivePlots:
    def __init__(self, master, controller):
        #Font
        LARGE_FONT = ("Verdana", 12)
        self.x_max_lim = 20

        #Set root component
        self.master = master
        self.master.title("Live Plot of Power bands")

        #Set label
        self.label1 = Label(self.master, text="Workload detector", font=LARGE_FONT, width=30, height=5)
        self.label1.pack(pady=10, padx=10)
        self.label1.config(bg="#fff000000")

        # #Set init button
        # self.button1 = Button(self.master, text="Start Animation", command=self.stop_start_animation)
        # self.button1.pack()

        #Create matplotlib graph
        self.figure, self.ax = plt.subplots(1,1, figsize=(6, 5), dpi=100)

        self.graph = FigureCanvasTkAgg(self.figure, self.master)
        self.graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
        self.configure_plot()

        #Create data line
        self.smoothed_line = Line2D( [],[], marker='o',markersize=4,linestyle='-', linewidth=1.0, label='Prediction', color='black')
        self.prediction_line = Line2D([],[], marker='*', markersize=3, linestyle='-', linewidth=0.5, label='Prediction', color='gray')
        self.threshold_line = Line2D([],[],linestyle='--', linewidth=1.5, label='Prediction', color='blue')


        self.ax.add_line(self.prediction_line)
        self.ax.add_line(self.smoothed_line)
        self.ax.add_line(self.threshold_line)

        #Create animation logic and animation
        self.logic = LivePlotLogic(controller, axe = self.ax, line = [self.prediction_line, self.smoothed_line, self.threshold_line],
                                   maxt = self.x_max_lim, workload_label = self.label1)

        self.ani = animation.FuncAnimation(self.figure,
                                           func=self.logic.update_animation,
                                           frames=self.logic.check_data_queue,
                                           init_func=self.logic.init_animation,
                                           interval=10, blit=True)

        self.animationRunning = True

    def stop_start_animation(self):
        if self.animationRunning:
            self.ani.event_source.stop()
            self.animationRunning = False
        else:
            self.ani.event_source.start()
            self.animationRunning = True

    def configure_plot(self):
        # Plot configuration
        self.ax.set_title('Live Plot')
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.set_xlim(0, self.x_max_lim)

        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.set_xticks([-10,-8.0,-6.0,-4.0,-2.0, 0.0])
        self.ax.set_xticklabels([-10,-8.0,-6.0,-4.0,-2.0, 0.0])
        self.ax.set_yticks([0.0,0.5,1.0])
        self.ax.set_yticklabels([0.0,0.5,1.0])
        self.ax.set_ylabel("Workload Index")
        self.ax.set_xlabel("Time (s)")
