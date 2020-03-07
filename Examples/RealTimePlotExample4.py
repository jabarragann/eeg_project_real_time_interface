import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from tkinter import Tk, Label, Button
import numpy as np
import matplotlib.animation as animation
import time
import queue
import threading
from threading import Thread
import random

class LivePlotLogic:
    def __init__(self, controller, axe = None, line = None, maxt=10, dt=0.02, workload_label=None):
        #Controller reference will allow to access to the data queue of the producer
        self.controller = controller

        #Artist and axe that is going to be updated
        self.animated_line_1 = line[0]
        self.animated_line_2 = line[1]
        self.ax = axe
        self.workload_label = workload_label

        #Logic variables
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.ydata2 = [0]

        self.init_time = time.time()
        self.init_time2 = time.time()
        self.animation_init_time = time.time()

        self.total_time = 0

        self.start_removing = False

    def update_animation(self, frame):

        d = time.time() - self.animation_init_time
        print("Update rate: {:08.4f}ms".format(d*1000))
        self.animation_init_time = time.time()

        isThereNewData = frame[3]
        if isThereNewData:

            # d = time.time() - self.animation_init_time
            # print("Update rate: {:08.4f}ms".format(d*1000))
            # self.animation_init_time = time.time()

            # Get Last Sample
            lastt = self.tdata[-1]

            # Calculate current time
            self.dt = time.time() - self.init_time
            self.total_time += self.dt
            # print(self.dt)

            if lastt > self.tdata[0] + self.maxt:  # reset the arrays
                self.start_removing = True
            if self.start_removing:
                self.tdata = self.tdata[1:]
                self.ydata = self.ydata[1:]
                self.ydata2 = self.ydata2[1:]
                self.ax.set_xlim(self.tdata[0], self.tdata[-1] + self.dt * 10)

            # Get new sample
            new_t =  frame[0]
            new_y =  frame[1]
            new_y2 = frame[2]

            #Get and update label color
            new_color = self.calculate_new_color(new_y2)
            self.workload_label.configure(bg=new_color)
            self.workload_label.update()

            # Update artists data
            self.tdata.append(new_t)
            self.ydata.append(new_y)
            self.ydata2.append(new_y2)
            self.animated_line_1.set_data(self.tdata, self.ydata)
            self.animated_line_2.set_data(self.tdata, self.ydata2)

            # Reset init time
            self.init_time = time.time()

        else:
            pass

        return self.animated_line_1, self.animated_line_2

    def init_animation(self):
        self.init_time = time.time()
        self.animation_init_time = time.time()
        self.animated_line_1.set_data([], [])
        self.animated_line_2.set_data([], [])
        return self.animated_line_1, self.animated_line_2

    def check_data_queue(self):
        #Check producer data queue
        while True:
            if self.controller.data_queue.empty():
                yield [0,0,0,False]
            else:
                data = self.controller.data_queue.get_nowait()
                yield [data['time'], data['data'], data['smooth'], True]


    @staticmethod
    def calculate_new_color(smooth_value):
        red_val = hex(int(smooth_value*4094))[2:].zfill(3)
        green_val = hex(int((1-smooth_value)*4094))[2:].zfill(3)

        new_color = "#{:}{:}000".format(red_val,green_val)

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
        self.label1.pack( pady=10, padx=10)
        self.label1.config(bg="#fff000000")

        # #Set init button
        # self.button1 = Button(self.master, text="Start Animation", command=self.stop_start_animation)
        # self.button1.pack()

        #Create matplotlib graph
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.graph = FigureCanvasTkAgg(self.figure, self.master)
        self.graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
        self.ax.set_title('Live Plot')
        self.ax.set_ylim(-0.3, 1.3)
        self.ax.set_xlim(0, self.x_max_lim)

        #Create data line
        self.line = Line2D([0,1,2,3,4,5], [1,2,4,5,6,8])
        self.line.set_linestyle('--')
        self.line.set_marker('*')
        self.line.set_markersize(2.0)
        self.line.set_color('red')

        self.line2 = Line2D([0, 1, 2, 3, 4, 5], [1, 2, 4, 5, 6, 8])
        self.line2.set_linestyle('--')
        self.line2.set_marker('*')
        self.line2.set_markersize(2.0)
        self.line2.set_color('blue')

        self.ax.add_line(self.line)
        self.ax.add_line(self.line2)

        #Create animation logic and animation
        self.logic = LivePlotLogic(controller, axe = self.ax, line = [self.line, self.line2],
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

class DataProducer(Thread):

    def __init__(self, controller, data_queue):
        super().__init__()
        self.init_time = time.time()
        self.controller = controller
        self.data_queue = data_queue

        self.alpha = 0.015
        self.init_value = 1

    def run(self):

        self.init_time = time.time()

        #Setting up Epoc
        print("Setting up Epoc ...")
        #time.sleep(5)

        while self.controller.running:
            #Get Time and data point
            t = time.time() - self.init_time
            y = self.sineSignal(t)

            smooth_y = self.alpha*y + (1-self.alpha)*self.init_value
            self.init_value = smooth_y

            self.data_queue.put_nowait({'time': t,'data': y, 'smooth':smooth_y})
            #print('remaining capacity:', self.data_queue.maxsize - self.data_queue.qsize())
            time.sleep(0.125)

    @staticmethod
    def sineSignal(t):
        # return np.clip(np.sin(0.05*np.pi * t), -1,1)
        data_point = 1 if np.sin(0.05 * np.pi * t) > 0 else 0

        flip_label = 1 if random.uniform(0.0, 1.01) > 0.96 else 0

        if flip_label:
            if data_point == 1:
                data_point =0
            elif data_point ==0:
                data_point = 1

        return data_point

class Controller:

    def __init__(self):

        self.root = Tk()
        self.lifePlots = LivePlots(self.root,self)

        self.data_queue = queue.Queue(maxsize = 1000)
        self.running = False

        self.data_producer = DataProducer(self, data_queue=self.data_queue)


    def run(self):
        #Set running Flag
        self.running = True
        #Run data producer Thread
        self.data_producer.start()
        #Run Live plot
        self.root.mainloop()
        #Stop data producer thread
        self.running = False

        print("Finish Script")


if __name__ == '__main__':
    controllerMain = Controller()
    controllerMain.run()