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

class LivePlotLogic:
    def __init__(self, controller, axe = None, line = None, maxt=10, dt=0.02):
        #Controller reference will allow to access to the data queue of the producer
        self.controller = controller

        #Artist and axe that is going to be updated
        self.animated_line = line
        self.ax = axe

        #Logic variables
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]

        self.init_time = time.time()
        self.init_time2 = time.time()
        self.total_time = 0

        self.start_removing = False

    def update_animation(self, y):

        #Get Last Sample
        lastt = self.tdata[-1]

        # Calculate current time
        self.dt = time.time() - self.init_time
        self.total_time += self.dt

        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
            self.start_removing=True
        if self.start_removing:
            self.tdata = self.tdata[1:]
            self.ydata = self.ydata[1:]
            self.ax.set_xlim(self.tdata[0], self.tdata[-1] + self.dt*10)

        #Calculate new sample
        new_y = self.sineSignal()

        #Update artists data
        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(new_y)
        self.animated_line.set_data(self.tdata, self.ydata)

        #Reset init time
        self.init_time = time.time()

        return self.animated_line,

    def init_animation(self):
        self.init_time = time.time()
        self.init_time2 = time.time()
        return self.animated_line,

    def sineSignal(self):
        return np.clip(np.sin(0.5*np.pi * self.total_time), -0.5,0.5)

    def check_data_queue(self):
        #Check producer data queue
        while True:
            if self.controller.data_queue.empty():
                yield [0,0,False]
            else:
                data = self.controller.data_queue.get_nowait()
                yield [data['time'], data['data'], True]


class LivePlots:
    def __init__(self, master, controller):
        #Font
        LARGE_FONT = ("Verdana", 12)
        self.x_max_lim = 10

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
        self.ax.set_xlim(0, self.x_max_lim)

        #Create data line
        self.line = Line2D([0,1,2,3,4,5], [1,2,4,5,6,8])
        self.line.set_linestyle('--')
        self.line.set_marker('*')
        self.line.set_markersize(2)
        self.line.set_color('red')

        self.ax.add_line(self.line)

        #Create animation logic and animation
        self.logic = LivePlotLogic(controller, axe = self.ax, line = self.line, maxt = self.x_max_lim)

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

    def run(self):

        self.init_time = time.time()

        while self.controller.running:
            #Get Time and data point
            t = time.time() - self.init_time
            y = self.sineSignal(t)
            self.data_queue.put_nowait({'time': t,'data': y})
            print('remaining capacity:', self.data_queue.maxsize - self.data_queue.qsize())
            time.sleep(0.10)

    @staticmethod
    def sineSignal(t):
        return np.clip(np.sin(0.5*np.pi * t), -0.8,0.8)

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