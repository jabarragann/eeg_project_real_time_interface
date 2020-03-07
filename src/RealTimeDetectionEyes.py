import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from tkinter import Tk, Label, Button
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import time
import queue
import threading
from threading import Thread
import asyncio
import datetime
from contextlib import suppress
from asyncio import CancelledError
import json
import random

#Deep Learning Modules
from tensorflow.keras.models import Model, load_model

try:
    from cortex_2 import Cortex
except ImportError:
    print("Pycharm error")
    from lib_2.cortex import Cortex as Cortex


class LivePlotLogic:
    def __init__(self, controller, axe = None, line = None, maxt=10, dt=0.02, workload_label = None):
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
        self.tdata  = [0]
        self.ydata  = [0]
        self.ydata2 = [0]

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

            if lastt > self.tdata[0] + self.maxt:  # reset the arrays
                self.start_removing = True
            if self.start_removing:
                self.tdata = self.tdata[1:]
                self.ydata = self.ydata[1:]
                self.ydata2 = self.ydata2[1:]
                self.ax.set_xlim(self.tdata[0], self.tdata[-1] + self.dt * 10)

            # Get new sample
            new_y = frame[1]
            new_t = frame[0]
            new_y2 = frame[2]

            # Get and update label color
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
        self.line.set_markersize(2)
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
        self.cortex = Cortex('./CortexInfo/clientId.txt')

        #Keras model to do predictions
        self.predictionModel = None

        #Normalization Values from training
        normalizationValues = pd.read_csv('./Model/modelEyesNormalization.csv', delimiter=',')
        self.X_mean = normalizationValues['mean'].values
        self.X_std = normalizationValues['std'].values
        self.epsilon = 1e-8

        #Moving average filter
        self.alpha = 0.1
        self.init_value = 1

    def run(self):

        #Setting up prediction Model
        print("Setting up Model ...")
        self.predictionModel = load_model('./Model/modelEyes.h5')
        # time.sleep(5)

        #Setting up Epoc
        print("Setting up Epoc ...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        with open('./CortexInfo/authorizationToken.txt', 'r') as f:
            token = f.readline()

        loop.run_until_complete(self.streamEpocData(self.cortex, token=token))

        # Graceful shutdown of cooroutine
        print("Finishing Remaining tasks ...")
        for task in asyncio.Task.all_tasks():
            task.cancel()
            with suppress(CancelledError):
                loop.run_until_complete(task)
        loop.close()
        time.sleep(5)

    async def streamEpocData(self, cortex, token):
        if token is None:
            print("** AUTHORIZE **")
            await cortex.authorize(debit=2)
        else:
            cortex.auth_token = token

        await cortex.get_license_info()
        await cortex.query_headsets()
        if len(cortex.headsets) > 0:
            print("** CREATE SESSION **")
            await cortex.create_session(activate=True, headset_id=cortex.headsets[0])
            print("** CREATE RECORD **")
            await cortex.create_record(title="test record 1")
            print("** SUBSCRIBE eeg **")
            await cortex.subscribe(['pow'])

            #Get Initial time
            self.init_time = time.time()
            self.update_time = time.time()

            # Collect Data until time session Time is over
            try:
                while self.controller.running:
                    # print(self.update_time - time.time())
                    self.update_time = time.time()

                    #Get EEG Measurement
                    resp = await cortex.get_data()
                    resp = json.loads(resp)
                    dataPoint = np.array(resp['pow'])
                    dataPoint = dataPoint.reshape(1, -1)
                    t = resp['time'] - self.init_time

                    # Normalize Data
                    dataPoint = (dataPoint - self.X_mean) / (self.X_std + self.epsilon)

                    # Make Prediction
                    prediction = self.predictionModel.predict(dataPoint)
                    prediction = prediction[0,1]
                    predicted_label = 1  if prediction > 0.5 else 0

                    # #Only for simulation
                    # predicted_label = self.sineSignal(t)

                    #Apply moving average
                    smooth_label = self.alpha * predicted_label + (1 - self.alpha) * self.init_value
                    self.init_value = smooth_label

                    #Send prediction for plotting
                    print('time', t, 'prediction', prediction, 'label', predicted_label)
                    self.data_queue.put_nowait({'time': t, 'data': predicted_label, 'smooth': smooth_label})

            except Exception as ex:
                print("Error!!")
                print(ex)
            finally:
                await cortex.close_session()

    @staticmethod
    def sineSignal(t):
        # return np.clip(np.sin(0.05*np.pi * t), -1,1)
        data_point = 1 if np.sin(0.1 * np.pi * t) > 0 else 0

        flip_label = 1 if random.uniform(0.0, 1.01) > 0.97 else 0

        if flip_label:
            if data_point == 1:
                data_point = 0
            elif data_point == 0:
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




if __name__ == '__main__':
    controllerMain = Controller()
    controllerMain.run()
    print("Finish Script")