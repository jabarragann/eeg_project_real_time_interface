from tkinter import Tk
from gnautilus_interface.DataProducerEyeTracker import DataProducer
from gnautilus_interface.LivePlotsV2 import LivePlots
from gnautilus_interface.AlarmModule import AlarmModule, FeedbackModule
from gnautilus_interface.OddballModule import OddBallModule
from gnautilus_interface.SuctionAssistantModule import SuctionAssistantModule
import queue
from collections import deque
import numpy as np

class Controller:
    def __init__(self):

        self.detection_threshold_up = 0.75
        self.detection_threshold_down = 0.0
        self.alarm_deque_mean = 0.0
        self.send_feedback = True  # If false send the random help

        self.root = Tk() ##This guy hides the attributes that are below!
        self.lifePlots = LivePlots(self.root,self)
        self.data_queue = queue.Queue(maxsize = 1000)
        self.alarm_deque = deque(np.zeros(1).tolist())

        self.running = True

        self.send_feedback = True #If false send the random help
        self.data_producer = DataProducer(self, data_queue=self.data_queue, alarm_deque=self.alarm_deque)

        #Feedback module
        self.suction_module = SuctionAssistantModule(self)

    def run(self):
        self.data_producer.start() #Run data producer Thread
        self.suction_module.start()

        #Run Live plot
        self.root.mainloop()

        #Stop data producer thread when gui is closed
        self.running = False


if __name__ == '__main__':
    controllerMain = Controller()
    controllerMain.run()
    print("Finish Script")