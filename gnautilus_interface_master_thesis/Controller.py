from tkinter import Tk
from gnautilus_interface_v1.DataProducer import DataProducer
from gnautilus_interface_v1.LivePlotsV2 import LivePlots
from gnautilus_interface_v1.AlarmModule import AlarmModule, FeedbackModule
from gnautilus_interface_v1.OddballModule import OddBallModule
from gnautilus_interface_v1.SuctionAssistantModule import SuctionAssistantModule
import queue
from collections import deque
import numpy as np

class Controller:
    def __init__(self):

        self.detection_threshold_up = 0.60
        self.detection_threshold_down = 0.20
        self.alarm_deque_mean = 0.0

        self.root = Tk() ##This guy hides the attributes that are below!
        self.lifePlots = LivePlots(self.root,self)
        self.data_queue = queue.Queue(maxsize = 1000)
        self.alarm_deque = deque(np.zeros(12).tolist())

        self.running = True

        self.send_feedback = True #If false send the random help
        self.data_producer = DataProducer(self, data_queue=self.data_queue, alarm_deque=self.alarm_deque)
        # self.alarm_module = FeedbackModule(self)# AlarmModule(self)
        # self.oddball_module = OddBallModule(self)
        # self.suction_module = SuctionAssistantModule(self)

    def run(self):
        self.data_producer.start() #Run data producer Thread
        # self.alarm_module.start()
        # self.oddball_module.start()
        # self.suction_module.start()

        #Run Live plot
        self.root.mainloop()
        #Stop data producer thread when gui is closed
        self.running = False


if __name__ == '__main__':
    controllerMain = Controller()
    controllerMain.run()
    print("Finish Script")