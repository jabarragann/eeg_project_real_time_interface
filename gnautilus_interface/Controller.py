from tkinter import Tk
from gnautilus_interface.DataProducer import DataProducer
from gnautilus_interface.LivePlotsV2 import LivePlots
from gnautilus_interface.AlarmModule import AlarmModule
import queue
from collections import deque

class Controller:
    def __init__(self):

        self.detection_threshold = 0.75
        self.root = Tk()
        self.lifePlots = LivePlots(self.root,self)
        self.data_queue = queue.Queue(maxsize = 1000)
        self.alarm_deque = deque([0, 0, 0, 0, 0, 0])
        self.running = False
        self.data_producer = DataProducer(self, data_queue=self.data_queue, alarm_deque=self.alarm_deque)
        self.alarm_module = AlarmModule(self)


    def run(self):
        self.running = True #Set running Flag
        self.data_producer.start() #Run data producer Thread
        self.alarm_module.start()

        #Run Live plot
        self.root.mainloop()
        #Stop data producer thread when gui is closed
        self.running = False


if __name__ == '__main__':
    controllerMain = Controller()
    controllerMain.run()
    print("Finish Script")