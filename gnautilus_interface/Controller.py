from tkinter import Tk
from gnautilus_interface.DataProducer import DataProducer
from gnautilus_interface.LivePlots import LivePlots
import queue


class Controller:
    def __init__(self):
        self.root = Tk()
        self.lifePlots = LivePlots(self.root,self)
        self.data_queue = queue.Queue(maxsize = 1000)
        self.running = False
        self.data_producer = DataProducer(self, data_queue=self.data_queue)

    def run(self):
        self.running = True #Set running Flag
        self.data_producer.start() #Run data producer Thread
        #Run Live plot
        self.root.mainloop()
        #Stop data producer thread when gui is closed
        self.running = False


if __name__ == '__main__':
    controllerMain = Controller()
    controllerMain.run()
    print("Finish Script")