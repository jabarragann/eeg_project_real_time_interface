from tkinter import Tk
from gnautilus_interface_v2.DataProducerEyeTracker import DataProducer
from gnautilus_interface_v2.LivePlotsV2 import LivePlots
from gnautilus_interface_v2.AlarmModule import AlarmModule, FeedbackModule
from gnautilus_interface_v2.OddballModule import OddBallModule
from gnautilus_interface_v2.SuctionAssistantModule import SuctionAssistantModule
import queue
from collections import deque
import numpy as np
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
from pylsl import  resolve_byprop
import threading

class Controller:
    def __init__(self):

        self.detection_threshold_up = 0.60
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

        #Threshold commands
        self.listenToCmd = threading.Thread(target=self.listen_threshold_markers)

    def run(self):
        self.data_producer.start() #Run data producer Thread
        self.suction_module.start()

        self.listenToCmd.start()

        #Run Live plot
        self.root.mainloop()

        #Stop data producer thread when gui is closed
        self.running = False

    def listen_threshold_markers(self):
        count = 0
        while self.running:
            streams = resolve_byprop('name', 'threshold_cmd', timeout=1)

            if len(streams) > 0 :
                inlet = StreamInlet(streams[0])
                print("oddball_speed stream found ")
                while self.running:
                    sample, timestamp = inlet.pull_sample(timeout=1)
                    if timestamp is not None:
                        print("got new speed %s at time %s" % (sample[0], timestamp))
                        cmd = str(sample[0])
                        if cmd == "u":
                            self.detection_threshold_up = self.detection_threshold_up + 0.05
                            print(self.detection_threshold_up)
                        elif cmd == "d":
                            self.detection_threshold_up = self.detection_threshold_up - 0.05
                            print(self.detection_threshold_up)
                        else:
                            print("not recognized")


                print('Close listening socket')
            else:
                print(count, "threshold cmd inlet not found")
                count += 1


if __name__ == '__main__':
    controllerMain = Controller()
    controllerMain.run()
    print("Finish Script")