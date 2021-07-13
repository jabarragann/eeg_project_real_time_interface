import time
from threading import Thread
import traceback

import serial
from pylsl import StreamInfo, StreamOutlet
import simpleaudio as sa

class OddBallModule(Thread):

    """
    Class that controls the oddBall speed
    """
    def __init__(self, controller):
        super().__init__()
        self.init_time = time.time()
        self.controller = controller
        self.alarm_threshold = controller.detection_threshold_up
        self.alarm_deque = controller.alarm_deque

        self.up_sound = sa.WaveObject.from_wave_file('./sounds/up-sound.wav')

        self.wait_time = 20

        #Speed control
        info = StreamInfo(name='Oddball_speed', type='Markers', channel_count=1, nominal_srate=0,
                          channel_format='string', source_id='velocity')
        self.oddball_speed_outlet = StreamOutlet(info)


    def run(self):

        idle = False
        activation_time = time.time()
        # Collect Data until time session Time is over
        try:
            while self.controller.running:


                l = list(self.alarm_deque)
                mean = sum(l)/len(l)
                self.controller.alarm_deque_mean = mean

                #If not idle activate feedback whenever high workload is activated
                if  (mean > self.controller.detection_threshold_up) and not idle:

                    # Send low speed command
                    if self.controller.send_feedback:
                        print("Reducing difficulty. Low speed command send - 1t_1r")
                        #self.oddball_speed_outlet.push_sample(['1600'])
                        self.oddball_speed_outlet.push_sample(['1t_1r'])

                    activation_time = time.time()
                    idle = True
                    # play_obj = self.up_sound.play()
                    # play_obj.wait_done()

                #After activating high workload wait for 'wait_time' until going back to normal
                if (time.time() - activation_time > self.wait_time) and idle:

                    #Send high speed command
                    if self.controller.send_feedback:
                        print("Going back to normal. High speed command send - 2t_1r")
                        # self.oddball_speed_outlet.push_sample(['700'])
                        self.oddball_speed_outlet.push_sample(['2t_1r'])


                    idle = False

                time.sleep(0.5)

        except Exception as ex:
            print("Error in alarm module!!")
            print(ex)
            print(traceback.format_exc())
        finally:
            print("Closing alarm module")