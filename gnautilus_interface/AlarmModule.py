import time
from threading import Thread
import traceback
import simpleaudio as sa

class AlarmModule(Thread):

    def __init__(self, controller):
        super().__init__()
        self.init_time = time.time()
        self.controller = controller
        self.alarm_threshold = controller.detection_threshold
        self.alarm_deque = controller.alarm_deque
        self.alarm = sa.WaveObject.from_wave_file('./sounds/beep-08.wav')

    def run(self):
        # Get Initial time
        self.init_time = time.time()

        # Collect Data until time session Time is over
        try:
            while self.controller.running:
                l = list(self.alarm_deque)
                mean = sum(l)/len(l)
                if mean > self.alarm_threshold:
                    play_obj = self.alarm.play()
                    play_obj.wait_done()

                time.sleep(5)

        except Exception as ex:
            print("Error in alarm module!!")
            print(ex)
            print(traceback.format_exc())
        finally:
            print("Closing alarm module")

