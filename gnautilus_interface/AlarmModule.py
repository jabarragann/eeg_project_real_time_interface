import time
from threading import Thread
import traceback
import simpleaudio as sa
import serial
from pylsl import StreamInfo, StreamOutlet


class RecordEventInLSL:

    def __init__(self):
        self.info = StreamInfo('AlarmEvents', 'Markers', 2, 0, 'string', 'alarmEvents43536')
        self.outlet = StreamOutlet(self.info)

    def record_event(self, event_description):
        self.outlet.push_sample([event_description , "{:0.3f}".format(time.time())])

class AlarmModule(Thread):

    def __init__(self, controller):
        super().__init__()
        self.init_time = time.time()
        self.controller = controller
        self.alarm_threshold = controller.detection_threshold
        self.alarm_deque = controller.alarm_deque
        self.alarm = sa.WaveObject.from_wave_file('./sounds/beep-08.wav')
        self.up_sound =sa.WaveObject.from_wave_file('./sounds/beep-08.wav')#up-sound.wav
        self.down_sound = sa.WaveObject.from_wave_file('./sounds/down-sound.wav')


    def run(self):
        # Get Initial time
        prev_time = time.time()

        # Collect Data until time session Time is over
        try:
            while self.controller.running:
                current_time = time.time()

                l = list(self.alarm_deque)
                mean = sum(l)/len(l)
                self.controller.alarm_deque_mean = mean

                if current_time - prev_time > 5:
                    prev_time = current_time
                    if mean > self.alarm_threshold:
                        play_obj = self.up_sound.play()
                        play_obj.wait_done()
                    else:
                        play_obj = self.down_sound.play()
                        play_obj.wait_done()

                # if mean > self.alarm_threshold:
                #     play_obj = self.alarm.play()
                #     play_obj.wait_done()
                time.sleep(0.5)

        except Exception as ex:
            print("Error in alarm module!!")
            print(ex)
            print(traceback.format_exc())
        finally:
            print("Closing alarm module")

class FeedbackModule(Thread):

    """
    Class that controls the Arduino motors of the bleeding simulator
    """
    def __init__(self, controller):
        super().__init__()
        self.init_time = time.time()
        self.controller = controller
        self.alarm_threshold = controller.detection_threshold_up
        self.alarm_deque = controller.alarm_deque
        self.alarm = sa.WaveObject.from_wave_file('./sounds/beep-08.wav')
        self.up_sound =sa.WaveObject.from_wave_file('./sounds/beep-08.wav')
        self.down_sound = sa.WaveObject.from_wave_file('./sounds/down-sound.wav')

        #LSL
        self.lslSender = RecordEventInLSL()

        #Serial communication
        # self.serial_connection = serial.Serial(port='COM3', baudrate=9600, timeout=.1)
        # time.sleep(1)
        # print(self.serial_connection.readline())

    def write_read(self, x):
        self.serial_connection.write(bytes(x, encoding='utf8'))
        time.sleep(0.05)
        data = self.serial_connection.readline()
        return data.decode('ascii')

    def run(self):
        # Get Initial time
        prev_time = time.time()

        # Collect Data until time session Time is over
        try:
            while self.controller.running:
                current_time = time.time()

                l = list(self.alarm_deque)
                mean = sum(l)/len(l)
                self.controller.alarm_deque_mean = mean

                if current_time - prev_time > 30:
                    prev_time = current_time
                    # if mean > self.controller.detection_threshold_up:
                    #     play_obj = self.up_sound.play()
                    #     play_obj.wait_done()
                    #     val = self.write_read(str(2))#0
                    #     print("[FeedbackModule] "+val)
                    #     self.lslSender.record_event("high workload")
                    # elif mean < self.controller.detection_threshold_down:
                    #     play_obj = self.down_sound.play()
                    #     play_obj.wait_done()
                    #     val = self.write_read(str(2))
                    #     print("[FeedbackModule] " + val)
                    #     self.lslSender.record_event("low workload")
                    # else:
                    #     play_obj = self.down_sound.play()
                    #     play_obj.wait_done()
                    #     val = self.write_read(str(1))
                    #     print("[FeedbackModule] " + val)
                    #     self.lslSender.record_event("medium workload")
                # if mean > self.alarm_threshold:
                #     play_obj = self.alarm.play()
                #     play_obj.wait_done()
                time.sleep(0.5)

        except Exception as ex:
            print("Error in alarm module!!")
            print(ex)
            print(traceback.format_exc())
        finally:
            print("Closing alarm module")