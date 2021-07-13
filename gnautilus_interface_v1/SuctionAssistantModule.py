import time
from threading import Thread
import traceback
import random
import serial
from pylsl import StreamInfo, StreamOutlet
import simpleaudio as sa

class SuctionAssistantModule(Thread):

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

        self.wait_time = 15 #Time doing suction
        self.min_time_between_suct = 3 #Minimum time between suctions

        #LSL outlet control
        info = StreamInfo(name='AssistantState', type='Markers', channel_count=1, nominal_srate=0,
                          channel_format='string', source_id='velocity')
        self.assistant_state_outlet = StreamOutlet(info)

        #Arduino serial
        try:
            self.serial_connection = serial.Serial(port='COM3', baudrate=9600, timeout=.2)
            time.sleep(1)
            # answer = str(self.serial_connection.readline(), 'utf-8')
            # print("[Serial response] {:}".format(answer))
        except Exception as e:
            print("Check serial connection with arduino")
            self.controller.running = False

    def send_random_feedback(self):
        random_sleep_time = random.randint(self.min_time_between_suct, 35)
        print(random_sleep_time)
        time.sleep(random_sleep_time)

        print("[State] {:}".format("Suction"))
        self.arduino_alarm("Suction")
        self.assistant_state_outlet.push_sample(['Suction'])
        time.sleep(self.wait_time)

        print("[State] {:}".format("Stop"))
        self.arduino_alarm("Stop")
        self.assistant_state_outlet.push_sample(['Stop'])

    def arduino_alarm(self, state):
        if state == "Motor on":
            answer = self.write_read("1")
        elif state == "Motor off":
            answer = self.write_read("0")
        elif state == "Suction":
            answer = self.write_read("2")
        elif state == "Stop":
            answer = self.write_read("3")
        else:
            answer = "state not recognized"
        print("[Serial response] {:}".format(answer))

    def write_read(self, x):
        self.serial_connection.write(bytes(x, encoding='utf8'))
        time.sleep(0.05)
        data = self.serial_connection.readline()
        return data.decode('ascii')

    def run(self):
        idle = False
        activation_time = time.time()
        # Collect Data until time session Time is over
        try:
            motor_prev_time = time.time()
            while self.controller.running:
                #Turn on and off motors
                motor_time = time.time()

                diff = int(motor_time - motor_prev_time) % 240
                # print("Diff", diff)
                if 60 < diff < 65:
                    print("Turn on the motor")
                    self.arduino_alarm("Motor on")
                    self.assistant_state_outlet.push_sample(['Motor on'])
                elif 0 < diff < 5:
                    print("Turn off the motor")
                    self.arduino_alarm("Motor off")
                    self.assistant_state_outlet.push_sample(['Motor off'])

                #Send feedback according to workload
                if self.controller.send_feedback:
                    l = list(self.alarm_deque)
                    mean = sum(l) / len(l)
                    self.controller.alarm_deque_mean = mean

                    # If not idle activate feedback whenever high workload is activated
                    if (mean > self.controller.detection_threshold_up) and not idle:
                        print("[State] {:}".format("Suction"))
                        self.arduino_alarm("Suction")
                        self.assistant_state_outlet.push_sample(['Suction'])
                        activation_time = time.time()
                        idle = True

                    # After activating high workload wait for 'wait_time' until going back to normal
                    if (time.time() - activation_time > self.wait_time) and idle:
                        print("[State] {:}".format("Stop"))
                        self.arduino_alarm("Stop")
                        self.assistant_state_outlet.push_sample(['Stop'])
                        # Arduino stop tone
                        time.sleep(self.min_time_between_suct)
                        idle = False

                    time.sleep(0.5)
                #Send random feedback
                else:
                    self.send_random_feedback()

        except Exception as ex:
            print("Error in alarm module!!")
            print(ex)
            print(traceback.format_exc())
        finally:
            print("Closing alarm module")