"""Example program to demonstrate how to send string-valued markers into LSL."""

import random
import time
from pylsl import StreamInfo, StreamOutlet


def main():
    info = StreamInfo(name ='Oddball_speed', type='Markers', channel_count =1, nominal_srate=0,
                      channel_format='string', source_id='velocity')
    outlet = StreamOutlet(info)

    state = False
    try:
        while True:
            # command = "700" if state else "1600"
            command = "2t_1r" if state else "1t_1r"
            state = not state
            outlet.push_sample([command])
            sleep_time = random.randint(20, 25)
            print("New ITI for odd ball is ",command)
            print("Sleeping for ", sleep_time)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")

if __name__ == '__main__':
    main()