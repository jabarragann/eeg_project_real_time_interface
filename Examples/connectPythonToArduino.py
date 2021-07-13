import serial
import time

arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)

time.sleep(1)
print(arduino.readline())


def write_read(x):
    arduino.write(bytes(x, encoding='utf8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data.decode('ascii')

if __name__ == '__main__':
    # while True:
    #     num = input("Enter a number: ") # Taking input from user
    #     value = write_read(num)
    #     print(value) # printing the value
    # time.sleep(2)
    for i in range(10):
        val = write_read(str(i%2))
        print(i, val)
        time.sleep(1)
