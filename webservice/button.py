import RPi.GPIO as GPIO
import time, photo

GPIO.setmode(GPIO.BCM)

GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    input1_state = GPIO.input(17)
    input2_state = GPIO.input(27)
    if input1_state == False:
        print('Button 1 Pressed')
        photo.take()
        time.sleep(3)
    if input2_state == False:
        print('Button 2 Pressed')
        time.sleep(3)
