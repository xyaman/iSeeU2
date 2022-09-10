import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

while True: # Run forever
    if GPIO.input(11) == GPIO.HIGH:
        print("Button was pushed!")
    if GPIO.input(13) == GPIO.HIGH:
        print("Button2 was pushed!")
