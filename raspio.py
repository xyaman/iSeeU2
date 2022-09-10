import RPi.GPIO as GPIO
import time, os, logging, uuid
from picamera import PiCamera

import webserver.db as db

camera = PiCamera()

def take():
    # path = "/home/pi/hackathon-git/webservice/static/images/"+str(len(os.listdir("../webservice/static/images"))+1)+".jpg"

    filename = f"{uuid.uuid4()}.jpg"
    path = f"{os.path.dirname(__file__)}/webservice/static/images/{filename}"
    camera.capture(path)

    # Save to database, keep in mind that it uses only filename
    database = db.get()
    database.insert_path(filename)

INPUT1 = 17
INPUT2 = 27

def setup():
    logging.info("GPIO setup starting...")

    GPIO.setmode(GPIO.BCM)

    GPIO.setup(INPUT1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(INPUT2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def get_state(input: int):
    return not GPIO.input(input)


if __name__ == "__main__":
    setup()
    while True:
        input1_state = GPIO.input(INPUT1)
        input2_state = GPIO.input(INPUT2)
        if input1_state == False:
            print('Button 1 Pressed')
            take()
            time.sleep(3)
        if input2_state == False:
            print('Button 2 Pressed')
            time.sleep(3)
