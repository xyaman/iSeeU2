from picamera import PiCamera
import time, os, db

camera = PiCamera()
time.sleep(2)

def take():
    path = "/home/pi/hackathon-git/webservice/static/images/"+str(len(os.listdir("../webservice/static/images"))+1)+".jpg"
    short_path = str(len(os.listdir("../webservice/static/images"))+1)+".jpg"
    camera.capture(path)
    database = db.get()
    database.insert_path(short_path)
