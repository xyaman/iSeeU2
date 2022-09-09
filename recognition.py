import os
import pickle
from imutils import paths

import cv2 
import numpy as np 
from PIL import Image

RESIZED_DIMENSIONS = (300, 300) # Dimensions that SSD was trained on. 
IMG_NORM_RATIO = 0.007843 # In grayscale a pixel can range between 0 and 255

CAMERA = 0

class Recognition:
    def __init__(self, source="recognition/video.mp4") -> None:
        self.is_running = False
        self.people = 0
        self.capture = cv2.VideoCapture(source)

        # Load the pre-trained neural network
        self.neural_network = cv2.dnn.readNetFromCaffe('recognition/MobileNetSSD_deploy.prototxt.txt', 
                'recognition/MobileNetSSD_deploy.caffemodel')

        # List of categories and classes
        self.categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 
                       4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 
                       9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 
                      13: 'horse', 14: 'motorbike', 15: 'person', 
                      16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
                      19: 'train', 20: 'tvmonitor'}
         
        self.classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
                    "bus", "car", "cat", "chair", "cow", 
                   "diningtable",  "dog", "horse", "motorbike", "person", 
                   "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
                              
        # Create the bounding boxes
        self.bbox_colors = np.random.uniform(255, 0, size=(len(self.categories), 3))
  
    def deinit(self):
        # Stop when the video is finished
        self.capture.release()

    def run(self, rects=False, window=False):
     
        # Process the video
        while self.capture.isOpened():
         
            # Capture one frame at a time
            ok, frame = self.run_once(rects=rects)

            if ok:
                print(self.people)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
                if(window):
                    # We now need to resize the frame so its dimensions
                    # are equivalent to the dimensions of the original frame
                    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)

                    # Write the frame to the output video file
                    # displaying image with bounding box
                    cv2.imshow('face_detect', frame)

            else:
                if window:
                    cv2.destroyWindow('face_detect')
                break

    def run_once(self, rects=False):

        # Capture one frame at a time
        ok, frame = self.capture.read() 

        # Do we have a video frame? If true, proceed.
        if ok:
            self.people = 0
     
            # Capture the frame's height and width
            (h, w) = frame.shape[:2]

            # Create a blob. A blob is a group of connected pixels in a binary 
            # frame that share some common property (e.g. grayscale value)
            # Preprocess the frame to prepare it for deep learning classification
            frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS), 
                           IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
         
            # Set the input for the neural network
            self.neural_network.setInput(frame_blob)

            # Predict the objects in the image
            neural_network_output = self.neural_network.forward()

            # Put the bounding boxes around the detected objects
            for i in np.arange(0, neural_network_output.shape[2]):
             
                confidence = neural_network_output[0, 0, i, 2]
     
                # Confidence must be at least 30%       
                
                if confidence > 0.30:
             
                    idx = int(neural_network_output[0, 0, i, 1])

                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                        [w, h, w, h])

                    (startX, startY, endX, endY) = bounding_box.astype("int")

                    if self.classes[idx] == "person":
                        self.people += 1

                    label = f"{self.classes[idx]} {confidence * 100}"
                    if rects:
     
                        cv2.rectangle(frame, (startX, startY), (
                            endX, endY), self.bbox_colors[idx], 2)     
                         
                        y = startY - 15 if startY - 15 > 15 else startY + 15    
 
                        cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, self.bbox_colors[idx], 2)

            return ok, frame

class FaceTrainer:
    """
    FaceTrainer
    ----------
    images_dir : str
       Directory where the trainer will read the data
    """

    def __init__(self, images_dir="images") -> None:
        self.image_dir = images_dir

    def _detect_faces(self, net, image, min_confidence=0.5):
        # grab the dimensions of the image and then construct a blob from it
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network to obtain the face detections,
	    # then initialize a list to store the predicted bounding boxes
        net.setInput(blob)
        detections = net.forward()
        boxes = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > min_confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # update our bounding box results list
                boxes.append((startX, startY, endX, endY))

        # return the face detection bounding boxes
        return boxes

    def _load_faces(self):
        imagePaths = list(paths.list_images(self.image_dir))
        print(imagePaths)

    def train(self, min_confidence=0.5):

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, _, files in os.walk(self.image_dir):
             for file in files:
                 if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root).replace(" ","-").lower()
                    print(label, path)
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ =label_ids[label]
                    print(label_ids)

                    pil_image = Image.open(path).convert("L")
                    size = (550, 550)
                    final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image, "uint8")
                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=2, minNeighbors=5)

                    for (x, y, w, h) in faces:
                       roi = image_array[y:y+h, x:x+w]
                       x_train.append(roi)
                       y_labels.append(id_)


        with open("labels.pickles", 'wb') as f:
            pickle.dump(label_ids, f)

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainner.yml")
            

if __name__ == "__main__":
    # r = Recognition()
    # r.run(rects=True, window=True)
    # r.deinit()

    trainer = FaceTrainer("webserver/static/images/")
    trainer._load_faces()
