import logging

# TODO: Change path
import webserver.db as db

import cv2 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

RESIZED_DIMENSIONS = (300, 300) # Dimensions that SSD was trained on. 
IMG_NORM_RATIO = 0.007843 # In grayscale a pixel can range between 0 and 255

CAMERA = 0

class Recognition:
    def __init__(self, source: int | str ="recognition/video.mp4") -> None:
        self.is_running = False
        self.people = 0
        self.recognized = []
        self.capture = cv2.VideoCapture(source)

        database = db.get()
        rows = database.get_sample()
        self.labels = [x["fname"] for x in rows]

        # Load the pre-trained neural network
        logging.info("Loading pre-trained neural network (objects)")
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

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trained_data.yml")

        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        prototxt_path = "recognition/deploy.prototxt.txt"
        # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
        model_path = "recognition/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        logging.info("Loading FACE Recognition Model")
        self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
  
    def deinit(self):
        # Stop when the video is finished
        self.capture.release()

    def update(self, images_path):
        trainer = FaceTrainer(images_path)
        trainer.train()

        self.recognizer.read("trained_data.yml")

    def run(self, rects=False, window=False):
     
        # Process the video
        while self.capture.isOpened():
         
            # Capture one frame at a time
            ok, frame = self.run_once(rects=rects)

            if ok:
                # print(self.people)

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

        self.recognized = {}

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
                
                if confidence > 0.50:
             
                    idx = int(neural_network_output[0, 0, i, 1])

                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                        [w, h, w, h])

                    (startX, startY, endX, endY) = bounding_box.astype("int")

                    self.recognized[self.classes[idx]] = 0

                    if self.classes[idx] == "person":
                        self.people += 1

                    if rects:
                        label = f"{self.classes[idx]} {confidence * 100}"
                        cv2.rectangle(frame, (startX, startY), (
                            endX, endY), self.bbox_colors[idx], 2)     
                         
                        y = startY - 15 if startY - 15 > 15 else startY + 15    
 
                        cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, self.bbox_colors[idx], 2)

            frame_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network to obtain the face detections,
            # then initialize a list to store the predicted bounding boxes
            self.face_net.setInput(frame_blob)
            detections = self.face_net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                     # extract the face ROI, resize it, and convert it to grayscale
                    faceROI = frame[startY:endY, startX:endX]
                    faceROI = cv2.resize(faceROI, (47, 62))
                    faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

                    label, _ = self.recognizer.predict(faceROI)
                    self.recognized[self.labels[label]] = 0

                    if rects:
                        label = f"{self.labels[label]} {confidence * 100}"
                        cv2.rectangle(frame, (startX, startY), (
                            endX, endY), self.bbox_colors[0], 2)     
                         
                        y = startY - 15 if startY - 15 > 15 else startY + 15    
 
                        cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, self.bbox_colors[1], 2)


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
        logging.info("calling _detect_faces")

        # grab the dimensions of the image and then construct a blob from it
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        #blob = cv2.dnn.blobFromImage(cv2.resize(image, RESIZED_DIMENSIONS), 
         #                  IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)

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

    def _load_faces_dataset(self, net):

        database = db.get()
        images = database.get_sample()
        images_path = []
        images_name = []

        for person in images:
            if person["fname"] == None:
                continue

            images_path.append(person["path"])
            images_name.append(person["fname"])

        # initialize lists to store our extracted faces and associated labels
        labels = []
        faces = []

        logging.info("Getting ROI of every image")
        for idx, img_path in enumerate(images_path):
            image = cv2.imread(self.image_dir + img_path)
            boxes = self._detect_faces(net, image)

            # TODO: Recognize more than one
            for (startX, startY, endX, endY) in boxes[:1]:
                 # extract the face ROI, resize it, and convert it to grayscale
                faceROI = image[startY:endY, startX:endX]
                faceROI = cv2.resize(faceROI, (47, 62))
                faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

                # update our faces and labels lists
                faces.append(faceROI)
                labels.append(idx)

        logging.info("ROI Done")

        # convert our faces and labels lists to NumPy arrays
        faces = np.array(faces)
        labels = np.array(labels)

        # return a 2-tuple of the faces and labels
        return (faces, labels)


    def train(self, min_confidence=0.5):
        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        prototxt_path = "recognition/deploy.prototxt.txt"
        # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
        model_path = "recognition/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        logging.info("Loading train model, face recognizer")
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        (faces, labels) = self._load_faces_dataset(net)

        # encode the string labels as integers
        le = LabelEncoder()
        labels = le.fit_transform(labels)

        # TODO: We should compare and see accuraccy
        # construct our training and testing split
        # (trainX, _, trainY, _) = train_test_split(faces,
        #     labels, test_size=0.5, stratify=labels, random_state=42)

        logging.info("Training model")
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
        recognizer.train(faces, labels)
        recognizer.save("trained_data.yml")

        logging.info("Training finished")

if __name__ == "__main__":
    # trainer = FaceTrainer("./webservice/static/images/")
    # trainer.train()

    r = Recognition(CAMERA)
    r.run(rects=True, window=True)
    r.deinit()

    #
    # face_recognition()
