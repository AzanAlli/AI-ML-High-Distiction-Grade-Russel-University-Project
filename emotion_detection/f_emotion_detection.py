import config as cfg # Importing config file for model
import cv2
import numpy as np
from tensorflow.keras.models import load_model # Load pre-trained deep learning model
from tensorflow.keras.preprocessing.image import img_to_array # Convert images to array format for model processing

class predict_emotions():
    def __init__(self):
        self.model = load_model(cfg.path_model) # Load the model from the specified path

    def preprocess_img(self,face_image,rgb=True,w=48,h=48):
        face_image = cv2.resize(face_image, (w,h))  # Resize image to model input size
        if rgb == False: # Convert to grayscale if required
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = face_image.astype("float") / 255.0
        face_image= img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0) # Expand dimensions to match model input shape
        return face_image

    def get_emotion(self,img,boxes_face):
        """
       Predict emotions for detected faces in an image.

       Steps:
       1. Iterate through each detected face in `boxes_face`.
       2. Extract the face region from the image.
       3. Preprocess the face image.
       4. Use the trained model to predict the emotion.
       5. Map the predicted label index to the corresponding emotion.
       6. Store the detected emotions in a list and return them.

       :param img: The input image containing faces
       :param boxes_face: List of bounding box coordinates for detected faces [(y0, x0, y1, x1), ...]
       :return: A tuple containing the face bounding boxes and corresponding predicted emotions
       """
        emotions = [] # List to store predicted emotions
        if len(boxes_face)!=0:
            for box in boxes_face:
                y0,x0,y1,x1 = box # Extract face bounding box coordinates
                face_image = img[x0:x1,y0:y1] # Crop the face region from the image
                face_image = self.preprocess_img(face_image ,cfg.rgb, cfg.w, cfg.h)
                prediction = self.model.predict(face_image)
                emotion = cfg.labels[prediction.argmax()]
                emotions.append(emotion) # Append detected emotion to list
        else:
            emotions = [] #Not detected
            boxes_face = [] #Not detected
        return boxes_face,emotions

