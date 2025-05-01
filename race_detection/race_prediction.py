"""
1. Instantiate the model
	emo = f_my_race.Race_Model()
2. Input an image containing only one face (use a face detection model to extract an image with only the face)
	emo.predict_race(face_image)
"""

#from basemodels import VGGFace
from deepface.basemodels import VGGFace
import os
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Activation
from tensorflow.keras.preprocessing import image
import cv2


class Race_Model():
    def __init__(self):
        self.model = self.loadModel()
        self.race_labels = ['asian', 'indian', 'african', 'white', 'middle eastern', 'hispanic']

    def predict_race(self,face_image):
        """
        Predicts the race of a given face image.
        :param face_image: Input face image (pre-processed)
        :return: Predicted race label
        """
        # Preprocess the input image
        image_preprocesing = self.transform_face_array2race_face(face_image)
        # Get the model's prediction (output probabilities)
        race_predictions = self.model.predict(image_preprocesing )[0,:]
        # Get the race with the highest probability
        result_race = self.race_labels[np.argmax(race_predictions)]
        return result_race

    def loadModel(self):
        model = VGGFace.baseModel()

        # Define the number of output classes (6 race categories)
        classes = 6
        base_model_output = Sequential()
        base_model_output = Conv2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation('softmax')(base_model_output)	

        race_model = Model(inputs=model.input, outputs=base_model_output)	

        #load weights	
        weight_path = "/Users/Azan/Documents/1UNI Computing and Mathematical sciences/3rd yr/2 SEM/Comp project/Project main/rtfprediction_face_info/deepface_backup/weights/race_model_single_batch.h5"

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Race model weights not found at: {weight_path}")

        race_model.load_weights(weight_path)
        return race_model

    def transform_face_array2race_face(self,face_array,grayscale=False,target_size = (224, 224)):
        detected_face = face_array
        # Convert to grayscale param if specified
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            # Resize image to the required input size
        detected_face = cv2.resize(detected_face, target_size)
        img_pixels = image.img_to_array(detected_face)
        # Expand dimensions to fit the model's expected input shape (batch size of 1)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #normalize input in [0, 1]
        img_pixels /= 255
        return img_pixels