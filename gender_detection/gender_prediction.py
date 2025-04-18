#from basemodels import VGGFace
from deepface.basemodels import VGGFace
import os
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Activation
from tensorflow.keras.preprocessing import image
import cv2


class Gender_Model():
    def __init__(self): # Loading the pre-trained gender classification model
        self.model = self.loadModel()

    def predict_gender(self, face_image):
        image_preprocesing = self.transform_face_array2gender_face(face_image)
        gender_predictions = self.model.predict(image_preprocesing )[0,:]
        # Determine the predicted gender based on the highest probability class
        if np.argmax(gender_predictions) == 0:
            result_gender = "Woman"
        elif np.argmax(gender_predictions) == 1:
            result_gender = "Man"
        return result_gender

    def loadModel(self):
        model = VGGFace.baseModel()
        #--------------------------
        # Modify the model to classify between two genders
        classes = 2
        base_model_output = Sequential()
        base_model_output = Conv2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation('softmax')(base_model_output)
        #--------------------------
        # Create the final model
        gender_model = Model(inputs=model.input, outputs=base_model_output)
        #--------------------------
        #load weights
        weight_path = "/Users/Azan/Documents/1UNI Computing and Mathematical sciences/3rd yr/2 SEM/Comp project/Project main/rtfprediction_face_info/deepface_backup/weights/gender_model_weights.h5"

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Gender model weights not found at: {weight_path}")

        gender_model.load_weights(weight_path)
        return gender_model
        #--------------------------

    def transform_face_array2gender_face(self,face_array,grayscale=False,target_size = (224, 224)):
        detected_face = face_array
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, target_size)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #normalize input in [0, 1]
        img_pixels /= 255
        return img_pixels