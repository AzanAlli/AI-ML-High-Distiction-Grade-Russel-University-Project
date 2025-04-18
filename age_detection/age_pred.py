#from basemodels import VGGFace
# tenserflow is used for image classification and object detection and keras is an api
from deepface.basemodels import VGGFace # Import the VGGFace model from DeepFace
import os
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Activation
from tensorflow.keras.preprocessing import image # Image preprocessing utilities
import cv2 # OpenCV for image processing


class Age_Model():
    def __init__(self):
        """
       Initialises the age prediction model by loading the pre-trained model
       and defining the output indexes for age classification.
       """
        self.model = self.loadModel() # Load the trained age prediction model
        self.output_indexes = np.array([i for i in range(0, 101)]) # Age class indexes (0-100)

    def predict_age(self,face_image):
        """
        Predicts the age of a given face image.

        Parameters:
        face_image (numpy array): The input image containing a face.

        Returns:
        float: The predicted apparent age of the person.
        """
        image_preprocesing = self.transform_face_array2age_face(face_image)
        age_predictions = self.model.predict(image_preprocesing )[0,:]
        result_age = self.findApparentAge(age_predictions) # Compute final age
        return result_age

    def loadModel(self):
        """
        Loads the VGGFace-based age prediction model and applies modifications.

        Returns:
        Model: The modified age prediction model with softmax output.
        """
        model = VGGFace.baseModel()
        #--------------------------
        classes = 101
        base_model_output = Sequential()
        base_model_output = Conv2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation('softmax')(base_model_output)
        #--------------------------
        age_model = Model(inputs=model.input, outputs=base_model_output)
        #--------------------------
        #load weights
        weight_path = "/Users/Azan/Documents/1UNI Computing and Mathematical sciences/3rd yr/2 SEM/Comp project/Project main/rtfprediction_face_info/deepface_backup/weights/age_model_weights.h5"
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file not found at {weight_path}")

        age_model.load_weights(weight_path)
        return age_model
        #--------------------------

    def findApparentAge(self,age_predictions):
        """
        Computes the apparent age based on the predicted age probabilities.

        Parameters:
        age_predictions (numpy array): The probability distribution over age classes.

        Returns:
        float: The computed apparent age.
        """
        apparent_age = np.sum(age_predictions * self.output_indexes)
        return apparent_age

    def transform_face_array2age_face(self,face_array,grayscale=False,target_size = (224, 224)):
        """
       Preprocesses the face image for model input.

       Parameters:
       face_array (numpy array): The input image containing a face.
       grayscale (bool): Whether to convert the image to grayscale. Default is False.
       target_size (tuple): The target image size for the model.

       Returns:
       numpy array: Preprocessed image ready for model prediction.
       """
        detected_face = face_array
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, target_size)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #normalize input in [0, 1]
        img_pixels /= 255
        return img_pixels