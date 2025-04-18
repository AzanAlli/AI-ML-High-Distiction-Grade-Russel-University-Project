import cv2
import numpy as np
import face_recognition
from age_detection import age_pred
from gender_detection import gender_prediction
from race_detection import race_prediction
from emotion_detection import emotion_prediction
from my_face_recognition import f_main



age_detector = age_pred.Age_Model()
gender_detector =  gender_prediction.Gender_Model()
race_detector = race_prediction.Race_Model()
emotion_detector = emotion_prediction.predict_emotions()
rec_face = f_main.rec()
#----------------------------------------------



def get_face_info(im):
    # face detection
    boxes_face = face_recognition.face_locations(im)
    out = []
    if len(boxes_face)!=0:
        for box_face in boxes_face:
            box_face_fc = box_face
            x0,y1,x1,y0 = box_face
            # Calculate percentage-based padding
            face_height = x1 - x0
            face_width = y1 - y0
            padding_h = int(face_height * 0.25)
            padding_w = int(face_width * 0.25)

            # Apply padding to the bounding box
            x0 = max(x0 - padding_h, 0)
            y0 = max(y0 - padding_w, 0)
            x1 = min(x1 + padding_h, im.shape[0])
            y1 = min(y1 + padding_w, im.shape[1])
            box_face = np.array([y0,x0,y1,x1])
            face_features = {
                "name":[],
                "age":[],
                "gender":[],
                "race":[],
                "emotion":[],
                "bbx_frontal_face":box_face             
            } 

            face_image = im[x0:x1,y0:y1]

            # -------------------------------------- face_recognition ---------------------------------------
            face_features["name"] = rec_face.recognize_face2(im,[box_face_fc])[0]

            # -------------------------------------- age_detection ---------------------------------------
            age = age_detector.predict_age(face_image)
            face_features["age"] = str(round(age,2))

            # -------------------------------------- gender_detection ---------------------------------------
            face_features["gender"] = gender_detector.predict_gender(face_image)

            # -------------------------------------- race_detection ---------------------------------------
            face_features["race"] = race_detector.predict_race(face_image)

            # -------------------------------------- emotion_detection ---------------------------------------
            _,emotion = emotion_detector.get_emotion(im,[box_face])
            face_features["emotion"] = emotion[0]

            # -------------------------------------- out ---------------------------------------       
            out.append(face_features)
    else:
        face_features = {
            "name":[],
            "age":[],
            "gender":[],
            "race":[],
            "emotion":[],
            "bbx_frontal_face":[]             
        }
        out.append(face_features)
    return out



def bounding_box(out, img):
    for data_face in out:
        box = data_face["bbx_frontal_face"]
        if len(box) == 0:
            continue
        else:
            x0, y0, x1, y1 = box
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (225, 255, 0), 2)

            thickness = 2
            fontSize = 0.6
            line_spacing = 22  # spacing between lines
            y_text_start = y0 - 10

            labels = [
                ("age: " + data_face["age"], (0, 165, 255)),      # orange
                ("gender: " + data_face["gender"], (139, 0, 0)),  # dark blue
                ("race: " + data_face["race"], (255, 0, 255)),    # magenta
                ("emotion: " + data_face["emotion"], (0, 0, 255)),# red
                ("name: " + data_face["name"], (0, 255, 0))       # green
            ]

            for i, (text, color) in enumerate(labels):
                try:
                    y = y_text_start - (i * line_spacing)
                    cv2.putText(img, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness)
                except:
                    continue
    return img