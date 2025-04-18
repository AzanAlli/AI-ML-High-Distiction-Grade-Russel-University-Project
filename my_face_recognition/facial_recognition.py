import face_recognition 
import numpy as np

def detect_face(image):
    '''
    Input: image numpy.ndarray, shape=(W,H,3)
    Output: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] , each tuple represents a detected face.
    If no face is detected  --> Output: []
    '''
    Output = face_recognition.face_locations(image)
    return Output

def get_features(img,box):
    '''
    Input:
        - img: image numpy.ndarray, shape=(W,H,3)
        - box: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)], each tuple represents a detected face.
    Output:
        - features: [array,array,...,array], each array represents the features of a detected face.
    '''
    features = face_recognition.face_encodings(img,box)
    return features

def compare_faces(face_encodings,db_features,db_names):
    '''
    Input:
        - db_features = [array,array,...,array], each array represents the features of a face.
        - db_names = [array,array,...,array], each array represents the features of a user.
    Output:
        - match_name: ['name', 'unknown'], a list with the names that matched.
        If there is a face but no match, it returns 'unknown'.
    '''
    match_name = []
    names_temp = db_names
    Feats_temp = db_features           

    for face_encoding in face_encodings:
        try:
            dist = face_recognition.face_distance(Feats_temp,face_encoding)
        except:
            dist = face_recognition.face_distance([Feats_temp],face_encoding)
        index = np.argmin(dist)
        if dist[index] <= 0.6:
            match_name = match_name + [names_temp[index]]
        else:
            match_name = match_name + ["unknown"]
    return match_name