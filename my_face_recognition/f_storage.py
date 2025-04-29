'''
load the images that are in the folder database_images
'''
import config as cfg
import os
from my_face_recognition import f_main
import cv2
import numpy as np
import traceback

DB_FEATURES_FILE = "db_features.npy"
DB_NAMES_FILE = "db_names.npy"

def load_images_to_database():
    """
    Load face features and names from precomputed files if they exist.
    Otherwise, compute from images and save for future fast loading.
    """
    if os.path.exists(DB_FEATURES_FILE) and os.path.exists(DB_NAMES_FILE):
        print("Loading precomputed database...")
        db_features = np.load(DB_FEATURES_FILE, allow_pickle=True)
        db_names = np.load(DB_NAMES_FILE, allow_pickle=True)
        return db_names.tolist(), db_features
    else:
        print("No precomputed database found. Building database from images...")
        list_images = os.listdir(cfg.path_images)
        list_images = [file for file in list_images if file.endswith(('.jpg', '.jpeg', 'JPEG'))]

        name = []
        Feats = []

        for file_name in list_images:
            im = cv2.imread(cfg.path_images + os.sep + file_name)

            if im is None:
                continue  # Skip if image not loaded properly

            box_face = f_main.rec_face.detect_face(im)
            feat = f_main.rec_face.get_features(im, box_face)
            if len(feat) != 1:
                continue  # Skip if no face or multiple faces detected
            else:
                new_name = file_name.split(".")[0]
                if new_name == "":
                    continue
                name.append(new_name)
                if len(Feats) == 0:
                    Feats = np.frombuffer(feat[0], dtype=np.float64)
                else:
                    Feats = np.vstack((Feats, np.frombuffer(feat[0], dtype=np.float64)))

        # Save precomputed database for next time
        np.save(DB_FEATURES_FILE, Feats)
        np.save(DB_NAMES_FILE, np.array(name))
        return name, Feats

def insert_new_user(rec_face, name, feat, im):
    """
    Insert a new user into both the in-memory database and saved .npy files.
    """
    try:
        rec_face.db_names.append(name)
        if len(rec_face.db_features) == 0:
            rec_face.db_features = np.frombuffer(feat[0], dtype=np.float64)
        else:
            rec_face.db_features = np.vstack((rec_face.db_features, np.frombuffer(feat[0], dtype=np.float64)))

        # save the image
        cv2.imwrite(cfg.path_images + os.sep + name + ".jpg", im)

        # Also update saved npy database
        np.save(DB_FEATURES_FILE, rec_face.db_features)
        np.save(DB_NAMES_FILE, np.array(rec_face.db_names))

        return 'ok'
    except Exception as ex:
        error = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        return error