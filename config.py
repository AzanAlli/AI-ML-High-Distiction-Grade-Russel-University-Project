# -------------------------------------- emotion_detection ---------------------------------------
# emotion detection model
path_model = 'emotion_detection/Modelos/model_dropout.hdf5'
# Model parameters: the image must be converted to a 48x48 grayscale image
w,h = 48,48
rgb = False
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# -------------------------------------- face_recognition ---------------------------------------
path_images = "images_db"

