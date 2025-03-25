import rtfprediction
import cv2
import time # Time module for FPS calculation
import imutils # Utility functions for image resizing
import argparse # Argument parser for command-line arguments

# Handles command-line args
parser = argparse.ArgumentParser(description="Face Info")
parser.add_argument('--input', type=str, default= 'webcam',
                    help="webcam or image")
parser.add_argument('--path_im', type=str,
                    help="path of image") # Path to image file if using an image input
args = vars(parser.parse_args()) # Parse arguments into a dictionary

type_input = args['input']
if type_input == 'image':
    # ----------------------------- image -----------------------------
    # ingest data
    frame = cv2.imread(args['path_im'])
    # get info from the frame
    out = rtfprediction.get_face_info(frame)
    # draw bounding box on the image
    res_img = rtfprediction.bounding_box(out,frame)
    cv2.imshow('Face info',res_img)
    cv2.waitKey(0)

if type_input == 'webcam':
    # ----------------------------- webcam -----------------------------
    cv2.namedWindow("Face info")
    cam = cv2.VideoCapture("demo.mp4")
    #cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture(1)

    while True:
        star_time = time.time() # Start timing for FPS calculation
        ret, frame = cam.read() # Capture frame from the webcam/video
        frame = imutils.resize(frame, width=1280)  # Resize the frame for consistency

        # get info from the frame
        out = rtfprediction.get_face_info(frame)
        # draw bounding box on the image
        res_img = rtfprediction.bounding_box(out,frame)

        end_time = time.time() - star_time    # Calculate elapsed time
        FPS = 1/end_time
        # Display FPS on the frame
        cv2.putText(res_img,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('Face info',res_img)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break