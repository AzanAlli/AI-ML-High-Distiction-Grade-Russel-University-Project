import rtfprediction
import cv2
import time
import imutils
import argparse

parser = argparse.ArgumentParser(description="Face Info")
parser.add_argument('--input', type=str, default='webcam', help="webcam, video, or image")
parser.add_argument('--path_im', type=str, help="path of image (used only if input is 'image')")
args = vars(parser.parse_args())

type_input = args['input']

# ----------------------------- image -----------------------------
if type_input == 'image':
    frame = cv2.imread(args['path_im'])
    if frame is None:
        print(f"Error: Could not load image from {args['path_im']}")
        exit(1)

    out = rtfprediction.get_face_info(frame)
    res_img = rtfprediction.bounding_box(out, frame)
    cv2.imshow('Face info - Press q to quit', res_img)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------------------- webcam -----------------------------
elif type_input == 'webcam':
    cam = cv2.VideoCapture(1)  # Use 0 if 1 doesn't work for your webcam
    if not cam.isOpened():
        print("Error: Could not access webcam.")
        exit(1)

    while True:
        start_time = time.time()
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=1280)
        out = rtfprediction.get_face_info(frame)
        res_img = rtfprediction.bounding_box(out, frame)

        fps = 1 / (time.time() - start_time)
        cv2.putText(res_img, f"FPS: {round(fps, 2)}", (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face info - Press q to quit', res_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------------------- video -----------------------------
elif type_input == 'video':
    cam = cv2.VideoCapture("videofile.mp4")
    if not cam.isOpened():
        print("Error: Could not open video file.")
        exit(1)

    while True:
        start_time = time.time()
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=1280)
        out = rtfprediction.get_face_info(frame)
        res_img = rtfprediction.bounding_box(out, frame)

        fps = 1 / (time.time() - start_time)
        cv2.putText(res_img, f"FPS: {round(fps, 2)}", (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face info - Press q to quit', res_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

else:
    print("Invalid input type. Use 'webcam', 'video', or 'image'.")
