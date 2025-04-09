# REAL TIME FACE PREDICTION - PREDICTING AGE, GENDER, RACE AND EMOTION USING CONVOLUTIONAL NEURAL NETWORKS

## Information about this repository

This repository contains a fully functioning system that uses **Convolutional Neural Networks (CNNs)** for real-time facial attribute prediction. It predicts **age, gender, race, emotion,** and performs **face recognition** from webcam, video files, or images.

This is the repository that you are going to use **individually** for developing your project. Please use the resources provided in the module to learn about **plagiarism** and how plagiarism awareness can foster your learning.

Regarding the use of this repository, once a feature (or part of it) is developed and **working** or parts of your system are integrated and **working**, define a commit and push it to the remote repository. You may find yourself making a commit after a productive hour of work (or even after 20 minutes!), for example. Choose commit message wisely and be concise.

Please choose the structure of the contents of this repository that suits the needs of your project but do indicate in this file where the main software artefacts are located.

## External Model Weights

Due to GitLab's 100MB file size limit, the following model weights are not included in the repository:

- age_model_weights.h5
- arcface_weights.h5
- gender_model_weights.h5
- race_model_single_batch.h5
- vgg_face_weights.h5

Please download them from the following OneDrive folder:
[FinalProjectModelWeights](https://uniofleicester-my.sharepoint.com/:f:/r/personal/sa1021_student_le_ac_uk/Documents/FinalProjectModelWeights?csf=1&web=1&e=LE5oGW)

Once downloaded, place them in:
deepface_backup/weights/

Refer to `MODEL_DOWNLOAD.txt` for instructions as well.

---

## 1) How to Use the Software

This system supports three modes of prediction:
- Real-time **webcam** detection
- **Video** file prediction (e.g. `demo.mp4`)
- Static **image** prediction (e.g. JPG files in `images_db/`)

### Run the Backend Directly

To run the backend manually without frontend:

```bash
python Face_info.py --input webcam        # Real-time webcam detection
python Face_info.py --input video         # Prediction on demo.mp4
python Face_info.py --input image --path_im images_db/[filename].jpg  # Prediction on an image
```

Press `q` in any prediction window to close it and return to the terminal.

---

### 2) Launch the Frontend Website

1. Start the Flask backend API:

```bash
python app.py
```

2. Open your browser and go to:

```
http://127.0.0.1:5050/
```
or
```
index.html
```

3. Use the interface buttons:

Buttons:

REAL TIME DETECTION: Opens webcam window for live face analysis              
VIDEO PREDICTION: Starts prediction on `demo.mp4` (make sure this file exists)
PREDICT IMAGE: Lets you select a JPG image from your `images_db` directory

---

### 3) Notes on Usage

- Only one prediction mode can run at a time. If switching modes, close the previous prediction window first.
- To use image prediction, ensure your `.jpg` files are inside the `images_db/` folder.
- Press `q` in any prediction window to safely close it.
- Press Ctrl + C in the terminal window. This will safely terminate the Flask backend or stop any Python process started through the frontend.

---

## Main Directories & Files

- `Face_info.py`: Main backend execution file
- `app.py`: Flask backend API to interface with frontend
- `frontend/index.html`: Main web interface
- `rtfprediction.py`: Core logic combining all models
- `deepface_backup/weights/`: Folder containing required model weight files
- `images_db/`: Folder where your test `.jpg` images should be placed

---

**please note: The key `q` does take some seconds to quit when doing real time facial predictions (5-15sec) and run time for all predictions does take 3-5 seconds**