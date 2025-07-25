from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
import subprocess

# Initialise Flask app
app = Flask(__name__, static_folder="frontend_face_info/static", static_url_path="/static")
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory("frontend_face_info", "index.html")

# API endpoint to trigger the prediction based on input type
@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON request data to web server
    input_type = request.json.get('input_type')
    image_path = request.json.get('image_path', '')

    if input_type == 'webcam':
        subprocess.Popen(["python", "Face_info.py", "--input", "webcam"])
    elif input_type == 'video':
        subprocess.Popen(["python", "Face_info.py", "--input", "video"])
    elif input_type == 'image' and image_path:
        subprocess.Popen(["python", "Face_info.py", "--input", "image", "--path_im", image_path])
    else:
        return jsonify({"status": "error", "message": "Invalid input type"}), 400

    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(port=5050)
