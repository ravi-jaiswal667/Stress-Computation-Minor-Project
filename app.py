
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from stress_estimator import StressEstimator
ROOT_DIR = Path(__file__).resolve().parent

# Add existing model folders to Python's import path
sys.path.insert(0, str(ROOT_DIR / "face_detection_yunet"))
sys.path.insert(0, str(ROOT_DIR / "facial_expression_recognition"))

from yunet import YuNet
from facial_fer_model import FacialExpressionRecog

app = Flask(__name__)
CORS(app)

EMOTIONS = [
    "angry", "disgust", "fearful", "happy",
    "neutral", "sad", "surprised"
]

# Load models once when the backend starts
face_model = YuNet(
    modelPath=str(
        ROOT_DIR / "face_detection_yunet"
        / "face_detection_yunet_2023mar.onnx"
    )
)

emotion_model = FacialExpressionRecog(
    modelPath=str(
        ROOT_DIR / "facial_expression_recognition"
        / "facial_expression_recognition_mobilefacenet_2022july.onnx"
    ),
    backendId=cv.dnn.DNN_BACKEND_OPENCV,
    targetId=cv.dnn.DNN_TARGET_CPU
)

stress_estimator = StressEstimator(window_size=30)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Stress Computation backend is running"
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "frame" not in request.files:
        return jsonify({"error": "No image frame provided"}), 400

    image_bytes = request.files["frame"].read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv.imdecode(image_array, cv.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    height, width = frame.shape[:2]
    face_model.setInputSize([width, height])
    faces = face_model.infer(frame)

    if faces is None or len(faces) == 0:
        return jsonify({"face_detected": False})

    # Use the first detected face
    face = np.asarray(faces[0], dtype=np.float32)

    # Keep bounding box and 10 landmark coordinates.
    # Exclude the final confidence value.
    
    # Use the first detected face
    face_points = np.asarray(faces[0], dtype=np.float32)

    # Exclude the final confidence value, as in demo.py
    face_for_fer = face_points[:-1]

    predicted_index, probabilities = emotion_model.infer(
        frame, face_for_fer
    )

    emotion = EMOTIONS[predicted_index]

    scores = {
        name: round(float(prob) * 100, 2)
        for name, prob in zip(EMOTIONS, probabilities)
    }

    experimental_score, score_category = stress_estimator.estimate(
        emotion, float(probabilities[predicted_index])
    )

    return jsonify({
        "face_detected": True,
        "emotion": emotion,
        "emotion_scores": scores,
        "experimental_score": experimental_score,
        "score_category": score_category,
        "message": "Experimental expression-based score; not a clinical measurement."
    })



if __name__ == "__main__":
    app.run(debug=True, port=5000)
