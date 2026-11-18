
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import sys

import cv2 as cv
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from stress_estimator import StressEstimator
import database

ROOT_DIR = Path(__file__).resolve().parent

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

# Initialize SQLite tables
database.initialize_database()

# Load existing models once
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

# Keep active estimators in memory.
# Actual session records are stored in SQLite.
sessions = {}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Stress Computation backend is running"
    })


# START A NEW SESSION
@app.route("/api/session/start", methods=["POST"])
def start_session():
    session_id = str(uuid4())

    started_at = database.create_session(session_id)

    sessions[session_id] = {
        "estimator": StressEstimator(window_size=30),
        "started_at": started_at
    }

    return jsonify({
        "message": "Session started successfully",
        "session_id": session_id
    })


# ANALYZE IMAGE OR WEBCAM FRAME
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "frame" not in request.files:
        return jsonify({"error": "No image frame provided"}), 400

    session_id = request.form.get("session_id")

    if session_id:
        session = database.get_session(session_id)

        if not session:
            return jsonify({"error": "Session not found"}), 404

        if session["status"] != "active":
            return jsonify({"error": "Session is already completed"}), 400

        if session_id not in sessions:
            return jsonify({
                "error": "Session cannot continue after backend restart"
            }), 400

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

    face_points = np.asarray(faces[0], dtype=np.float32)

    predicted_index, probabilities = emotion_model.infer(
        frame, face_points[:-1]
    )

    emotion = EMOTIONS[predicted_index]

    scores = {
        name: round(float(prob) * 100, 2)
        for name, prob in zip(EMOTIONS, probabilities)
    }

    if len(probabilities) != len(EMOTIONS):
        return jsonify({
            "error": "Unexpected number of emotion probabilities"
        }), 500

    confidence = float(probabilities[predicted_index])

    estimator = (
        sessions[session_id]["estimator"]
        if session_id
        else stress_estimator
    )

    instant_score, score, category = estimator.estimate(scores)

    if session_id:
        reading = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "emotion": emotion,
            "confidence": round(confidence * 100, 2),
            "instantaneous_score": instant_score,
            "experimental_score": score,
            "score_category": category,
            "emotion_scores": scores
        }
        # Save every reading immediately to SQLite
        database.save_reading(session_id, reading)

    return jsonify({
        "face_detected": True,
        "emotion": emotion,
        "confidence": round(confidence * 100, 2),
        "emotion_scores": scores,

        "instantaneous_score": instant_score,
        "experimental_score": score,

        "score_category": category,
        "message": "Experimental expression-based score; not a clinical measurement."
    })


# GET SESSION HISTORY
@app.route("/api/session/<session_id>/history", methods=["GET"])
def session_history(session_id):
    session = database.get_session(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({
        "session_id": session_id,
        "readings": database.get_readings(session_id)
    })


# GET SESSION SUMMARY
@app.route("/api/session/<session_id>/summary", methods=["GET"])
def session_summary(session_id):
    session = database.get_session(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    # A summary request happens after the webcam is stopped
    if session["status"] == "active":
        database.finish_session(session_id)
        sessions.pop(session_id, None)

    readings = database.get_readings(session_id)

    if not readings:
        return jsonify({
            "session_id": session_id,
            "total_readings": 0,
            "average_score": 0,
            "maximum_score": 0,
            "minimum_score": 0,
            "latest_category": "No readings",
            "started_at": session["started_at"],
            "ended_at": database.get_session(session_id)["ended_at"]
        })

    scores = [
        reading["experimental_score"]
        for reading in readings
    ]

    return jsonify({
        "session_id": session_id,
        "total_readings": len(readings),
        "average_score": round(sum(scores) / len(scores), 2),
        "maximum_score": round(max(scores), 2),
        "minimum_score": round(min(scores), 2),
        "latest_category": readings[-1]["score_category"],
        "started_at": session["started_at"],
        "ended_at": database.get_session(session_id)["ended_at"]
    })


# GET ALL PREVIOUS SESSIONS
@app.route("/api/sessions", methods=["GET"])
def all_sessions():
    return jsonify({
        "sessions": database.get_all_sessions()
    })


# DELETE A SESSION
@app.route("/api/session/<session_id>", methods=["DELETE"])
def remove_session(session_id):
    session = database.get_session(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    sessions.pop(session_id, None)
    database.delete_session(session_id)

    return jsonify({
        "message": "Session deleted successfully"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
