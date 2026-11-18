
from pathlib import Path
import sys
import cv2 as cv
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT_DIR / "face_detection_yunet"))
sys.path.insert(0, str(ROOT_DIR / "facial_expression_recognition"))

from yunet import YuNet
from facial_fer_model import FacialExpressionRecog

EMOTIONS = [
    "angry", "disgust", "fearful", "happy",
    "neutral", "sad", "surprised"
]


def main():
    video_path = ROOT_DIR / "stress_model" / "data" / "raw" / "webcam_sample.mp4"

    if not video_path.exists():
        print("Video not found:", video_path)
        return

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

    cap = cv.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    frame_interval = max(1, round(fps / 2))  # Sample 2 frames per second

    rows = []
    frame_index = 0

    print("Processing video...")

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        height, width = frame.shape[:2]
        face_model.setInputSize([width, height])

        faces = face_model.infer(frame)

        if faces is not None and len(faces) > 0:
            face = np.asarray(faces[0], dtype=np.float32)

            predicted_index, probabilities = emotion_model.infer(
                frame, face[:-1]
            )

            x, y, box_width, box_height = face[:4]

            row = {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / fps, 2),
                "predicted_emotion": EMOTIONS[predicted_index],
                "detection_confidence": float(face[-1]),
                "face_width_ratio": float(box_width / width),
                "face_height_ratio": float(box_height / height),
                "face_area_ratio": float(
                    (box_width * box_height) / (width * height)
                )
            }

            for emotion, probability in zip(EMOTIONS, probabilities):
                row[f"{emotion}_probability"] = float(probability)

            rows.append(row)

        frame_index += 1

    cap.release()

    if not rows:
        print("No faces detected in the video.")
        return

    output_dir = ROOT_DIR / "stress_model" / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "video_emotion_features.csv"

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Frames processed:", len(rows))
    print("Video duration:", round(frame_index / fps, 2), "seconds")
    print("\nEmotion distribution:")
    print(df["predicted_emotion"].value_counts())
    print("\nCSV saved at:", output_file)


if __name__ == "__main__":
    main()
