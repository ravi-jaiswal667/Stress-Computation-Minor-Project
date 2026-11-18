
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
    image_path = ROOT_DIR / "test_face.jpg"

    if not image_path.exists():
        print("Image not found:", image_path)
        return

    # Initialize existing models
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

    image = cv.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        return

    height, width = image.shape[:2]
    face_model.setInputSize([width, height])

    faces = face_model.infer(image)

    if faces is None or len(faces) == 0:
        print("No face detected.")
        return

    rows = []

    for face_number, face in enumerate(faces):
        face = np.asarray(face, dtype=np.float32)

        # Remove YuNet's final detection-confidence value
        predicted_index, probabilities = emotion_model.infer(
            image, face[:-1]
        )

        x, y, box_width, box_height = face[:4]

        row = {
            "image_name": image_path.name,
            "face_number": face_number,
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

    output_dir = ROOT_DIR / "stress_model" / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "emotion_features.csv"

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False
    )

    print("\nFeature extraction successful!")
    print("Faces processed:", len(rows))
    print("Predicted emotion:", rows[0]["predicted_emotion"])
    print("CSV saved at:", output_file)

if __name__ == "__main__":
    main()
