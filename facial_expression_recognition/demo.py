import sys
import argparse
import copy
import datetime

import numpy as np
import cv2 as cv
# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# Allow imports from the project root
sys.path.insert(0, str(ROOT_DIR))

from stress_estimator import StressEstimator
from stress_graph import StressGraph

sys.path.insert(
    0, str(ROOT_DIR / "face_detection_yunet")
)

from facial_fer_model import FacialExpressionRecog
from yunet import YuNet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--input', '-i', type=str,
                    help='Path to the input image. Omit for using default camera.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default=str(
        BASE_DIR /
        "facial_expression_recognition_mobilefacenet_2022july.onnx"
    ),
                    help='Path to the facial expression recognition model.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true',
                    help='Specify to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Specify to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()


def visualize(image, det_res, fer_res, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()

    emotion_names = [
        "Angry", "Disgust", "Fearful", "Happy",
        "Neutral", "Sad", "Surprised"
    ]

    for det, result in zip(det_res, fer_res):
        predicted_index, probabilities = result
        bbox = det[0:4].astype(np.int32)

        emotion = FacialExpressionRecog.getDesc(predicted_index)
        confidence = float(probabilities[predicted_index]) * 100

        cv.rectangle(
            output,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            box_color,
            2
        )

        cv.putText(
            output,
            f"{emotion}: {confidence:.1f}%",
            (bbox[0], max(20, bbox[1] - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_color,
            2
        )

        # Display all seven emotion scores
        y = 70
        for name, score in zip(emotion_names, probabilities):
            text = f"{name}: {float(score) * 100:.1f}%"

            cv.putText(
                output,
                text,
                (20, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1
            )
            y += 25

    return output



def process(detect_model, fer_model, frame):
    h, w, _ = frame.shape
    detect_model.setInputSize([w, h])

    dets = detect_model.infer(frame)

    if dets is None or len(dets) == 0:
        return False, None, None

    # Store each face's predicted emotion and scores
    fer_res = []

    for face_points in dets:
        result = fer_model.infer(frame, face_points[:-1])
        fer_res.append(result)

    return True, dets, fer_res



if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    detect_model = YuNet(
    modelPath=str(
        ROOT_DIR / "face_detection_yunet" /
        "face_detection_yunet_2023mar.onnx"
    )
)

    fer_model = FacialExpressionRecog(modelPath=args.model,
                                      backendId=backend_id,
                                      targetId=target_id)

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Get detection and fer results
        status, dets, fer_res = process(detect_model, fer_model, image)

        if status:
            # Draw results on the input image
            image = visualize(image, dets, fer_res)

        # Save results
        if args.save:
            cv.imwrite('result.jpg', image)
            print('Results saved to result.jpg\n')

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
            cv.waitKey(0)
    else:  # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        estimator = StressEstimator()
        graph = StressGraph()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Get detection and fer results
            status, dets, fer_res = process(detect_model, fer_model, frame)

            
            if status:
                frame = visualize(frame, dets, fer_res)

                # Estimate stress from the first detected face
                predicted_index, probabilities = fer_res[0]
                emotion = FacialExpressionRecog.getDesc(predicted_index)
                confidence = float(probabilities[predicted_index])

                stress_score, stress_level = estimator.estimate(
                        emotion, confidence
                    )
                graph.update(stress_score)

                cv.putText(
                    frame,
                    f"Experimental score: {stress_score:.1f}/100",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                cv.putText(
                    frame,
                    f"Score category: {stress_level}",
                    (20, 75),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
            else:
                cv.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )
            # Display every frame, even if no face is detected
            cv.imshow('FER Demo', frame)
            cv.moveWindow('FER Demo', 20, 50)

        # Release resources after the loop ends
        graph.save_history()
        graph.print_session_summary()

        cap.release()
        cv.destroyAllWindows()
        graph.close()
