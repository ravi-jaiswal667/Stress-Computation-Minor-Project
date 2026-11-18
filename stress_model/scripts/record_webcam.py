
from pathlib import Path
import cv2 as cv

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT_DIR / "stress_model" / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "webcam_sample.mp4"

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

fps = 20.0
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*"mp4v")

writer = cv.VideoWriter(
    str(OUTPUT_FILE),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    print("Error: Could not create video file.")
    cap.release()
    exit()

print("Recording started!")
print("Press Q to stop recording.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    writer.write(frame)

    cv.putText(
        frame,
        "RECORDING - Press Q to stop",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv.imshow("Webcam Recorder", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv.destroyAllWindows()

print("\nRecording saved successfully!")
print("File:", OUTPUT_FILE)
