
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def generate_session_report(csv_file="stress_history.csv"):
    csv_path = Path(csv_file)

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path.resolve()}")
        return

    frames = []
    scores = []
    timestamps = []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            frames.append(int(row["Frame"]))
            scores.append(float(row["Experimental Score"]))
            timestamps.append(row["Timestamp"])

    if not scores:
        print("No readings found.")
        return

    average_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    max_time = timestamps[scores.index(max_score)]

    print("\n========== SESSION REPORT ==========")
    print(f"Total readings     : {len(scores)}")
    print(f"Average score      : {average_score:.2f}/100")
    print(f"Maximum score      : {max_score:.2f}/100")
    print(f"Minimum score      : {min_score:.2f}/100")
    print(f"Maximum score time : {max_time}")
    print("====================================")

    plt.figure(figsize=(10, 5))
    plt.plot(frames, scores, label="Experimental score")
    plt.title("Stress Computation - Session History")
    plt.xlabel("Frame")
    plt.ylabel("Experimental score (0-100)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = Path("session_report.png")
    plt.savefig(output_path)
    plt.show()

    print(f"Graph saved to: {output_path.resolve()}")
