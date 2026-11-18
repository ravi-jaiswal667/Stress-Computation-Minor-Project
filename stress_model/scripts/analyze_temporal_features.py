
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR / "stress_model" / "data" /
    "processed" / "video_emotion_features.csv"
)

OUTPUT_FILE = (
    ROOT_DIR / "stress_model" / "data" /
    "processed" / "temporal_features.csv"
)

def main():
    df = pd.read_csv(INPUT_FILE)

    emotions = [
        "angry", "disgust", "fearful", "happy",
        "neutral", "sad", "surprised"
    ]

    probability_columns = [
        f"{emotion}_probability" for emotion in emotions
    ]

    features = {}

    # Average emotion probabilities
    for column in probability_columns:
        features[f"mean_{column}"] = df[column].mean()

    # Variability of emotion probabilities
    for column in probability_columns:
        features[f"std_{column}"] = df[column].std()

    # Emotion frequency
    counts = df["predicted_emotion"].value_counts(normalize=True)

    for emotion in emotions:
        features[f"{emotion}_frequency"] = counts.get(emotion, 0)

    # Number of expression changes
    features["emotion_changes"] = (
        df["predicted_emotion"].ne(
            df["predicted_emotion"].shift()
        ).sum() - 1
    )

    features["total_frames"] = len(df)
    features["video_duration_seconds"] = (
        df["timestamp_seconds"].max()
    )

    result = pd.DataFrame([features])
    result.to_csv(OUTPUT_FILE, index=False)

    print("Temporal feature extraction successful!")
    print(result.T)
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
