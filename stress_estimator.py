
from collections import deque


class StressEstimator:
    def __init__(self, window_size=30):
        self.emotion_history = deque(maxlen=window_size)

    def estimate(self, emotion, confidence):
        """
        Experimental expression-based score.
        This is not a validated stress measurement.
        """

        stress_related_emotions = {
            "angry",
            "disgust",
            "fearful",
            "sad"
        }

        score = 100.0 if emotion in stress_related_emotions else 0.0

        self.emotion_history.append(score)

        smoothed_score = (
            sum(self.emotion_history) / len(self.emotion_history)
        )

        if smoothed_score < 30:
            level = "Low"
        elif smoothed_score < 60:
            level = "Moderate"
        else:
            level = "High"

        return round(smoothed_score, 1), level
