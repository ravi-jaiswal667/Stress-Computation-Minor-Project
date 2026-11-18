
from collections import deque


class StressEstimator:
    def __init__(self, window_size=30):
        self.stress_history = deque(maxlen=window_size)

        self.emotion_weights = {
            "happy": 0,
            "neutral": 10,
            "surprised": 25,
            "sad": 60,
            "disgust": 65,
            "fearful": 80,
            "angry": 85
        }

    def estimate(self, emotion_scores):
        """
        Calculate instantaneous and smoothed
        expression-based experimental scores.
        """

        total_probability = sum(emotion_scores.values())

        if total_probability <= 0:
            return 0.0, 0.0, "Insufficient Data"

        normalized_scores = {
            emotion: probability / total_probability
            for emotion, probability in emotion_scores.items()
        }

        # 1. Calculate score for the current frame
        instant_score = sum(
            normalized_scores.get(emotion, 0) * weight
            for emotion, weight in self.emotion_weights.items()
        )

        # 2. Store current score for smoothing
        self.stress_history.append(instant_score)

        # 3. Calculate average of recent frames
        smoothed_score = (
            sum(self.stress_history) / len(self.stress_history)
        )

        # 4. Categorize using smoothed score
        if smoothed_score < 20:
            level = "Minimal"
        elif smoothed_score < 40:
            level = "Low"
        elif smoothed_score < 60:
            level = "Moderate"
        elif smoothed_score < 80:
            level = "High"
        else:
            level = "Severe"

        return (
            round(instant_score, 1),
            round(smoothed_score, 1),
            level
        )
