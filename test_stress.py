
from stress_estimator import StressEstimator

test_cases = {
    "Mostly Happy": {
        "angry": 2, "disgust": 1, "fearful": 2,
        "happy": 85, "neutral": 5, "sad": 3,
        "surprised": 2
    },

    "Mostly Neutral": {
        "angry": 2, "disgust": 1, "fearful": 2,
        "happy": 3, "neutral": 85, "sad": 5,
        "surprised": 2
    },

    "Mostly Fearful": {
        "angry": 5, "disgust": 2, "fearful": 80,
        "happy": 2, "neutral": 2, "sad": 5,
        "surprised": 4
    },

    "Mostly Angry": {
        "angry": 85, "disgust": 3, "fearful": 3,
        "happy": 2, "neutral": 2, "sad": 3,
        "surprised": 2
    }
}

for name, probabilities in test_cases.items():
    estimator = StressEstimator(window_size=1)
    instant_score, smoothed_score, level = estimator.estimate(probabilities)

    print(f"Instantaneous Score: {instant_score}")
    print(f"Smoothed Score: {smoothed_score}")
    print(f"Category: {level}")
