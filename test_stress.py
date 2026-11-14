
from stress_estimator import StressEstimator

estimator = StressEstimator()

print(estimator.estimate("neutral", 0.95))
print(estimator.estimate("angry", 0.90))
print(estimator.estimate("angry", 0.92))
print(estimator.estimate("happy", 0.98))
