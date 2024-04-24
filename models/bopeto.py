from plot.filtering import plot_segmented_one_line
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np


class BOPETO:
    def __init__(self, params):
        self.params = params

    def sdc(self):
        return np.std(np.diff(self.params.dynamics, axis=1), axis=1)

    def refine(self, to_plot=False):
        dynamics_scores = self.sdc()
        n = len(dynamics_scores)
        target = ["synthetic" if self.params.dynamics[i, -1] == 2 else "training" for i in range(n)]
        db = pd.DataFrame(data={'sample': range(n), self.params.metric: dynamics_scores, "class": target})
        values = db[self.params.metric].values.reshape(-1, 1)
        detector = IsolationForest(n_estimators=50, random_state=42)
        y_pred = detector.fit_predict(values)
        anomaly_scores = detector.decision_function(values)
        ood = anomaly_scores[y_pred == -1]
        in_ = anomaly_scores[y_pred == 1]
        threshold = (np.max(ood) + np.min(in_)) / 2
        threshold = np.percentile(ood, np.random.randint(60, 70, 1)[0])
        y_pred = anomaly_scores >= threshold
        indices = list(db[(y_pred==1) & (db["class"] != "synthetic")].index)
        if to_plot:
            plot_segmented_one_line(self.params.id, db, threshold, dynamics_scores)
        return indices

