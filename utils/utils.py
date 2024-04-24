from sklearn import metrics as sk_metrics
import numpy as np
from trainer.ae import Trainer
from synthetic.generation import JMF, FGM

class Utils:
    def __init__(self, params):
        self.params = params

    def get_reconstruction_errors(self):
        trainer = Trainer(self.params)
        errors = trainer.train(True)
        return errors

    def initial_train(self):
        trainer = Trainer(self.params)
        errors = trainer.train(False)
        return errors

    def generate_synthetic_data(self):
        if self.params.synthetic == "JMF":
            generator = JMF(self.params)
        else:
            generator = FGM(self.params)
        return generator.generate()

    def contaminate(self, in_dist, oo_dist):
        data = np.vstack((in_dist, oo_dist))
        np.random.shuffle(data)
        self.params.rate = oo_dist.shape[0]/(in_dist.shape[0]+oo_dist.shape[0])
        return data

    def optimal_threshold(self, model):
        pass

def compute_metrics(val_score, y_val, thresh, pos_label=1):
    y_pred = (val_score >= thresh).astype(int)
    y_true = y_val.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, val_score)
    roc = sk_metrics.roc_auc_score(y_true, val_score)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm

def compute_metrics_binary(y_pred, y_val, pos_label=1):
    y_true = y_val.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, cm
def estimate_optimal_threshold(val_score, y_val, pos_label=1, nq=100):
    ratio = 100 * sum(y_val == 0) / len(y_val)
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(val_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(val_score, y_val, thresh, pos_label)

        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }
def contamination(data):
    in_dist = len(data[data[:, -1] == 0])
    oo_dist = len(data[data[:, -1] == 1])
    n = in_dist + oo_dist
    return n, oo_dist/n

def compute_metrics(test_score, y_test, thresh, pos_label=1):
    y_pred = (test_score >= thresh).astype(int)
    y_true = y_test.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, test_score)
    roc = sk_metrics.roc_auc_score(y_true, test_score, max_fpr=1e-2)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm

def _estimate_threshold_metrics(test_score, y_test, pos_label=1, nq=100, optimal=True):
    ratio = 100 * sum(y_test != pos_label) / len(y_test)

    if not optimal:
        thresh = np.percentile(test_score, ratio)
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(test_score, y_test, thresh, pos_label)

        return accuracy, precision, recall, f_score, roc, avgpr

    print(f"Ratio of normal data:{round(ratio,2)}%")
    q = np.linspace(max(ratio - 5, .1), min(ratio + 5, 100), nq)
    thresholds = np.percentile(test_score, q)

    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # Prediction using the threshold value
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(test_score, y_test, thresh, pos_label)

        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }







