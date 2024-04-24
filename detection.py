import argparse
import numpy as np
import pandas as pd
from models.shallow import IF, LOF, OCSVM
from utils.params import Params
from trainer.ae import  TrainerAE
from copy import deepcopy
from trainer.base import TrainerBaseShallow
from trainer.dagmm import TrainerDAGMM
from trainer.dsebm import TrainerDSEBM
from trainer.alad import TrainerALAD
from trainer.svdd import TrainerSVDD
from utils.utils import estimate_optimal_threshold, compute_metrics, compute_metrics_binary
from models.svdd import DeepSVDD
from models.alad import ALAD
from models.dsebm import  DSEBM
from models.ae import AEDetecting
from models.dagmm import DAGMM
import logging

logging.basicConfig(filename='logs/robustness.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


outputs = "outputs/"

model_trainer_map = {
    "alad": (TrainerALAD, ALAD),
    "dagmm": (TrainerDAGMM, DAGMM),
    "dsebm": (TrainerDSEBM, DSEBM),
    "if": (TrainerBaseShallow, IF),
    "lof": (TrainerBaseShallow, LOF),
    "ocsvm": (TrainerBaseShallow, OCSVM),
    "ae": (TrainerAE, AEDetecting),
    "svdd": (TrainerSVDD, DeepSVDD),
}

def resolve_model_trainer(model_name):
    t, m = model_trainer_map.get(model_name, None)
    assert t, "Model %s not found" % model_name
    return t, m

def get_contamination(key, model_name):
    if "bopeto" in key:
        model_name_ = "Bopeto_" + model_name
    else:
        model_name_ = model_name

    contamination = 0
    splits = key.split("_")
    if len(splits) >= 3:
        cont = splits[-1]
        contamination = float("." + cont.split(".")[1])
    return contamination, model_name_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OoD detection",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=64)
    parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=20)
    parser.add_argument('-n', '--num_workers', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--dataset', type=str, default='nsl', help='data set name')
    parser.add_argument('--model', type=str, default='AE', help='model name')

    #DaGMM
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=194)
    parser.add_argument('--model_save_step', type=int, default=194)

    # Jeff
    parser.add_argument(
        '--n-runs',
        help='number of runs of the experiment',
        type=int,
        default=1
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help="The learning rate"
    )
    parser.add_argument(
        '--test_pct',
        type=float,
        default=0.5,
        help="The percentage of normal data used for training"
    )

    parser.add_argument(
        '--patience',
        type=float,
        default=10,
        help='Early stopping patience')

    parser.add_argument(
        "--pct",
        type=float,
        default=1.0,
        help="Percentage of original data to keep"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of validation set from the training set"
    )

    parser.add_argument(
        "--hold_out",
        type=float,
        default=0.0,
        help="Percentage of anomalous data to holdout for possible contamination of the training set"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Anomaly ratio within training set"
    )

    parser.add_argument('--drop_lastbatch', dest='drop_lastbatch', action='store_true')
    parser.add_argument('--no-drop_lastbatch', dest='drop_lastbatch', action='store_false')
    parser.set_defaults(drop_lastbatch=False)

    # Robustness parameters
    parser.add_argument('--rob', dest='rob', action='store_true')
    parser.add_argument('--no-rob', dest='rob', action='store_false')
    parser.set_defaults(rob=False)

    parser.add_argument('--rob-sup', dest='rob_sup', action='store_true')
    parser.add_argument('--no-rob-sup', dest='rob_sup', action='store_false')
    parser.set_defaults(rob_sup=False)

    parser.add_argument('--rob-reg', dest='rob_reg', action='store_true')
    parser.add_argument('--no-rob-reg', dest='rob_reg', action='store_false')
    parser.set_defaults(rob_reg=False)

    parser.add_argument('--eval-test', dest='eval_test', action='store_true')
    parser.add_argument('--no-eval-test', dest='eval_test', action='store_false')
    parser.set_defaults(eval_test=False)

    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true')
    parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false')
    parser.set_defaults(early_stopping=True)

    parser.add_argument(
        '--rob_method',
        type=str,
        choices=['refine', 'loe', 'our', 'sup'],
        default='daecd',
        help='methods used, either blind, refine, loe, daecd'
    )

    parser.add_argument(
        '--alpha-off-set',
        type=int,
        default=0.0,
        help='values between o and 1 used to offset the true value of the contamination ratio'
    )

    parser.add_argument(
        '--reg_n',
        type=float,
        default=1e-3,
        help='regulizer factor for the latent representation norm  '
    )

    parser.add_argument(
        '--reg_a',
        type=float,
        default=1e-3,
        help='regulizer factor for the anomalies loss'
    )

    parser.add_argument(
        '--num_clusters',
        type=int,
        default=3,
        help='number of clusters'
    )

    args = parser.parse_args()

    configs = vars(args)

    gmm_k = configs['gmm_k']
    lambda_energy = configs['lambda_energy']
    lambda_cov_diag = configs['lambda_cov_diag']
    log_step = configs['log_step']
    sample_step = configs['sample_step']
    model_save_step = configs['model_save_step']

    params = Params()
    params.patience = configs['patience']
    params.learning_rate = configs['learning_rate']
    params.weight_decay = configs['weight_decay']
    params.batch_size = configs['batch_size']
    params.num_workers = configs['num_workers']
    params.alpha = configs['alpha']
    params.epochs = configs['epochs']
    params.model_name = configs['model']
    params.early_stopping = configs['early_stopping']
    params.dataset_name = configs['dataset']
    data = np.load("detection/"+params.dataset_name+".npz", allow_pickle=True)
    keys = list(data.keys())
    filter_keys = list(filter(lambda s: "train" in s, keys))
    params.test = data[params.dataset_name+"_test"]
    params.val = data[params.dataset_name + "_val"]
    params.in_features = params.val.shape[1]-1
    performances = pd.DataFrame([], columns=["dataset", "contamination", "model", "accuracy","precision", "recall", "f1"])
    params.data = data[filter_keys[0]]
    tr, mo = resolve_model_trainer(params.model_name)
    mo = mo(params)
    params.model = mo
    tr = tr(params)
    n_cases = len(filter_keys)
    for i, key in enumerate(filter_keys):
        try:
            print("{}/{}: training on {}".format(i+1, n_cases, key))
            model = deepcopy(mo)
            model.params.data = data[key]
            trainer = deepcopy(tr)
            trainer.params.data = data[key]
            trainer.params.model = model
            contamination, model_name_ = get_contamination(key, params.model_name)
            trainer.train()
        except RuntimeError as e:
            logging.error(
                "OoD detection on {} with {} and contamination rate {} unfinished caused by {} ...".format(params.dataset_name,
                                                                                               params.model_name,
                                                                                               contamination, e))
        except Exception as e:
            logging.error(
                "Error for OoD detection on {} with {} and contamination rate {}: {} ...".format(
                    params.dataset_name,
                    params.model_name,
                    contamination, e))
        finally:
            if trainer.name == "shallow":
                X, y_test = params.test[:, :-1], params.test[:, -1]
                y_pred = trainer.test(X)
                metrics = compute_metrics_binary(y_pred, y_test, pos_label=1)
                perf = [params.dataset_name, contamination, model_name_, metrics[0], metrics[1], metrics[2], metrics[3]]
                performances.loc[len(performances)] = perf
                print("performance on", key, metrics[:4])
            else:
                y_val, score_val = trainer.test(params.val)
                y_test, score_test = trainer.test(params.test)
                threshold = estimate_optimal_threshold(score_val, y_val, pos_label=1, nq=100)
                threshold = threshold["Thresh_star"]
                metrics = compute_metrics(score_test, y_test, threshold, pos_label=1)
                perf = [params.dataset_name, contamination, model_name_, metrics[0], metrics[1], metrics[2], metrics[3]]
                performances.loc[len(performances)] = perf
                print("performance on", key, metrics[:4])

    performances.to_csv("outputs/performances_"+params.dataset_name+"_"+params.model_name+".csv", header=True, index=False)











