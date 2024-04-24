import numpy as np
from models.ae import AECleaning
import torch

class Params:

    def __init__(self):
        self.i = 0
        self.rate = 0
        self.patience = 10
        self.id = None
        self.batch_size = None
        self.learning_rate = None
        self.weight_decay = None
        self.alpha = None
        self.gamma = None
        self.epochs = None
        self.dataset_name = None
        self.metric = None
        self.data = None
        self.synthetic = None
        self.model_name = None
        self.num_workers = None
        self.dynamics = None
        self.fragment = None
        self.val = None
        self.test = None
        self.val_scores = None
        self.test_scores = None
        self.y_pred = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lambda_1 = 0.005
        self.lambda_2 =  0.1
        self.reg_covar = 0.01 #1e-12
        self.n_jobs_dataloader = 1
        self.early_stopping = True
        self.score_metric = "reconstruction"
        self.ae_latent_dim = 1
        self.in_features = None
        self.D = 8
        self.c = .8
        self.R = None
        self.model = None
        self.dropout = 0

    def set_model(self):
        self.id = self.dataset_name+"_"+self.synthetic + "_" +self.metric+"_ae_rate_"+str(self.rate)
        self.model = AECleaning(self)
        self.model.load()
        self.model.name = self.id

    def init_model(self, load=False):
        self.model = AECleaning(self)
        if load:
            self.model.load()
        self.model.save()


    def update_data(self, synthetic):
        y = [2]*len(synthetic)
        synthetic = np.column_stack((synthetic, y))
        self.data = np.vstack((self.data, synthetic))
        np.random.shuffle(self.data)

    def update_rate(self, rate):
        self.rate = rate

    def update_metric(self, metric):
        self.metric = metric





