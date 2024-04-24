import tempfile
import torch


class EarlyStopper:

    def stop(self, epoch, val_loss, val_auc=None, test_loss=None, test_auc=None, test_ap=None, test_f1=None,
             test_score=None, train_loss=None):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return self.train_loss, self.val_loss, self.val_auc, self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score, self.best_epoch


class Patience(EarlyStopper):

    def __init__(self, patience=10, use_train_loss=True, model=None):
        self.local_val_optimum = float("inf")
        self.use_train_loss = use_train_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.model = model
        self.temp_dir = tempfile.TemporaryDirectory()
        self.best_model_path = f"{self.temp_dir.name}/model.pk"
        self.train_loss = None
        self.val_loss, self.val_auc, = None, None
        self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score = None, None, None, None, None

    def stop(self, epoch, val_loss, val_auc=None, test_loss=None, test_auc=None, test_ap=None, test_f1=None,
             test_score=None, train_loss=None):
        if self.use_train_loss:
            if train_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = train_loss
                self.best_epoch = epoch
                self.train_loss = train_loss
                self.val_loss, self.val_auc = val_loss, val_auc
                self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score \
                    = test_loss, test_auc, test_ap, test_f1, test_score

                self.model.save(self.best_model_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.train_loss = train_loss
                self.val_loss, self.val_auc = val_loss, val_auc
                self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score \
                    = test_loss, test_auc, test_ap, test_f1, test_score

                self.model.save(self.best_model_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience

    def get_best_vl_metrics(self):
        self.model.load_state_dict(torch.load(self.best_model_path))
        return self.model, self.train_loss, self.val_loss, self.val_auc, self.test_loss, self.test_auc, self.test_ap, self











































