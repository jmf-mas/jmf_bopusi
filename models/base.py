import gzip
import pickle
from pathlib import Path

import torch
from abc import abstractmethod, ABC
from torch import nn
import joblib

class BaseAEModel(nn.Module):

    def __init__(self, params):
        super(BaseAEModel, self).__init__()
        self.params = params

    def save(self):
        parent_name = "checkpoints"
        Path(parent_name).mkdir(parents=True, exist_ok=True)
        with open(parent_name + "/" + self.params.model_name+ "_"+self.params.dataset_name + ".pickle", "wb") as fp:
            pickle.dump(self.state_dict(), fp)

    def load(self):
        with open("checkpoints/" + self.params.model_name+ "_"+self.params.dataset_name  + ".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))


class BaseModel(nn.Module):

    def __init__(self, params):
        super(BaseModel, self).__init__()
        self.dataset_name = params.dataset_name
        self.device = params.device
        self.n_instances, self.in_features = params.data.shape
        self.in_features -=1
        self.resolve_params()
        self.params = params

    def get_params(self) -> dict:
        return {
            "in_features": self.in_features,
            "n_instances": self.n_instances
        }

    def reset(self):
        self.apply(self.weight_reset)

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    @staticmethod
    def load(filename):
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        assert isinstance(model, BaseModel)
        return model

    @abstractmethod
    def resolve_params(self):
        pass

    def save_(self, filename):
        torch.save(self.state_dict(), filename)

    def save(self):
        parent_name = "checkpoints"
        Path(parent_name).mkdir(parents=True, exist_ok=True)
        with open(parent_name + "/" + self.params.model_name+ "_"+self.params.dataset_name + ".pickle", "wb") as fp:
            pickle.dump(self.state_dict(), fp)

    def load(self):
        with open("checkpoints/" + self.params.model_name+ "_"+self.params.dataset_name  + ".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))

    def load_from(self, file):
        with open("checkpoints/" + file + ".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))



class BaseShallowModel(ABC):

    def __init__(self, params):
        self.params = params
        self.resolve_params(params.dataset_name)

    def resolve_params(self, dataset_name: str):
        pass

    def reset(self):
        """
        This function does nothing.
        It exists only for consistency with deep models
        """
        pass

    def save(self):
        joblib.dump(self.params.model, self.params.model_name+"_"+self.params.dataset_name+".pkl")

    def load(self):
        self.params.model = joblib.load(self.params.model_name+"_"+self.params.dataset_name+".pkl")
