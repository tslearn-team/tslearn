from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from . import hdftools
import json
import pickle
import numpy as np


class BaseModelPackage(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Does something to check if the model has been fit.
        Usually a model specific call to sklearn.utils.validation.check_is_fitted
        """
        pass

    @abstractmethod
    def get_model_params(self) -> dict:
        """Get model parameters that are sufficient to recapitulate it."""
        pass

    def to_dict(self, arrays_to_lists=False) -> dict:
        """
        Package relevant attributes so it can be exported.
        :return: dict with relevant attributes sufficient for describing the model.
        """

        if not self.is_fitted:
            raise ValueError("Model must be fit before it can be packaged")

        d = dict.fromkeys(['params', 'model_params'])

        params = self.get_params()
        model_params = self.get_model_params()

        d['params'] = params
        d['model_params'] = model_params

        # This is just for json support to convert numpy arrays to lists
        if arrays_to_lists:
            d['model_params'] = BaseModelPackage._listify(model_params, d)

        return d

    @staticmethod
    def _listify(model_params, d) -> dict:
        d['attr_types'] = dict.fromkeys(model_params)

        for param_name in model_params.keys():
            param = model_params[param_name]

            if isinstance(param, np.ndarray):
                d[param_name] = param.tolist()  # for json support
                d['attr_types'][param_name] = 'ndarray'
            else:
                d[param_name] = param
                d['attr_types'][param_name] = str(type(d[param]))
        return d

    @staticmethod
    def _organize_model(cls, model):
        """
        Instantiate the model with all hyper-parameters
        and then set all model parameters

        :param cls:    The model class, meant to be passed from classmethod decorators
        :param model:  Model dict containing hyper-parameters and model parameters
        :return: instance of the model class with hyper-parameters and model parameters set
        """

        model_params = model.pop('model_params')
        params = model.pop('params')  # hyper-params

        # instantiate with hyper-parameters
        inst = cls(**params)

        # set all model params
        for p in model_params.keys():
            setattr(inst, p, model_params[p])

        return inst

    def to_hdf5(self, path: str):
        """Save as an HDF5 file"""
        d = self.to_dict()
        hdftools.save_dict(d, path, 'data')

    @classmethod
    def from_hdf5(cls, path: str):
        """Load from an HDF5 file"""
        model = hdftools.load_dict(path, 'data')
        return BaseModelPackage._organize_model(cls, model)

    def to_json(self, path: str):
        """Save as a json file"""
        d = self.to_dict(arrays_to_lists=True)
        json.dump(d, open(path, 'w'))

    @classmethod
    def from_json(cls, path: str):
        """Load from a json file"""
        model = json.load(open(path, 'r'))

        # Convert the lists back to arrays
        for attr in model['model_params']['attr_types'].keys():
            if model['model_params']['attr_types'][attr] == 'ndarray':
                model['model_params'][attr] = np.array(model[attr])

        return BaseModelPackage._organize_model(cls, model)

    def to_pickle(self, path: str):
        """Save as a pickle. Not recommended."""
        d = self.to_dict()
        pickle.dump(d, open(path, 'wb'), protocol=4)

    @classmethod
    def from_pickle(cls, path: str):
        """Load from a pickle"""
        model = pickle.load(open(path, 'rb'))
        return BaseModelPackage._organize_model(cls, model)
