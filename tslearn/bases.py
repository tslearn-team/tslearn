from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from . import hdftools
import json
import pickle
import numpy as np


class BaseModelPackage(BaseEstimator, metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        """
        Raises an exception if 'model_attrs' is not defined as a class attribute
        """
        if not hasattr(cls, 'model_attrs'):
            raise AttributeError('Must define class attribute "model_attrs"')

    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Does something to check if the model has been fit.
        Usually a model specific call to sklearn.utils.validation.check_is_fitted
        """
        pass

    def to_dict(self, arrays_to_lists=False) -> dict:
        """
        Package relevant attributes so it can be exported.
        :return: dict with relevant attributes sufficient for describing the model.
        """

        if not self.is_fitted:
            raise ValueError("Model must be fit before it can be packaged")

        d = dict()
        d['attr_types'] = dict.fromkeys(self.model_attrs)
        for a in self.model_attrs:
            attr = getattr(self, a)

            if isinstance(attr, np.ndarray) and arrays_to_lists:
                d[a] = attr.tolist()  # for json support
                d['attr_types'][a] = 'ndarray'
            else:
                d[a] = attr
                d['attr_types'][a] = str(type(d[a]))

        params = self.get_params()
        params.pop('model')
        d['params'] = params

        return d

    @staticmethod
    def _organize_model(model):
        """
        Remove some keys and organize the params so it can be
        directly passed as kwargs
        """

        model.pop('attr_types')
        params = model.pop('params')

        return params, model

    def to_hdf5(self, path: str):
        """Save as an HDF5 file"""
        d = self.to_dict()
        hdftools.save_dict(d, path, 'data')

    @classmethod
    def from_hdf5(cls, path: str):
        """Load from an HDF5 file"""
        model = hdftools.load_dict(path, 'data')
        params, model = BaseModelPackage._organize_model(model)

        return cls(**params, model=model)

    def to_json(self, path: str):
        """Save as a json file"""
        d = self.to_dict(arrays_to_lists=True)
        json.dump(d, open(path, 'w'))

    @classmethod
    def from_json(cls, path: str):
        """Load from a json file"""
        model = json.load(open(path, 'r'))

        # Convert the lists back to arrays
        for attr in model['attr_types'].keys():
            if model['attr_types'][attr] == 'ndarray':
                model[attr] = np.array(model[attr])

        params, model = BaseModelPackage._organize_model(model)

        return cls(**params, model=model)

    def to_pickle(self, path: str):
        """Save as a pickle"""
        d = self.to_dict()
        pickle.dump(d, open(path, 'wb'), protocol=4)

    @classmethod
    def from_pickle(cls, path: str):
        """Load from a pickle"""
        model = pickle.load(open(path, 'rb'))
        params, model = BaseModelPackage._organize_model(model)

        return cls(**params, model=model)
