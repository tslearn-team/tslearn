from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from tslearn import hdftools
import json
import pickle
import numpy as np


class BaseModelPackage(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def _is_fitted(self) -> bool:
        """
        Implement this method in a subclass to check
        if the model has been fit.

        Usually implements a model specific call to
        sklearn.utils.validation.check_is_fitted

        Returns
        -------
        bool
        """
        pass

    @abstractmethod
    def _get_model_params(self) -> dict:
        """Get model parameters that are sufficient to recapitulate it."""
        pass

    def _to_dict(self, arrays_to_lists=False) -> dict:
        """
        Get model hyper-parameters and model-parameters
        as a dict that can be saved to disk.

        Returns
        -------
        params : dict
            dict with relevant attributes sufficient to describe the model.
        """

        if not self._is_fitted:
            raise ValueError("Model must be fit before it can be packaged")

        d = {'hyper_params': self.get_params(),
             'model_params': self._get_model_params()}

        # This is just for json support to convert numpy arrays to lists
        if arrays_to_lists:
            d['model_params'] = BaseModelPackage._listify(d['model_params'])

        return d

    @staticmethod
    def _listify(model_params) -> dict:
        """
        Convert all numpy arrays in model-parameters to lists.
        Used for json support
        """
        for k in model_params.keys():
            param = model_params[k]

            if isinstance(param, np.ndarray):
                model_params[k] = param.tolist()  # for json support
            else:
                model_params[k] = param
        return model_params

    @staticmethod
    def _organize_model(cls, model):
        """
        Instantiate the model with all hyper-parameters,
        set all model parameters and then return the model.

        Do not use directly. Use the designated classmethod to load a model.

        Parameters
        ----------
        cls : instance of model that inherits from `BaseModelPackage`
            a model instance

        model : dict
            Model dict containing hyper-parameters and model-parameters

        Returns
        -------
        model: instance of model that inherits from `BaseModelPackage`
            instance of the model class with hyper-parameters and
            model parameters set from the passed model dict
        """

        model_params = model.pop('model_params')
        hyper_params = model.pop('hyper_params')  # hyper-params

        # instantiate with hyper-parameters
        inst = cls(**hyper_params)

        # set all model params
        for p in model_params.keys():
            setattr(inst, p, model_params[p])

        return inst

    def to_hdf5(self, path: str):
        """
        Save this model as an HDF5 file.

        Parameters
        ----------
        path : str
            Full file path. File must not already exist.

        Returns
        -------
        None

        Raises
        ------
        FileExistsError
            If a file with the same path already exists.
        """

        d = self._to_dict()
        hdftools.save_dict(d, path, 'data')

    @classmethod
    def from_hdf5(cls, path: str):
        """
        Load from an HDF5 file

        Parameters
        ----------
        path : str
            Full path to file.

        Returns
        -------
        Model instance
        """

        model = hdftools.load_dict(path, 'data')
        return BaseModelPackage._organize_model(cls, model)

    def to_json(self, path: str):
        """
        Save as a JSON file.

        Parameters
        ----------
        path : str
            Full file path.

        Returns
        -------
        None
        """

        d = self._to_dict(arrays_to_lists=True)
        json.dump(d, open(path, 'w'))

    @classmethod
    def from_json(cls, path: str):
        """
        Load from a json file.

        Parameters
        ----------
        path : str
            Full path to file.

        Returns
        -------
        Model instance
        """

        model = json.load(open(path, 'r'))

        # Convert the lists back to arrays
        for k in model['model_params'].keys():
            param = model['model_params'][k]
            if type(param) is list:
                model['model_params'][k] = np.array(param)

        return BaseModelPackage._organize_model(cls, model)

    def to_pickle(self, path: str):
        """
        Save as a pickle. Not recommended for interoperability.

        Parameters
        ----------
        path : str
            Full file path.

        Returns
        -------
        None
        """

        d = self._to_dict()
        pickle.dump(d, open(path, 'wb'), protocol=4)

    @classmethod
    def from_pickle(cls, path: str):
        """
        Load from a pickle file.

        Parameters
        ----------
        path : str
            Full path to file.

        Returns
        -------
        Model instance
        """
        model = pickle.load(open(path, 'rb'))
        return BaseModelPackage._organize_model(cls, model)
