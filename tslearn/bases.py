from abc import ABCMeta, abstractmethod
from sklearn.exceptions import NotFittedError
import json
import pickle
import numpy as np
from warnings import warn

h5py_msg = 'h5py not installed, hdf5 features will not be supported.\n'\
           'Install h5py to use hdf5 features: http://docs.h5py.org/'

try:
    import h5py
except ImportError:
    warn(h5py_msg)
    HDF5_INSTALLED = False
else:
    from tslearn import hdftools
    HDF5_INSTALLED = True


class BaseModelPackage:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _is_fitted(self):
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
    def _get_model_params(self):
        """Get model parameters that are sufficient to recapitulate it."""
        pass

    @abstractmethod
    def get_params(self):
        """Get the hyper-parameters for this model"""
        pass

    def _to_dict(self, output=None):
        """
        Get model hyper-parameters and model-parameters
        as a dict that can be saved to disk.

        Returns
        -------
        params : dict
            dict with relevant attributes sufficient to describe the model.
        """

        if not self._is_fitted():
            raise NotFittedError("Model must be fit before it can be packaged")

        d = {'hyper_params': self.get_params(),
             'model_params': self._get_model_params()}

        # This is just for json support to convert numpy arrays to lists
        if output == 'json':
            d['model_params'] = BaseModelPackage._listify(d['model_params'])

        elif output == 'hdf5':
            d['hyper_params'] = \
                BaseModelPackage._none_to_str(d['hyper_params'])

        return d

    @staticmethod
    def _none_to_str(mp):
        """Use str to store Nones. Used for HDF5"""
        for k in mp.keys():
            if mp[k] is None:
                mp[k] = 'None'

        return mp

    @staticmethod
    def _listify(model_params):
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

    def to_hdf5(self, path):
        """
        Save model to a HDF5 file.
        Requires ``h5py`` http://docs.h5py.org/

        Parameters
        ----------
        path : str
            Full file path. File must not already exist.

        Raises
        ------
        FileExistsError
            If a file with the same path already exists.
        """
        if not HDF5_INSTALLED:
            raise ImportError(h5py_msg)

        d = self._to_dict(output='hdf5')
        hdftools.save_dict(d, path, 'data')

    @classmethod
    def from_hdf5(cls, path):
        """
        Load model from a HDF5 file.
        Requires ``h5py`` http://docs.h5py.org/

        Parameters
        ----------
        path : str
            Full path to file.

        Returns
        -------
        Model instance
        """
        if not HDF5_INSTALLED:
            raise ImportError(h5py_msg)

        model = hdftools.load_dict(path, 'data')

        for k in model['hyper_params'].keys():
            if model['hyper_params'][k] == 'None':
                model['hyper_params'][k] = None

        return BaseModelPackage._organize_model(cls, model)

    def to_json(self, path):
        """
        Save model to a JSON file.

        Parameters
        ----------
        path : str
            Full file path.
        """

        d = self._to_dict(output='json')
        json.dump(d, open(path, 'w'))

    @classmethod
    def from_json(cls, path):
        """
        Load model from a JSON file.

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

    def to_pickle(self, path):
        """
        Save model to a pickle file.

        Parameters
        ----------
        path : str
            Full file path.
        """

        d = self._to_dict()
        pickle.dump(d, open(path, 'wb'), protocol=2)

    @classmethod
    def from_pickle(cls, path):
        """
        Load model from a pickle file.

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
