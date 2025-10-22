from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
import json
import pickle
from warnings import warn

import numpy as np

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

from sklearn.exceptions import NotFittedError
try:
    from sklearn.utils import Tags

    @dataclass
    class TsLearnTags(Tags):
        allow_variable_length: bool = False

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.input_tags.sparse = False
            self.input_tags.three_d_array = True

except ImportError:
    # sklearn < 1.6
    TsLearnTags = dict


ALLOW_VARIABLE_LENGTH = 'allow_variable_length'
_DEFAULT_TAGS = {
    ALLOW_VARIABLE_LENGTH: False,
    'sparse': False
}


class TimeSeriesMixin(object):

    def _get_tags(self):
        # sklearn < 1.6 super()._get_tags() returns dict based on _more_tags
        # 1.7 > sklearn >= 1.6 super()._get_tags() returns dict based on __sklearn_tags__
        # through _to_old_tags if available, defaults to _more_tags if not available
        # BaseEstimator.__get_tags remove in sklearn 1.7 -> raises attribute error
        tags = super()._get_tags()

        # Make sure to update tags for sklearn 1.6
        # because _to_old_tags trims allow_variable_length
        # custom tag and _xfails_checks tag
        if not hasattr(tags, ALLOW_VARIABLE_LENGTH) and tags.get(ALLOW_VARIABLE_LENGTH) is None:
            more_tags = self._more_tags()
            tags[ALLOW_VARIABLE_LENGTH] = more_tags[ALLOW_VARIABLE_LENGTH]
            if more_tags.get("_xfail_checks") is not None:
                tags["_xfail_checks"] = more_tags["_xfail_checks"]
        return tags

    @contextmanager
    def _patch_attribute(self, attribute_name, value):
        """Context manager to patch a sklearn estimator's attribute."""
        orig = getattr(self, attribute_name)
        setattr(self, attribute_name, value)
        try:
            yield
        finally:
            setattr(self, attribute_name, orig)

    def _more_tags(self):
        tags = super()._more_tags()
        tags.update(_DEFAULT_TAGS)
        if "3darray" not in tags.setdefault("X_types", []):
            tags["X_types"].append("3darray")
        if "2darray" not in tags.setdefault("X_types", []):
            tags["X_types"].append("2darray")
        # in sklearn < 1.6, BaseEstimator _xfail_checks is False
        if not tags.get("_xfail_checks"):
            tags["_xfail_checks"] = {}
        tags["_xfail_checks"]["check_estimators_pickle"] = "Pickling is currently NOT tested!"
        return tags

    def __sklearn_tags__(self):
        tags_orig = super().__sklearn_tags__()
        as_dict = {
            field.name: getattr(tags_orig, field.name)
            for field in fields(tags_orig)
        }
        tags = TsLearnTags(**as_dict)
        return tags


class BaseModelPackage(object):
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

    def _get_model_params(self):
        """Get model parameters that are sufficient to recapitulate it."""
        params = {}
        for attr in dir(self):
            # Do not save properties
            if (hasattr(type(self), attr) and
                    isinstance(getattr(type(self), attr), property)):
                continue
            if (not attr.startswith("__") and
                    attr.endswith("_") and
                    not callable(getattr(self, attr))):
                params[attr] = getattr(self, attr)
        return params

    def _to_dict(self, output=None, hyper_parameters_only=False):
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
            d['hyper_params'] = BaseModelPackage._listify(d['hyper_params'])

        elif output == 'hdf5':
            d['hyper_params'] = \
                BaseModelPackage._none_to_str(d['hyper_params'])

        if hyper_parameters_only:
            del d["model_params"]

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
            elif isinstance(param, list) and isinstance(param[0], np.ndarray):
                model_params[k] = [p.tolist() for p in param]  # json support
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

    @classmethod
    def _byte2string(cls, model):
        for param_set in ['hyper_params', 'model_params']:
            for k in model[param_set].keys():
                if type(model[param_set][k]) == type(b''):
                    model[param_set][k] = model[param_set][k].decode('utf-8')
        return model


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
        model = cls._byte2string(model)

        for k in model['hyper_params'].keys():
            if isinstance(model['hyper_params'][k], str):
                if model['hyper_params'][k] == 'None':
                    model['hyper_params'][k] = None

        return cls._organize_model(cls, model)

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
        model = cls._byte2string(model)

        # Convert the lists back to arrays
        for param_type in ['model_params', 'hyper_params']:
            for k in model[param_type].keys():
                param = model[param_type][k]
                if type(param) is list:
                    try:
                        arr = np.array(param)
                        if arr.dtype == object:
                            # Then maybe it was rather a list of arrays
                            # This is very hacky...
                            arr = [np.array(p) for p in param]
                    except ValueError:
                        arr = [np.array(p) for p in param]
                    model[param_type][k] = arr

        return cls._organize_model(cls, model)

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
        model = cls._byte2string(model)
        return cls._organize_model(cls, model)
