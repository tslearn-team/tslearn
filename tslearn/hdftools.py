from typing import *
import h5py
import json
import os
import numpy as np
import pandas as pd
import traceback
from warnings import warn


def save_dataframe(path: str, dataframe: pd.DataFrame, metadata: Optional[dict] = None,
                   metadata_method: str = 'json', raise_meta_fail: bool = True):
    """
    Save DataFrame to hdf5 file along with a meta data dict.

    Meta data dict can either be serialized with json and stored as a str in the hdf5 file, or recursively saved
    into hdf5 groups if the dict contains types that hdf5 can deal with. Experiment with both methods and see what works best

    Currently the hdf5 method can work with these types: [str, bytes, int, float, np.int, np.int8, np.int16,
    np.int32, np.int64, np.float, np.float16, np.float32, np.float64, np.float128, np.complex].

    If it encounters an object that is not of these types it will store whatever that object's __str__() method
    returns if on_meta_fail is False, else it will raise an exception.

    :param path:            path to save the file to
    :type path:             str

    :param dataframe:       DataFrame to save in the hdf5 file
    :type dataframe:        pd.DataFrame

    :param metadata:        Any associated meta data to store along with the DataFrame in the hdf5 file
    :type metadata:         Optional[dict]

    :param metadata_method: method for storing the metadata dict, either 'json' or 'recursive'
    :type metadata_method:  str

    :param raise_meta_fail: raise an exception if recursive metadata saving encounters an unsupported object
                            If false, it will save the unsupported object's __str__() return value
    :type raise_meta_fail:  bool
    """

    if os.path.isfile(path):
        raise FileExistsError

    f = h5py.File(path, mode='w')

    f.create_group('DATAFRAME')

    if metadata is not None:
        mg = f.create_group('META')
        mg.attrs['method'] = metadata_method

        if metadata_method == 'json':
            bad_keys = []
            for k in metadata.keys():
                try:
                    mg.create_dataset(k, data=json.dumps(metadata[k]))
                except TypeError as e:
                    bad_keys.append(str(e))

            if len(bad_keys) > 0:
                bad_keys = '\n'.join(bad_keys)
                raise TypeError(f"The following meta data keys are not JSON serializable\n{bad_keys}")

        elif metadata_method == 'recursive':
            _dicts_to_group(h5file=f, path='META/', d=metadata, raise_meta_fail=raise_meta_fail)

    f.close()

    dataframe.to_hdf(path, key='DATAFRAME', mode='r+')


def load_dataframe(filepath: str) -> Tuple[pd.DataFrame, Union[dict, None]]:
    """
    Load a DataFrame along with meta data that were saved using ``HdfTools.save_dataframe``

    :param filepath: file path to the hdf5 file
    :type filepath:  str

    :return: tuple, (DataFrame, meta data dict if present else None)
    :rtype: Tuple[pd.DataFrame, Union[dict, None]]
    """

    with h5py.File(filepath, 'r') as f:
        if 'META' in f.keys():

            if f['META'].attrs['method'] == 'json':
                ks = f['META'].keys()
                metadata = dict.fromkeys(ks)
                for k in ks:
                    metadata[k] = json.loads(f['META'][k][()])

            elif f['META'].attrs['method'] == 'recursive':
                metadata = _dicts_from_group(f, 'META/')

        else:
            metadata = None
    df = pd.read_hdf(filepath, key='DATAFRAME', mode='r')

    return (df, metadata)


def save_dict(d: dict, filename: str, group: str, raise_type_fail=True):
    """
    Recursively save a dict to an hdf5 group.

    :param d:        dict to save
    :type d:         dict

    :param filename: filename
    :type filename:  str

    :param group:    group name to save the dict to
    :type group:     str

    :param raise_type_fail: whether to raise if saving a piece of data fails
    :type raise_type_fail:  bool
    """

    if os.path.isfile(filename):
        raise FileExistsError

    with h5py.File(filename, 'w') as h5file:
        _dicts_to_group(h5file, f'{group}/', d, raise_meta_fail=raise_type_fail)


def _dicts_to_group(h5file: h5py.File, path: str, d: dict, raise_meta_fail: bool):
    for key, item in d.items():

        if isinstance(item, np.ndarray):

            if item.dtype == np.dtype('O'):
                # see if h5py is ok with it
                try:
                    h5file[path + key] = item
                    # h5file[path + key].attrs['dtype'] = item.dtype.str
                except:
                    msg = f"numpy dtype 'O' for item: {item} not supported by HDF5\n{traceback.format_exc()}"

                    if raise_meta_fail:
                        raise TypeError(msg)
                    else:
                        h5file[path + key] = str(item)
                        warn(f"{msg}, storing whatever str(obj) returns.")

            # numpy array of unicode strings
            elif item.dtype.str.startswith('<U'):
                h5file[path + key] = item.astype(h5py.special_dtype(vlen=str))
                h5file[path + key].attrs['dtype'] = item.dtype.str  # h5py doesn't restore the right dtype for str types

            # other types
            else:
                h5file[path + key] = item
                # h5file[path + key].attrs['dtype'] = item.dtype.str

        # single pieces of data
        elif isinstance(item, (str, bytes, int, float, np.int, np.int8, np.int16, np.int32, np.int64, np.float,
                               np.float16, np.float32, np.float64, np.float128, np.complex)):
            h5file[path + key] = item

        elif isinstance(item, dict):
            _dicts_to_group(h5file, path + key + '/', item, raise_meta_fail)

        # last resort, try to convert this object to a dict and save its attributes
        elif hasattr(item, '__dict__'):
            _dicts_to_group(h5file, path + key + '/', item.__dict__, raise_meta_fail)

        else:
            msg = f"{type(item)} for item: {item} not supported not supported by HDF5"

            if raise_meta_fail:
                raise ValueError(msg)

            else:
                h5file[path + key] = str(item)
                warn(f"{msg}, storing whatever str(obj) returns.")


def load_dict(filename: str, group: str) -> dict:
    """
    Recursively load a dict from an hdf5 group.

    :param filename: filename
    :type filename:  str

    :param group:    group name of the dict
    :type group:     str

    :return:         dict recursively loaded from the hdf5 group
    :rtype:          dict
    """

    with h5py.File(filename, 'r') as h5file:
        return _dicts_from_group(h5file, f'{group}/')


def _dicts_from_group(h5file: h5py.File, path: str) -> dict:
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            if item.attrs.__contains__('dtype'):
                ans[key] = item[()].astype(item.attrs['dtype'])
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _dicts_from_group(h5file, path + key + '/')
    return ans