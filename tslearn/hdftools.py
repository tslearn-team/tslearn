import h5py
import os
import numpy as np
import traceback
from warnings import warn


def save_dict(d: dict, filename: str, group: str, raise_type_fail=True):
    """
    Recursively save a dict to an hdf5 group in a new file.

    Parameters
    ----------
    d: dict
        dict to save as an hdf5 file

    filename : str
        Full path to save the file to. File must not already exist.

    group : str
        group name to save the dict to

    raise_type_fail : bool
        If True: raise an exception if saving a part of the dict fails.
        If False: prints a warning instead and saves the
        object's __str__() return value.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If the path specified by the `filename` parameter already exists.

    TypeError
        If a particular entry within the dict cannot be saved to hdf5 AND
        the argument `raise_type_fail` is set to `True`
    """

    if os.path.isfile(filename):
        raise FileExistsError

    with h5py.File(filename, 'w') as h5file:
        _dicts_to_group(h5file, f'{group}/', d,
                        raise_meta_fail=raise_type_fail)


def _dicts_to_group(h5file: h5py.File, path: str,
                    d: dict, raise_meta_fail: bool):

    for key, item in d.items():

        if isinstance(item, np.ndarray):

            if item.dtype == np.dtype('O'):
                # see if h5py is ok with it
                try:
                    h5file[path + key] = item
                    # h5file[path + key].attrs['dtype'] = item.dtype.str
                except:
                    msg = f"numpy dtype 'O' for item: {item} " \
                          f"not supported by HDF5\n{traceback.format_exc()}"

                    if raise_meta_fail:
                        raise TypeError(msg)
                    else:
                        h5file[path + key] = str(item)
                        warn(f"{msg}, storing whatever str(obj) returns.")

            # numpy array of unicode strings
            elif item.dtype.str.startswith('<U'):
                h5file[path + key] = item.astype(h5py.special_dtype(vlen=str))

                # otherwise h5py doesn't restore the right dtype for str types
                h5file[path + key].attrs['dtype'] = item.dtype.str

            # other types
            else:
                h5file[path + key] = item
                # h5file[path + key].attrs['dtype'] = item.dtype.str

        # single pieces of data
        elif isinstance(item, (str, bytes, int, float, np.int, np.int8,
                               np.int16, np.int32, np.int64, np.float,
                               np.float16, np.float32, np.float64,
                               np.float128, np.complex)):
            h5file[path + key] = item

        elif isinstance(item, dict):
            _dicts_to_group(h5file, path + key + '/', item, raise_meta_fail)

        # last resort, try to convert this object
        # to a dict and save its attributes
        elif hasattr(item, '__dict__'):
            _dicts_to_group(h5file, path + key + '/',
                            item.__dict__, raise_meta_fail)

        else:
            msg = f"{type(item)} for item: {item} " \
                  f"not supported not supported by HDF5"

            if raise_meta_fail:
                raise TypeError(msg)

            else:
                h5file[path + key] = str(item)
                warn(f"{msg}, storing whatever str(obj) returns.")


def load_dict(filename: str, group: str) -> dict:
    """
    Recursively load a dict from an hdf5 group in a file.

    Parameters
    ----------
    filename : str
        full path to the hdf5 file

    group : str
        Name of the group that contains the dict to load

    Returns
    -------
    d : dict
        dict loaded from the specified hdf5 group.
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