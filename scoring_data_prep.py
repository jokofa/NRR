#
from typing import Union, Dict, List, Optional
from copy import deepcopy
from warnings import warn, filterwarnings
from datetime import datetime
from timeit import default_timer
import itertools as it
import os
import shutil
import pickle
import argparse
import numpy as np

from lib.problem import RPInstance

filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def hash_instance(inst: Union[Dict, RPInstance]):
    if isinstance(inst, RPInstance):
        return hash(inst.coords.tobytes() + inst.demands.tobytes())
    elif isinstance(inst, dict):
        return hash(inst['coords'].tobytes() + inst['demands'].tobytes())
    else:
        raise TypeError(type(inst))


def hash_subgraph(sg: List[List[int]]):
    return tuple(np.unique(list(it.chain.from_iterable(sg))).tolist())


def cat_array(arrs: List, dim: int = 0):
    a = arrs[0]
    if a is None:
        return arrs
    elif isinstance(a, np.ndarray):
        try:
            return np.stack(arrs, axis=dim)
        except ValueError:
            return np.array(arrs, dtype=object)
    elif isinstance(a, (int, float, np.generic)):
        return np.array(arrs)
    elif isinstance(a, (list, tuple)):
        return np.array(arrs)
    else:
        raise ValueError(a)


def convert(
        file_paths: Union[str, List[str]],
        fname: Optional[str] = None,
        offset: int = 0,
        limit: int = None,
        remove_duplicates: bool = True,
        try_load: bool = False
):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    sg_hash_table = {}
    instances = {}
    data = None
    n_duplicates = 0
    size, num_inst = 0, 0
    cfgs, fstrings = [], []

    for fpth in file_paths:
        fstrings.append(os.path.basename(os.path.splitext(fpth)[0]))
        filepath = os.path.normpath(os.path.expanduser(fpth))

        i = 1
        print(f"loading data from: {filepath}")
        with open(filepath, 'rb') as f:
            try:
                # first file is cfg dict
                cfg = pickle.load(f)
                cfgs.append(cfg)
                while True:
                    if limit and i > limit:
                        break
                    if i > offset:
                        insert = True
                        dat = pickle.load(f)
                        if data is None:
                            data = {k: [] for k in dat}
                        inst = dat['instance']
                        ihash = hash_instance(inst)
                        if ihash not in instances:
                            instances[ihash] = deepcopy(inst)
                            sg_hash_table[ihash] = {}

                        if remove_duplicates:
                            sghash = hash_subgraph(dat['sg_old_routes'])
                            if sghash not in sg_hash_table[ihash]:
                                sg_hash_table[ihash][sghash] = dat['sg_old_cost']-dat['sg_old_cost']
                            else:
                                imp_diff = abs(sg_hash_table[ihash][sghash] -
                                               (dat['sg_old_cost']-dat['sg_old_cost']))
                                if imp_diff < 0.001:  # same sg, same cost -> duplicate
                                    insert = False
                                    n_duplicates += 1
                                else:   # same sg, vastly different cost -> adapt
                                    sg_hash_table[ihash][sghash] = (
                                            sg_hash_table[ihash][sghash] + (dat['sg_old_cost']-dat['sg_old_cost'])
                                    )/2

                        if insert:
                            dat['instance'] = ihash
                            for k, v in dat.items():
                                if k != "sg_solver_cfg":
                                    data[k].append(v)
                    else:
                        # skip
                        pickle.load(f)

                    if i % 10000 == 1:
                        print(f"loaded: {i-1:09d}")
                    i += 1

            except EOFError:
                pass
        #
        print(f"finished conversion: {len(data['instance']) - size} data points "
              f"for {len(instances) - num_inst} instances.")
        size = len(data['instance'])
        num_inst = len(instances)

    print(f"excluded {n_duplicates}.")
    # save data as npz
    data.pop("sg_solver_cfg")   # remove solver cfgs
    for k in data.keys():
        data[k] = cat_array(data[k])
    data['cfg'] = np.array(cfgs, dtype=object)
    data['size'] = np.array([size])
    data['all_instances'] = np.array([instances], dtype=object)
    if fname is None:
        fname = '+'.join(fstrings)
    if os.path.splitext(fname)[1] != ".npz":
        fname = f"{fname}.npz"
    fname = os.path.join(os.path.dirname(file_paths[0]), fname)
    print(f"saving to: {fname}")
    if os.path.isfile(fname) and os.path.exists(fname):
        print(f'Dataset file with same name exists already: {fname}')
        pre, ext = os.path.splitext(fname)
        new_f = pre + '_' + datetime.utcnow().strftime('%Y%m%d%H%M%S%f') + ext
        print(f'archiving existing file to: {new_f}')
        shutil.copy2(fname, new_f)
        os.remove(fname)

    np.savez_compressed(fname, **data)
    print("done.\n")

    if try_load:
        del data
        del instances
        del sg_hash_table
        try:
            data = np.load(fname, allow_pickle=True)
            keys = data.files.copy()
            print(type(data), data, keys)
            keys.remove('cfg')
            keys.remove('size')
            keys.remove('all_instances')
            _keys = deepcopy(keys)
            _data = {k: data[k] for k in _keys}
            for idx in [0, 5, -1]:
                d = {
                    # k: self._file_handle[k][idx] for k in self._keys
                    k: _data[k][idx] for k in _keys
                }
                print(d)
            inst = next(iter(data.get('all_instances')[0].values()))
            print(inst)
            try:
                inst = RPInstance.make(**inst)
                print(inst)
            except Exception as e:
                warn(f"ERROR when creating instances: {e}")

        except Exception as e:
            warn(f"ERROR when loading: {e}")
            del data


# ============= #
# ### TEST #### #
# ============= #
def _test():
    PTHS = [
        "data/_TEST/nrr_data_2_lkh_sweep_sweep_rnd_all_max_trials100.dat",
        "data/_TEST/nrr_data_2_lkh_sweep_sweepxx_rnd_all_max_trials100.dat"
    ]
    convert(PTHS, limit=2000)


# ## MAIN ## #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, nargs='+')
    parser.add_argument('-o', '--offset', type=int, default=0)
    parser.add_argument('-l', '--limit', type=int, default=None)
    parser.add_argument('-n', '--fname', type=str, default=None)
    parser.add_argument('--try_load', action='store_true')
    parser.add_argument('--with_duplicates', action='store_true')
    args = parser.parse_args()

    convert(
        args.path,
        fname=args.fname,
        offset=args.offset,
        limit=args.limit,
        remove_duplicates=not args.with_duplicates,
        try_load=args.try_load
    )
