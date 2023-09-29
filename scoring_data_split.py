#
from typing import Optional
from copy import deepcopy
from warnings import filterwarnings
import os
import argparse
import numpy as np

filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def split(
        file_path: str,
        fname: Optional[str] = None,
        num_val_inst: int = 3,
        seed: int = 42,
):
    rng = np.random.default_rng(seed)

    print(f"loading file from: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    cfg = data.get('cfg')
    keys = data.files.copy()
    keys.remove('cfg')
    keys.remove('size')
    keys.remove('all_instances')
    _keys = deepcopy(keys)
    _data = {k: data[k] for k in _keys}
    print(f"number of data points: {len(_data['instance'])}")

    instances = data.get('all_instances')[0]
    num_inst = len(instances)
    if isinstance(num_val_inst, float):
        assert num_val_inst < 1.0
        num_val_inst = int(np.ceil(num_val_inst*num_inst))
    else:
        assert num_inst > num_val_inst
    num_train_inst = num_inst-num_val_inst
    print(f"splitting according to {num_val_inst} val instances and {num_train_inst} train instances.")

    val_idx = rng.choice(np.arange(num_inst), size=num_val_inst, replace=False)
    val_inst = {k: v for i, (k, v) in enumerate(instances.items()) if i in val_idx}
    train_inst = {k: v for i, (k, v) in enumerate(instances.items()) if i not in val_idx}

    # split according to instance hash into val and train sets
    val_data_idx = [i for i, ihash in enumerate(_data['instance']) if ihash in val_inst]
    train_data_idx = np.delete(np.arange(len(_data['instance'])), val_data_idx)
    print(f"resulting in a total of {len(val_data_idx)} val data "
          f"and {len(train_data_idx)} training data points.")

    val_data = {k: v[val_data_idx] for k, v in _data.items()}
    train_data = {k: v[train_data_idx] for k, v in _data.items()}

    val_data['cfg'] = deepcopy(cfg)
    val_data['size'] = np.array([len(val_data_idx)])
    val_data['all_instances'] = np.array([val_inst], dtype=object)

    train_data['cfg'] = deepcopy(cfg)
    train_data['size'] = np.array([len(train_data_idx)])
    train_data['all_instances'] = np.array([train_inst], dtype=object)

    if fname is None:
        fname = os.path.basename(file_path)
    else:
        if os.path.splitext(fname)[1] != ".npz":
            fname = f"{fname}.npz"
    pth = os.path.dirname(file_path)
    val_pth = os.path.join(pth, f"val_{fname}")
    train_pth = os.path.join(pth, f"train_{fname}")
    print(f"saving to: \n  {val_pth}\n  {train_pth}")

    np.savez_compressed(val_pth, **val_data)
    np.savez_compressed(train_pth, **train_data)
    print("done.\n")


# ============= #
# ### TEST #### #
# ============= #
def _test():
    PTH = "data/_TEST/nrr_data_sgbs_merged.npz"
    split(PTH, num_val_inst=3)


# ## MAIN ## #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--num_val_inst', type=int, default=3)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-n', '--fname', type=str, default=None)
    args = parser.parse_args()

    split(
        args.path,
        fname=args.fname,
        num_val_inst=args.num_val_inst,
        seed=args.seed,
    )
