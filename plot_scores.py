#
from typing import List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, ScalarFormatter)


UCHOA_BKS = [
    38.684,
    18.839,
    26.558,
    75.478,
    35.291,
    21.245,
    33.503,
    20.215,
    95.151,
    47.161,
    34.231,
    21.736,
    25.859,
    94.043,
    78.355,
    29.834,
    27.532,
    31.102,
    139.111,
    42.05,
    25.896,
    51.505,
    22.814,
    147.713,
    65.928,
    38.26,
    66.154,
    19.712,
    107.798,
    65.449,
    36.391,
    55.233,
    24.139,
    221.824,
    89.449,
    66.483,
    69.226,
    24.201,
    154.593,
    94.846,
    86.7,
    42.717,
    50.673,
    190.316,
    108.451,
    59.535,
    62.164,
    63.682,
    106.78,
    146.332,
    68.205,
    81.923,
    43.373,
    136.187,
    77.269,
    114.417,
    72.386,
    73.305,
    158.121,
    193.737,
    88.965,
    99.299,
    53.86,
    329.179,
    132.715,
    85.465,
    118.976,
    72.355,
]
UCHOA_N = [
    250,
    255,
    260,
    265,
    269,
    274,
    279,
    283,
    288,
    293,
    297,
    302,
    307,
    312,
    316,
    321,
    326,
    330,
    335,
    343,
    350,
    358,
    366,
    375,
    383,
    392,
    400,
    410,
    419,
    428,
    438,
    448,
    458,
    468,
    479,
    490,
    501,
    512,
    523,
    535,
    547,
    560,
    572,
    585,
    598,
    612,
    626,
    640,
    654,
    669,
    684,
    700,
    715,
    732,
    748,
    765,
    782,
    800,
    818,
    836,
    855,
    875,
    894,
    915,
    935,
    956,
    978,
    1000
]
TAM_AM = [
    73.691077,
    26.9768547,
    180.4564089,
    104.6341072,
    99.0981,
    47.245002,
    55.993665,
    207.254124,
    118.8189156,
    66.464874,
    68.5731084,
    69.1331792,
    114.585618,
    174.5594428,
    75.7280115,
    88.2556479,
    48.57776,
    147.899082,
    85.1658918,
    125.6756328,
    81.759987,
    82.0356255,
    173.458737,
    208.4222646,
    96.9629535,
    108.6033163,
    62.983884,
    376.3174328,
    144.5399065,
    97.361728,
    128.7677248,
    81.7539145
]


def agg_scores(
        m,
        solution_lists: List,
        sizes: List[int],
        num_seeds: int,
        limit: Optional[int] = None,
):
    _n_instances = None
    sols_per_size = []
    for sol_list, sz in zip(solution_lists, sizes):
        if sol_list is None:
            instance_sol = [[np.nan]*num_seeds]*_n_instances
        else:
            print(f"loading total of {len(sol_list)} solutions")
            assert len(sol_list) % num_seeds == 0
            num_instances = len(sol_list) // num_seeds
            _n_instances = num_instances
            instance_sol = []
            for i in range(num_instances):
                instance_sol.append([sol_list[j].cost for j in range(i, num_seeds * num_instances, num_instances)])
        if limit is not None:
            instance_sol = instance_sol[:limit]
            print(f"limited to {len(instance_sol)*num_seeds}")
        sols_per_size.append(instance_sol)

    x = np.array(sizes).repeat(num_seeds).reshape(-1, num_seeds)
    y = np.stack(sols_per_size).mean(axis=1)

    data = pd.DataFrame(
        np.stack([x.flatten(), y.flatten()], axis=-1),
        columns=["size", "cost"]
    )
    data["model"] = np.array([m]*num_seeds*len(sizes))
    return data


def bin_x_y(x, y, bin_size, num_seeds,
            cut_from_back: bool = False):
    n = len(y)
    sort_idx = np.argsort(x[:, 0])
    x = x[sort_idx]
    y = y[sort_idx]

    if bin_size > 1:
        # cutoff start
        ct = n % bin_size
        if cut_from_back:
            x = x[:-ct]
            y = y[:-ct]
        else:
            x = x[ct:]
            y = y[ct:]
        print(f"number of instances {n} not divisible by bin_size {bin_size}. "
              f"Dropping {'last' if cut_from_back else 'first'} {ct} instances.")
        # mean over bins
        x = x.reshape(-1, bin_size, num_seeds).mean(axis=1)
        y = y.reshape(-1, bin_size, num_seeds).mean(axis=1)
    else:
        x = x.reshape(-1, num_seeds)
        y = y.reshape(-1, num_seeds)
    return x, y


def agg_uchoa_scores(
        m,
        solution_lists: List,
        num_seeds: int,
        bin_size: int = 6,
        cut_from_back: bool = False,
):
    sols_per_ds = []
    sizes_per_ds = []
    for sol_list in solution_lists:
        if sol_list is None:
            raise RuntimeError
        else:
            assert len(sol_list) % num_seeds == 0
            num_instances = len(sol_list) // num_seeds
            instance_sol = []
            instance_size = []
            for i in range(num_instances):
                instance_sol.append([sol_list[j].cost for j in range(i, num_seeds * num_instances, num_instances)])
                s = [sol_list[j].instance.graph_size for j in range(i, num_seeds * num_instances, num_instances)]
                s0 = s[0]
                assert np.all(np.array(s) == s0)
                instance_size.append(s)

        sols_per_ds.append(instance_sol)
        sizes_per_ds.append(instance_size)

    x = np.concatenate(sizes_per_ds, axis=0)
    y = np.concatenate(sols_per_ds, axis=0)

    x, y = bin_x_y(x, y, bin_size, num_seeds, cut_from_back)

    data = pd.DataFrame(
        np.stack([x.flatten(), y.flatten()], axis=-1),
        columns=["size", "cost"]
    )
    data["model"] = np.array([m]*len(data))
    return data


if __name__ == "__main__":

    BASELINES = [
        "pomo-greedy",
        "pomo-sampling",
        "sgbs",
        "savings",
    ]
    MODELS = [
    ]

    ROOT_DIR = "./outputs_constr"
    MODEL_DIR = "nrr"

    _PROBLEMS = ["cvrp", "cvrp", "uchoa"]
    _PDISTS = ["mixed", "uniform", ""]
    _GRAPH_SIZES = [
        [200, 500, 1000, 2000],
        [200, 500, 1000, 2000],
        [2, 3]
    ]
    _DS = [
        "data_test_seed333_size100_mixed_random_k_variant",
        "data_test_seed333_size100_uniform_random_int",
        "n"
    ]
    _ALPH_IDX = ["(b)", "(a)", "(c)"]

    ###
    # PROBLEM = "cvrp"
    # PDIST = "mixed" #"uniform"
    # GRAPH_SIZE = [200, 500, 1000, 2000]
    # DS = "data_val_seed222_size100_mixed_random_k_variant"
    # DS = "data_test_seed333_size100_mixed_random_k_variant"
    # DS = "data_test_seed333_size100_uniform_random_int"
    ###
    # PROBLEM = "uchoa"
    # PDIST = ""
    # GRAPH_SIZE = [2, 3]
    # DS = "n"

    NUM_SEEDS = 3
    FNAME = f"eval_results_full_{NUM_SEEDS}.pkl"
    IGNORE = []
    LIMIT = 50
    LOG = False
    res_plt = None
    YLIM = None
    BIN_SIZE = 1 #5

    for PROBLEM, PDIST, GRAPH_SIZE, DS, alphabetical_idx in zip(
        _PROBLEMS, _PDISTS, _GRAPH_SIZES, _DS, _ALPH_IDX,
    ):
        uchoa = "uchoa" in PROBLEM.lower()

        ###
        # # TODO: remove
        if not uchoa:
            continue
        ###

        sns.set_style('ticks')
        fig, ax = plt.subplots()
        fontsize = 8
        fig.set_size_inches(8, 4)
        if uchoa:
            nstr = [str(i) for i in GRAPH_SIZE]
            p_str = f"uchoa_n{'n'.join(nstr)}"
        else:
            assert PDIST in DS
            p_str = f"{PROBLEM}_{PDIST}"
        print(f"Running on: {p_str}")

        dfs = []
        ## BASELINES ##
        for m in BASELINES:
            if m not in IGNORE:
                print(f"plot for {m}...")
                if uchoa:
                    pths = [os.path.join(ROOT_DIR, "baselines", m, "uchoa", f"{DS}{gs}", FNAME) for gs in GRAPH_SIZE]
                    sols = []
                    for pth in pths:
                        try:
                            sols.append(torch.load(pth))
                        except FileNotFoundError as fnfe:
                            print(f"ERROR: {fnfe}")
                            sols.append(None)
                    data = agg_uchoa_scores(
                        m,
                        solution_lists=sols,
                        num_seeds=NUM_SEEDS,
                        bin_size=BIN_SIZE,
                    )
                    dfs.append(data)

                else:
                    pths = [os.path.join(ROOT_DIR, "baselines", m, f"{PROBLEM}{gs}", DS, FNAME) for gs in GRAPH_SIZE]
                    sols = []
                    for pth in pths:
                        try:
                            sols.append(torch.load(pth))
                        except FileNotFoundError as fnfe:
                            print(f"ERROR: {fnfe}")
                            sols.append(None)
                    data = agg_scores(m,
                                       solution_lists=sols,
                                       sizes=GRAPH_SIZE.copy(),
                                       num_seeds=NUM_SEEDS,
                                       limit=LIMIT,
                                       )
                    dfs.append(data)

        ## NRR ##
        fname = f"_eval_results_{NUM_SEEDS}.pkl"
        for m in MODELS:
            print(f"plot for {m}...")
            if uchoa:
                pths = [os.path.join(ROOT_DIR, MODEL_DIR, "uchoa", f"{DS}{gs}", m+fname) for gs in GRAPH_SIZE]
                sols = []
                for pth in pths:
                    try:
                        sols.append(torch.load(pth))
                    except FileNotFoundError as fnfe:
                        print(f"ERROR: {fnfe}")
                        sols.append(None)
                data = agg_uchoa_scores(
                    m,
                    solution_lists=sols,
                    num_seeds=NUM_SEEDS,
                    bin_size=BIN_SIZE,
                )
                dfs.append(data)
            else:
                pths = [os.path.join(ROOT_DIR, MODEL_DIR, f"{PROBLEM}{gs}", DS, m+fname) for gs in GRAPH_SIZE]
                sols = []
                for pth in pths:
                    try:
                        sols.append(torch.load(pth))
                    except FileNotFoundError as fnfe:
                        print(f"ERROR: {fnfe}")
                        sols.append(None)
                data = agg_scores(m,
                                   solution_lists=sols,
                                   sizes=GRAPH_SIZE.copy(),
                                   num_seeds=NUM_SEEDS,
                                   limit=LIMIT
                                   )
                dfs.append(data)

        if uchoa:
            # add BKS and TAM-AM
            sizes = np.array(UCHOA_N)+1
            x, y = bin_x_y(
                sizes[:, None], np.array(UCHOA_BKS),
                bin_size=BIN_SIZE, num_seeds=1
            )
            bks_df = pd.DataFrame(
                np.stack([x.reshape(-1), y.reshape(-1)], axis=-1),
                columns=["size", "cost"]
            )
            bks_df["model"] = np.array(["BKS"] * len(bks_df))

            n_tam = len(TAM_AM)
            cut = n_tam % BIN_SIZE
            cost = np.zeros(len(UCHOA_N), dtype=float)
            cost.fill(np.nan)
            cost[-(n_tam-cut):] = TAM_AM[cut:]
            x, y = bin_x_y(
                sizes[:, None], cost,
                bin_size=BIN_SIZE, num_seeds=1
            )
            tamam_df = pd.DataFrame(
                np.stack([x.reshape(-1), y.reshape(-1)], axis=-1),
                columns=["size", "cost"]
            )
            tamam_df["model"] = np.array(["tam_am"] * len(tamam_df))
            dfs.append(tamam_df)
            dfs.append(bks_df)

            df = pd.concat(dfs)
            df.reset_index(drop=True, inplace=True)
            res_plt = sns.lineplot(df, x="size", y="cost",
                                   markers=True, hue="model",
                                   style="model", ax=ax,
                                   #linewidth=1,
                                   )
            sns.move_legend(ax, "upper left", bbox_to_anchor=(0.02, 1))
        else:
            df = pd.concat(dfs)
            df.reset_index(drop=True, inplace=True)
            res_plt = sns.lineplot(df, x="size", y="cost",
                                   markers=True, hue="model",
                                   style="model", ax=ax)
        if LOG:
            res_plt.set(yscale='log')
        if YLIM is not None:
            res_plt.set_ylim(YLIM)

        h, l = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        plt.legend(h, l,
                   title="$\\bf{model}$",
                   bbox_to_anchor=(0.02, 0.9),
                   loc='upper left',
                   borderaxespad=0,
                   prop={'size': fontsize}
                   )

        if uchoa:
            ax.set_xscale('log')
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
            ax.set_xticks([250, 500, 750, 1000])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
        else:
            ax.set_xscale('log')
            ax.set_xticks([200, 500, 1000, 2000])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())

        res_plt.set_xlabel("size", fontdict={'fontsize': int(fontsize*1.6), 'fontweight': 'bold'})
        res_plt.set_ylabel("cost", fontdict={'fontsize': int(fontsize*1.6), 'fontweight': 'bold'})
        save_dir = "./"
        plt_fname = os.path.join(save_dir, f"plot_{p_str}_scores.pdf")
        plt.savefig(plt_fname, format='pdf', bbox_inches="tight")

