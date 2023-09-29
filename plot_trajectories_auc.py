#
from typing import Union, List
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch

from lib.problem import RPSolution


TAM_AM_cost = [
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
TAM_AM_time = [
    1.01,
    0.99,
    1.00,
    0.95,
    0.88,
    0.93,
    0.99,
    1.08,
    1.13,
    1.02,
    1.01,
    1.53,
    1.13,
    1.24,
    1.16,
    1.16,
    1.23,
    1.36,
    1.55,
    1.34,
    1.35,
    1.63,
    1.81,
    1.49,
    1.4,
    1.43,
    2.91,
    2.58,
    3.09,
    3.41,
    2.95,
    2.93
]


def agg_trj(
    m,
    sol_list: List,
    num_seeds: int,
    t_max: Union[int, float],
    t_min: Union[int, float] = 0,
    t_step: Union[int, float] = 0.1,
    constructive=False,
    report_iters=False,
    limit: int = None,
    offset: int = None,
    fill_val: Union[int, float] = None
):
    if offset is not None:
        sol_list = sol_list[offset:]
    print(f"loading total of {len(sol_list)} solutions")
    assert len(sol_list) % num_seeds == 0
    num_instances = len(sol_list)//num_seeds

    instance_sol = []
    for i in range(num_instances):
        if constructive:
            instance_sol.append([sol_list[j] for j in range(i, num_seeds*num_instances, num_instances)])
        else:
            instance_sol.append([sol_list[j].trajectory for j in range(i, num_seeds*num_instances, num_instances)])

    x = np.arange(t_min, t_max, t_step)
    y = []
    for i, runs in enumerate(instance_sol):
        padded_trjs = []

        if constructive:
            for trj in runs:
                cost_trj = np.zeros_like(x, dtype=float)
                cost_trj.fill(trj.cost)
                t = trj.run_time
                msk = t > x
                cost_trj[:msk.sum()] = fill_val
                padded_trjs.append(cost_trj)
        else:
            for trj in runs:
                times = trj['time']
                costs = trj['cost']
                cost_trj = np.zeros_like(x, dtype=float)
                cost_trj.fill(fill_val)
                for t, c in zip(times, costs):
                    msk = t < x
                    cost_trj[msk] = c
                padded_trjs.append(cost_trj)
                if report_iters:
                    #print(trj)
                    print(f"instance: {i}")
                    iters = trj['iter']
                    half_time = T_MAX // 2
                    half_idx = np.argmax(times>half_time)
                    print(f"start_iter: {iters[0]} - {costs[0]}")
                    print(f"half_iter: {iters[half_idx]} - {costs[half_idx]}")
                    print(f"max_iter: {iters[-1]} - {costs[-1]}")
                    if isinstance(report_iters, (int, float)):
                        iter_idx = np.argmax(times>report_iters)
                        print(f"t={report_iters}: {iters[iter_idx]} - {costs[iter_idx]}")

        y.append(np.array(padded_trjs))

        if limit is not None and i+1 >= limit:
            print(f"limited to {len(y) * num_seeds}")
            break

    y = np.stack(y)
    std = np.std(y, axis=1)
    mean = y.mean(0).reshape(-1)
    rt = None
    if np.isnan(mean).all():
        if constructive:
            rt = np.mean(np.array([[i.run_time for i in l] for l in instance_sol]).reshape(-1))
            final_cost = np.mean(np.array([[i.cost for i in l] for l in instance_sol]).reshape(-1))
            std = np.array([[i.cost for i in l] for l in instance_sol]).std(axis=-1)
        else:
            rt = np.array([[i['time'][-1] for i in l] for l in instance_sol]).mean()
            final_cost = np.array([[i['cost'][-1] for i in l] for l in instance_sol]).mean()
            std = np.array([[i['cost'][-1] for i in l] for l in instance_sol]).std(axis=-1)

        warnings.warn(f"{m}: No feasible result achieved in t_max={t_max}s "
                      f"since mean runtime={rt} > {t_max}!\n"
                      f"final result: {final_cost}")

        mean = mean.reshape(-1, num_seeds)[:, 0].reshape(-1)
        mean[-1] = final_cost

        data = pd.DataFrame(
            np.stack([x, mean], axis=1),
            columns=["time", "cost"]
        )

    else:
        data = pd.DataFrame(
            np.stack([
                x[:, None].repeat(num_seeds, axis=1).T.reshape(-1),
                mean,
            ], axis=1),
            columns=["time", "cost"]
        )

    data["model"] = np.array([m]*len(data))
    cst = round(mean[~np.isnan(mean)].min(), 2)
    stdv = round(std[~np.isnan(std)].mean(), 2)
    print(f"model {m} costs: {cst}  ( {stdv} )")
    return data, rt


def compute_ausc(
        cost_trj: Union[pd.Series, np.ndarray],
        savings_trj: Union[pd.Series, np.ndarray],
        t_step: Union[int, float] = 0.1,
):
    """area between curve and savings curve"""
    # take min of savings curve and cost trajectory
    cost_trj = np.minimum(cost_trj, savings_trj)
    assert np.all(cost_trj >= 0)
    if np.all(cost_trj == 1.0):
        return 0.0
    # Integrate using the composite trapezoidal rule.
    # and compute area under savings curve between
    # savings curve and cost trajectory
    area_savings = np.trapz(y=savings_trj, dx=t_step)
    return (area_savings-np.trapz(y=cost_trj, dx=t_step))/area_savings


###
if __name__ == "__main__":

    BASELINES = [
        "savings",
        "pomo-greedy",
        "sgbs",
        "lkh",
        "neuro_lkh",
        "pomo-sampling",
        "dact",
        "l2d",
        "lkh_popmusic"
    ]
    MODELS = [
        # "sgbs:greedy_rnd_savings_tour_nn_add_greedy_sa",
        "bis_rnd_savings_tour_knn_multi_sa",
        "sgbs:sampling_nsf_savings_sweep_disjoint_sa",
        "sgbs:sampling_nsf_sweep_sweep_disjoint_sa"
    ]
    MODEL_DIRS = [
        "nrr/rnd/",
        "nrr/nsf/",
        "nrr/nsf/",
    ]
    MODEL_NAMES = [
        "rr",
        "nrr",
        "nrr_sweep"
    ]
    # MODEL_DIR = "nrr_nsf_hpo"
    #MODELS = "infer"
    CONSTR = ["pomo-greedy", "pomo-sampling", "sgbs", "savings", "tam-am", "BKS"]

    #ROOT_DIR = "./outputs_eval"
    #ROOT_DIR = "./outputs_hpo"
    ROOT_DIR = "outputs_eval_cluster"

    PROBLEM = "cvrp"
    #PROBLEM = "real_l2d"
    #PROBLEM = "real_square"
    #PROBLEM = "uchoa"
    PDIST = "mixed"

    DS = "data_test_seed333_size100_mixed_random_k_variant"
    #DS = "data_test_seed333_size100_uniform_random_int"
    #DS = "data_test_size50_arnold_l2d"
    #DS = "data_test_seed333_size50_arnold_square-rnd"
    #DS = "n3"

    ###
    NUM_SEEDS = 3
    FNAME = f"eval_results_full_{NUM_SEEDS}.pkl"

    ###
    GRAPH_SIZE = 4000
    COMPUTE_AUC = False
    PLOT = True
    BROKEN_AX = True
    BL_VAL = True
    YLIM = None     #(38, 47.5) # for cvrp500
    INF = ["dact"] if GRAPH_SIZE == 4000 else []

    ###
    UCHOA = False
    if "real" in PROBLEM.lower():
        GRAPH_SIZE = 2000
        T_MAX = 240
        T_STEP = 1
        if "l2d" in PROBLEM.lower():
            if BL_VAL:
                BREAK_VALS = [(145, 172), (345, 370)]
            else:
                BREAK_VALS = [(145, 168.5), (345, 370)]
        else:
            BREAK_VALS = [(23.5, 26.5), (106, 127)]
    elif "uchoa" in PROBLEM.lower():
        UCHOA = True
        GRAPH_SIZE = 1000
        T_MAX = 120
        T_STEP = 0.5
        if BL_VAL:
            BREAK_VALS = [(103.5, 119), (127, 133.5)]
        else:
            BREAK_VALS = [(101, 116.5), (127, 133.5)]
    else:
        _BRK_MAP = {
            500: [(38, 41), (43, 45.5)],
            1000: [(70.5, 84), (105, 113.5)],
            2000: [(138, 159), (274, 296)],
            4000: [(222, 254), (538, 584)]
        }
        BREAK_VALS = _BRK_MAP[GRAPH_SIZE]
        _TMAP = {
            500: 60,
            1000: 120,
            2000: 240,
            4000: 480,
        }
        T_MAX = _TMAP[GRAPH_SIZE]
        T_STEP = GRAPH_SIZE / 1000

    ###
    T_MIN = 0
    LIMIT = 50
    LOG = False
    FONTSIZE = 12
    LEG_NCOLS = 2

    if "cvrp" in PROBLEM.lower():
        p_str = f"{PROBLEM}{GRAPH_SIZE}"
    else:
        p_str = f"{PROBLEM}"

    dfs, invalid_rt = {}, {}
    for m in BASELINES:
        print(f"aggregate data for {m}...")
        pth = os.path.join(ROOT_DIR, "baselines", m, p_str, DS, FNAME)
        if m in INF:
            data = pd.DataFrame(
                np.array([[np.nan, np.nan]]),
                columns=["time", "cost"]
            )
            data["model"] = np.array([m] * len(data))
            dfs[m] = data
        else:
            sol_list = torch.load(pth)
            data, rt = agg_trj(
                m, sol_list,
                num_seeds=NUM_SEEDS,
                t_max=T_MAX, t_min=T_MIN, t_step=T_STEP,
                constructive=(m in CONSTR),
                limit=LIMIT
            )
            dfs[m] = data
            if rt is not None:
                invalid_rt[m] = rt

    if isinstance(MODELS, str) and MODELS == "infer":
        assert len(MODEL_DIRS) == 1
        fname = ""
        MODELS = list(os.listdir(os.path.join(ROOT_DIR, MODEL_DIRS[0], p_str, DS)))
    else:
        fname = f"_eval_results_{NUM_SEEDS}.pkl"
    for m, mdir, m_str in zip(MODELS, MODEL_DIRS, MODEL_NAMES):
        print(f"aggregate data for {m}...")
        pth = os.path.join(ROOT_DIR, mdir, p_str, DS, m+fname)
        sol_list = torch.load(pth)
        data, rt = agg_trj(
            m_str, sol_list,
            num_seeds=NUM_SEEDS,
            t_max=T_MAX, t_min=T_MIN, t_step=T_STEP,
            limit=LIMIT+2 if LIMIT is not None else None,
            offset=2     # first 2 are cfg and summary
        )
        dfs[m_str] = data
        if rt is not None:
            invalid_rt[m_str] = rt

    if UCHOA:
        from plot_scores import UCHOA_BKS
        # load results for TAM-AM, NLNS and BKS
        sol_list = [RPSolution(solution=[], cost=cst, run_time=tm)
                    for cst, tm in zip(TAM_AM_cost, TAM_AM_time)]
        m = 'tam-am'
        data, rt = agg_trj(
            m, sol_list,
            num_seeds=1,
            t_max=T_MAX, t_min=T_MIN, t_step=T_STEP,
            constructive=(m in CONSTR),
            limit=LIMIT
        )
        dfs[m] = data
        if rt is not None:
            invalid_rt[m] = rt
        m = 'nlns'
        sol_list = torch.load("outputs_eval_cluster/baselines/nlns/uchoa/n3/eval_results_full_1.pkl")
        data, rt = agg_trj(
            m, sol_list,
            num_seeds=1,
            t_max=T_MAX, t_min=T_MIN, t_step=T_STEP,
            constructive=(m in CONSTR),
            limit=LIMIT
        )
        dfs[m] = data
        if rt is not None:
            invalid_rt[m] = rt
        # BKS
        sol_list = [RPSolution(solution=[], cost=cst, run_time=0)
                    for cst in UCHOA_BKS[-32:]]
        m = 'BKS'
        data, rt = agg_trj(
            m, sol_list,
            num_seeds=1,
            t_max=T_MAX, t_min=T_MIN, t_step=T_STEP,
            constructive=(m in CONSTR),
            limit=LIMIT
        )
        dfs[m] = data

    # baseline cost
    savings_trj = dfs['savings'].copy().cost.to_numpy()
    savings_cost = savings_trj[~np.isnan(savings_trj)].min()
    bl_cost = savings_cost * 1.1    # +10%
    print(f"baseline cost: {bl_cost}")

    # AUSC
    if COMPUTE_AUC:
        savings_auc = {}
        savings_trj[np.isnan(savings_trj)] = savings_cost
        savings_trj = savings_trj.reshape(NUM_SEEDS, -1).mean(axis=0)
        for m, df in dfs.items():
            cost_trj = df.copy().cost.to_numpy()
            if len(cost_trj)-np.isnan(cost_trj).sum() == 1:
                # no valid trajectory, since no value achieved during available runtime
                # -> savings val
                savings_auc[m] = 0.0
            else:
                # replace nan with savings cost
                cost_trj[np.isnan(cost_trj)] = savings_cost
                if m not in ["tam-am", "nlns", "BKS"]:
                    cost_trj = cost_trj.reshape(NUM_SEEDS, -1).mean(axis=0)
                ###
                savings_auc[m] = compute_ausc(
                    cost_trj=cost_trj,
                    savings_trj=savings_trj.copy(),
                    t_step=0.5,
                )
        print(f"savings AUC for all methods:")
        for k, v in savings_auc.items():
            print(f"{k}: {v}")

        sdir = "./AUSC_results/"
        os.makedirs(sdir, exist_ok=True)
        spth = os.path.join(sdir, f"ausc_{p_str}_{PDIST if 'cvrp' in PROBLEM.lower() else ''}.pkl")
        print(f"saving results to: '{spth}'")
        torch.save(savings_auc, spth)

    # PLOTTING
    print("creating plot ...")
    if PLOT:
        up_ax = []

        if BL_VAL:
            # fill in baseline cost
            for m in dfs.keys():
                if m not in CONSTR:
                    df = dfs[m]
                    costs = df['cost'].copy()
                    # if method cost lies below 110% savings cost, replace all nan values with it
                    if costs.max() <= bl_cost and len(costs[~np.isnan(costs)]) >= 2:
                        costs[np.isnan(costs)] = bl_cost
                        df['cost'] = costs

        if len(dfs) <= 10:
            colors = sns.color_palette()
        else:
            if len(dfs) <= 20:
                colors = sns.color_palette() + sns.color_palette('dark')
            else:
                raise ValueError()

        dash_list = [
            (4, 1.5),
            (1, 1),
            (3, 1, 1.5, 1),
            (5, 1, 1, 1),
            (5, 1, 2, 1, 2, 1),
            (2, 2, 3, 1.5),
            (1, 2.5, 3, 1.2),
            (3, 3, 1, 3, 1, 3),
            (3, 1, 1, 1, 1, 1)
        ]
        dash_iter = iter(dash_list)

        if BROKEN_AX:
            for m, df in dfs.items():
                if df["cost"].min() > BREAK_VALS[0][-1]:
                    up_ax.append(m)

            fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor='w',
                                          gridspec_kw={'height_ratios': [0.35, 0.65]})     # noqa
            ax.set_ylim(*BREAK_VALS[1])
            ax2.set_ylim(*BREAK_VALS[0])
            ax.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax.tick_params(axis="x", which="both", length=0)
            ax.tick_params(labeltop=False)  # don't put tick labels at the top
            ax2.xaxis.tick_bottom()

            d = 0.5
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
            ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

            # joint y axis label
            fig.text(0.055, 0.5, r"cost", va="center", rotation="vertical",
                     fontdict={'fontsize': int(FONTSIZE), 'fontweight': 'bold'})

            for i, (m, df) in enumerate(dfs.items()):
                if m in INF:
                    continue

                df.reset_index(drop=True, inplace=True)
                dsh = "" if m in CONSTR else next(dash_iter)
                if m in up_ax:
                    if (len(df) - df['cost'].isnull().sum()) == 1:
                        sns.lineplot(
                            df, x="time", y="cost",
                            ax=ax,
                            marker="X",
                            palette=[colors[i]],
                            hue="model",
                            style="model",
                            dashes=[dsh],
                            markersize=FONTSIZE*0.9
                        )
                        lbl = float(invalid_rt.get(m, None))
                        xc, yc = df['time'].max(), df['cost'].min()
                        ax.text(xc, yc * 1.001, f"{round(lbl)}s",
                                 horizontalalignment='right',
                                 verticalalignment='bottom',
                                 fontsize=FONTSIZE * 0.7,
                                 )
                    else:
                        sns.lineplot(
                            df, x="time", y="cost",
                            ax=ax,
                            palette=[colors[i]],
                            hue="model",
                            style="model",
                            dashes=[dsh],
                        )
                else:
                    if (len(df) - df['cost'].isnull().sum()) == 1:
                        sns.lineplot(
                            df, x="time", y="cost",
                            ax=ax2,
                            marker="X",
                            palette=[colors[i]],
                            hue="model",
                            style="model",
                            dashes=[dsh],
                            markersize=FONTSIZE*0.9
                        )
                        lbl = float(invalid_rt.get(m, None))
                        xc, yc = df['time'].max(), df['cost'].min()
                        ax2.text(xc, yc * 1.001, f"{round(lbl)}s",
                                 horizontalalignment='right',
                                 verticalalignment='bottom',
                                 fontsize=FONTSIZE * 0.7,
                                 )
                    else:
                        sns.lineplot(
                            df, x="time", y="cost",
                            ax=ax2,
                            palette=[colors[i]],
                            hue="model",
                            style="model",
                            dashes=[dsh],
                        )

            ax.yaxis.label.set_visible(False)
            ax2.yaxis.label.set_visible(False)
            plt.subplots_adjust(
                hspace=0.1, wspace=0.0
            )

            h, l = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.get_legend().remove()
            ax2.get_legend().remove()

            plt.legend(h+h2, l+l2,
                       title="$\\bf{model}$",
                       fontsize=int(FONTSIZE*0.75),
                       bbox_to_anchor=(0.02, 1.5),
                       loc='upper left',
                       borderaxespad=0,
                       prop={'size': FONTSIZE*0.85},
                       ncol=LEG_NCOLS
                       )
            plt.xlabel("time (s)", fontdict={'fontsize': int(FONTSIZE), 'fontweight': 'bold'})

            bv = BREAK_VALS[0]
            spacing = max(max(1, int(GRAPH_SIZE/1000)), (bv[1] - bv[0]) // 5)
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
            bv = BREAK_VALS[1]
            spacing = max(GRAPH_SIZE/500, (bv[1]-bv[0])//5)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(spacing))

        else:
            fig, ax = plt.subplots()
            for i, (m, df) in enumerate(dfs.items()):
                df.reset_index(drop=True, inplace=True)
                dsh = "" if m in CONSTR else next(dash_iter)
                if (len(df) - df['cost'].isnull().sum()) == 1:
                    sns.lineplot(
                        df, x="time", y="cost",
                        ax=ax,
                        marker="X",
                        palette=[colors[i]],
                        hue="model",
                        style="model",
                        dashes=[dsh],
                        markersize=FONTSIZE*0.9
                    )
                    lbl = float(invalid_rt.get(m, None))
                    xc, yc = df['time'].max(), df['cost'].max()
                    plt.text(xc, yc*1.001, f"{round(lbl)}s",
                             horizontalalignment='right',
                             verticalalignment='bottom',
                             fontsize=FONTSIZE*0.7)
                else:
                    sns.lineplot(
                        df, x="time", y="cost",
                        ax=ax,
                        palette=[colors[i]],
                        hue="model",
                        style="model",
                        dashes=[dsh],
                    )

            ax.set_xlabel("time", fontdict={'fontsize': int(FONTSIZE), 'fontweight': 'bold'})
            ax.set_ylabel("cost", fontdict={'fontsize': int(FONTSIZE), 'fontweight': 'bold'})
            sns.move_legend(ax, bbox_to_anchor=(0.02, 1.7), loc='upper left')
            h, l = ax.get_legend_handles_labels()
            ax.get_legend().remove()

            plt.legend(h, l,
                       title="$\\bf{model}$",
                       fontsize=int(FONTSIZE * 0.75),
                       prop={'size': FONTSIZE * 0.85},
                       )

        sns.set_style('white')
        fig.set_size_inches(10, 8)

        if LOG:
            ax.set(yscale='log')
        if YLIM is not None:
            ax.set_ylim(YLIM)

        plt_fname = f"trj_plot_{p_str}_{PDIST if 'cvrp' in PROBLEM.lower() else ''}.pdf"
        save_dir = "./"
        plt_fname = os.path.join(save_dir, plt_fname)
        print(f"saving to: '{plt_fname}'")
        plt.savefig(plt_fname, format='pdf', bbox_inches="tight")

