#
import os
import time
import warnings
import tqdm
from typing import List, Optional
from multiprocessing import Pool
from subprocess import check_call
import tempfile

import math
import numpy as np
import torch
from torch.autograd import Variable

from lib.problem import RPInstance, RPSolution
from baselines.NeuroLKH.NeuroLKH.net.sgcn_model import SparseGCNModel
from baselines.NeuroLKH.NeuroLKH.CVRP_test import (
    write_candidate,
    read_feat,
)


def write_instance(instance, instance_name, instance_filename, k: int = None):
    with open(instance_filename, "w") as f:
        n_nodes = len(instance[0]) - 1
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRP\n")
        if k is not None:
            f.write("VEHICLES : " + str(int(k)) + "\n")
        f.write("DIMENSION : " + str(len(instance[0])) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("CAPACITY : " + str(instance[2]) + "\n")
        f.write("NODE_COORD_SECTION\n")
        s = 1000000
        for i in range(n_nodes + 1):
            f.write(" " + str(i + 1) + " " + str(instance[0][i][0] * s)[:15] + " " + str(instance[0][i][1] * s)[:15] + "\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for i in range(n_nodes):
            f.write(str(i + 2)+" "+str(instance[1][i])+"\n")
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")


def write_para(dataset_name,
               instance_name,
               instance_filename,
               method,
               para_filename,
               max_trials=1000,
               time_limit=None,
               seed=1234,
               solution_filename=None,
               popmusic: bool = False,
               ):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        f.write("MTSP_MIN_SIZE = 0\n")
        f.write("SEED = " + str(seed) + "\n")
        if time_limit is not None:
            f.write("TIME_LIMIT = " + str(time_limit) + "\n")
        f.write("TRACE_LEVEL = 1\n")
        if method == "NeuroLKH":
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/candidate/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
            f.write("CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR\n")
            f.write("MAX_CANDIDATES = 20\n")
        else:
            assert method == "LKH"
            if popmusic:
                f.write("CANDIDATE_SET_TYPE = POPMUSIC\n")
                f.write("INITIAL_PERIOD = 101\n")
        if solution_filename is not None:
            f.write(f"TOUR_FILE = {solution_filename}\n")   # to write best solution to file
        #f.write(f"OUTPUT_TOUR_FILE = {solution_filename}\n")


def read_results(log_filename, sol_filename):
    s = 1000000.0  # precision hardcoded by authors in write_instance()
    objs = []
    penalties = []
    runtimes = []
    running_objs, running_times = [], []
    num_vehicles = 0
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read the obj and runtime for each trial
            if "VEHICLES" in line:
                l = line.strip().split(" ")
                num_vehicles = int(l[-1])
            elif line[:6] == "-Trial":
                line = line.strip().split(" ")
                #assert len(objs) + 1 == int(line[1])
                try:
                    objs.append(int(line[-2]))
                    runtimes.append(float(line[-1]))
                    penalties.append(int(line[-3]))
                except (IndexError, ValueError):
                    pass
            elif line[:1] == "*" and not line[:3] == "***":
                line = line.strip().split(" ")
                # print('line', line)
                # print('line[-8][:-1]', line[-8][:-1])
                # assert len(objs) + 1 == int(line[-8][:-1])
                try:
                    num = int(str(line[-5][2:-1]).strip("_"))
                    #running_objs.append(int(line[-5][2:-1]) / s)
                except ValueError as ve:
                    print(ve)
                running_objs.append(num / s)
                rt = np.nan
                try:
                    rt = float(line[-2])
                except ValueError:
                    warnings.warn(f"line read: << {line} >>")
                    for str_e in line:
                        if '_' in str_e:
                            continue
                        try:
                            rt_ = float(str_e)
                        except Exception:
                            continue
                        else:
                            rt = rt_
                            break

                running_times.append(rt)

        #final_obj = int(lines[-11].split(",")[0].split(" ")[-1])
        #assert objs[-1] == final_obj
        #final_obj = int(lines[-12].split(",")[0].split(" ")[-1])

    tours = []
    dim, total_length = 0, 0
    with open(sol_filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):    # read out solution tours
            if "DIMENSION" in line:
                l = line.strip().split(" ")
                dim = int(l[-1])
            elif "Length" in line:
                l = line.strip().split(" ")
                total_length = int(l[-1])
            elif i > 5 and not "EOF" in line:
                idx = int(line)
                if i == 6:
                    assert idx == 1
                tours.append(idx)

    assert tours[-1] == -1
    assert len(tours) == dim + 1
    N = dim-num_vehicles

    # reformat tours
    tours = (np.array(tours) - 1).tolist()  # reduce idx by 1 (since TSPLIB format starts at 1)
    plan = []
    t = []
    for n in tours[1:]:
        if n <= 0 or n > N:
            plan.append(t)
            t = []
        else:
            t.append(n)
    if len(plan) != num_vehicles:
        warnings.warn(f"plan has different dimension than reported number of vehicles! "
                      f"({len(plan)} != {num_vehicles})")

    # return objs, penalties, runtimes
    return {
        "objs": objs,
        "penalties": penalties,
        "runtimes": runtimes,
        "N": N,
        "num_vehicles": num_vehicles,
        "total_length": total_length,
        "solution": plan,
        "running_costs": running_objs,
        "running_times": running_times,
    }


def solve_LKH(dataset_name,
              instance,
              instance_name,
              rerun=False,
              max_trials=1000,
              time_limit=None,
              seed=1234,
              exe_path=None,
              k=None,
              popmusic: bool = False,
              ):
    para_filename = "result/" + dataset_name + "/LKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    solution_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + ".sol"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename, k)
        write_para(dataset_name, instance_name, instance_filename,
                   "LKH", para_filename,
                   max_trials=max_trials,
                   time_limit=time_limit,
                   seed=seed,
                   solution_filename=solution_filename,
                   popmusic=popmusic,
                   )
        with open(log_filename, "w") as f:
            #check_call(["./LKH", para_filename], stdout=f)
            check_call([str(exe_path), para_filename], stdout=f)
    return read_results(log_filename, solution_filename)


def solve_NeuroLKH(dataset_name,
                   instance,
                   instance_name,
                   candidate,
                   n_nodes_extend,
                   rerun=False,
                   max_trials=1000,
                   time_limit=None,
                   seed=1234,
                   exe_path=None,
                   k=None):
    para_filename = "result/" + dataset_name + "/NeuroLKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/NeuroLKH_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    solution_filename = "result/" + dataset_name + "/NeuroLKH_log/" + instance_name + ".sol"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename, k)
        write_para(dataset_name, instance_name, instance_filename,
                   "NeuroLKH", para_filename,
                   max_trials=max_trials,
                   time_limit=time_limit,
                   seed=seed,
                   solution_filename=solution_filename)
        write_candidate(dataset_name, instance_name, candidate, n_nodes_extend)
        with open(log_filename, "w") as f:
            check_call([str(exe_path), para_filename], stdout=f)
    return read_results(log_filename, solution_filename)


def generate_feat(dataset_name,
                  instance,
                  instance_name,
                  max_nodes,
                  exe_path=None,
                  k=None):
    para_filename = "result/" + dataset_name + "/featgen_para/" + instance_name + ".para"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    feat_filename = "result/" + dataset_name + "/feat/" + instance_name + ".txt"
    write_instance(instance, instance_name, instance_filename, k)
    write_para(dataset_name, instance_name, instance_filename, "FeatGenerate", para_filename)
    with tempfile.TemporaryFile() as f:
        check_call([str(exe_path), para_filename], stdout=f)
    return read_feat(feat_filename, max_nodes)


def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])


def infer_SGN(net,
              dataset_node_feat,
              dataset_edge_index,
              dataset_edge_feat,
              dataset_inverse_edge_index,
              batch_size=100,
              device=torch.device("cpu"),
              ):
    candidate = []
    for i in tqdm.trange(dataset_edge_index.shape[0] // batch_size):
        node_feat = dataset_node_feat[i * batch_size:(i + 1) * batch_size]
        edge_index = dataset_edge_index[i * batch_size:(i + 1) * batch_size]
        edge_feat = dataset_edge_feat[i * batch_size:(i + 1) * batch_size]
        inverse_edge_index = dataset_inverse_edge_index[i * batch_size:(i + 1) * batch_size]
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.FloatTensor),
                             requires_grad=False).to(device)
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.FloatTensor),
                             requires_grad=False).view(batch_size, -1, 1).to(device)
        edge_index = Variable(torch.FloatTensor(edge_index).type(torch.FloatTensor),
                              requires_grad=False).view(batch_size, -1).to(device)
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.FloatTensor),
                                      requires_grad=False).view(batch_size, -1).to(device)
        y_edges, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, 20)
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges.shape[1], 20)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges.shape[1]).reshape(1, -1, 1), y_edges]
        candidate.append(candidate_index[:, :, :5])
    candidate = np.concatenate(candidate, 0)
    return candidate


def cvrp_inference(
        data: List[RPInstance],
        method: str,
        model_path: str,
        lkh_exe_path: str,
        batch_size: int = 1,
        num_workers: int = 1,
        max_trials: int = 1000,
        time_limit: Optional[int] = None,
        seed: int = 1234,
        device=torch.device("cpu"),
        int_prec: int = 10000,
        popmusic: bool = False,
        **kwargs
):
    assert method in ["NeuroLKH", "LKH", "NeuroLKH_M"]
    if method == "NeuroLKH_M":
        method = "NeuroLKH"     # just depends on loaded model checkpoint
    n_samples = len(data)
    dataset_name = "CVRP"
    rerun = True
    if num_workers > os.cpu_count():
        warnings.warn(f"num_workers > num logical cores! This can lead to "
                      f"decrease in performance if env is not IO bound.")

    # set up directories
    os.makedirs("result/" + dataset_name + "/" + method + "_para", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/" + method + "_log", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/cvrp", exist_ok=True)

    # convert data to input format for LKH
    # [1:, ...] since demand for depot node is always 0 and hardcoded in "write_instance"
    dataset = [
        [
            d.coords.tolist(),
            np.ceil(d.demands[1:] * int_prec).astype(int).tolist(),
            int(d.vehicle_capacity * int_prec)
        ] for d in data
    ]
    N = data[0].graph_size-1
    ks = [d.max_num_vehicles for d in data if d.max_num_vehicles is not None]
    if len(ks) > 0:
        max_k = np.max(ks)
        K = int(min(math.ceil(max_k * 1.5), N - 1))
    else:
        K = N//2

    # run solver
    if method == "NeuroLKH":
        # convert data to input format for NeuroLKH
        x = np.stack([d.coords for d in data])
        # [1:, ...] since demand for depot node is always 0 and hardcoded in "write_instance"
        demand = np.stack([d.demands[1:] for d in data])

        os.makedirs("result/" + dataset_name + "/featgen_para", exist_ok=True)
        os.makedirs("result/" + dataset_name + "/feat", exist_ok=True)
        n_nodes = len(dataset[0][0]) - 1    # w/o depot
        max_nodes = int(N + K + 1)
        n_neighbours = 20

        # compute features
        with Pool(num_workers) as pool:
            feats = list(tqdm.tqdm(pool.imap(method_wrapper, [
                ("FeatGen", dataset_name, dataset[i], str(i), max_nodes, lkh_exe_path, K)
                for i in range(len(dataset))
            ]), total=len(dataset)))
        edge_index, n_nodes_extend, feat_runtime = list(zip(*feats))

        # consolidate features
        feat_runtime = np.sum(feat_runtime)
        feat_start_time = time.time()
        edge_index = np.concatenate(edge_index, 0)
        demand = np.concatenate([np.zeros([n_samples, 1]), demand, np.zeros([n_samples, max_nodes - n_nodes - 1])], -1)
        if demand.max() > 1.0:
            demand = demand / dataset[0][2]
        capacity = np.zeros([n_samples, max_nodes])
        capacity[:, 0] = 1
        capacity[:, n_nodes + 1:] = 1
        x = np.concatenate([x] + [x[:, 0:1, :] for _ in range(max_nodes - n_nodes - 1)], 1)
        node_feat = np.concatenate(
            [x, demand.reshape([n_samples, max_nodes, 1]), capacity.reshape([n_samples, max_nodes, 1])], -1)
        dist = node_feat[:, :, :2].reshape(n_samples, max_nodes, 1, 2) - node_feat[:, :, :2].reshape(n_samples, 1,
                                                                                                     max_nodes, 2)
        dist = np.sqrt((dist ** 2).sum(-1))
        edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
        inverse_edge_index = -np.ones(shape=[n_samples, max_nodes, max_nodes], dtype="int")
        inverse_edge_index[
            np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(max_nodes).reshape(1, -1, 1)] = np.arange(
            n_neighbours).reshape(1, 1, -1) + np.arange(max_nodes).reshape(1, -1, 1) * n_neighbours
        inverse_edge_index = inverse_edge_index[
            np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
        feat_runtime += time.time() - feat_start_time
        feat_runtime /= n_samples   # per instance avg

        # load SGN
        net = SparseGCNModel(problem="cvrp")
        #net.cuda()
        net.to(device=device)
        ckpt = torch.load(model_path)
        net.load_state_dict(ckpt["model"])

        # do inference
        sgn_start_time = time.time()
        with torch.no_grad():
            candidate = infer_SGN(net, node_feat, edge_index, edge_feat, inverse_edge_index,
                                  batch_size=batch_size, device=device)
        sgn_runtime = time.time() - sgn_start_time
        sgn_runtime /= n_samples    # per instance avg

        # run LKH
        os.makedirs("result/" + dataset_name + "/candidate", exist_ok=True)
        with Pool(num_workers) as pool:
            results = list(tqdm.tqdm(pool.imap(
                method_wrapper, [
                ("NeuroLKH", dataset_name, dataset[i],
                 str(i), candidate[i], n_nodes_extend[i],
                 rerun, max_trials, time_limit, seed, lkh_exe_path, K)
                for i in range(len(dataset))
            ]), total=len(dataset)))
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0

        if num_workers <= 1:
            results = list(tqdm.tqdm([
                method_wrapper(("LKH", dataset_name, dataset[i], str(i),
                                rerun, max_trials, time_limit, seed, lkh_exe_path, K, popmusic))
                for i in range(len(dataset))
            ], total=len(dataset)))
        else:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    ("LKH", dataset_name, dataset[i], str(i),
                     rerun, max_trials, time_limit, seed, lkh_exe_path, K, popmusic)
                    for i in range(len(dataset))
                ]), total=len(dataset)))

    s = 1000000.0   # precision hardcoded by authors in write_instance()
    objs = [np.array(r['objs'])/s for r in results]
    # objs = objs / s
    penalties = [r['penalties'] for r in results]
    runtimes = [r['runtimes'] for r in results]
    t_base = feat_runtime + sgn_runtime

    # read out trajectories
    trajectories = []
    for obj, rt in zip(objs, runtimes):
        trj_obs, trj_rts, trj_iter = [], [], []
        assert len(obj) == len(rt)
        best_obj = np.inf
        for i in range(len(obj)):
            if obj[i] < best_obj:
                best_obj = obj[i]
                trj_obs.append(obj[i])
                trj_rts.append(rt[i] + t_base)
                trj_iter.append(i)
        trajectories.append({
            "iter": np.array(trj_iter),
            "time": np.array(trj_rts),
            "cost": np.array(trj_obs),
        })
    #running_costs = r['running_costs']
    #running_times = [r['running_times'][t] + feat_runtime + sgn_runtime for t in range(len(r['running_times']))]

    solutions = [
        RPSolution(
            solution=r['solution'],
            run_time=r['runtimes'][-1] + t_base,
            problem=dataset_name.upper(),
            instance=d,
            trajectory=trj,
        ) for r, d, trj in zip(results, data, trajectories)
    ]

    # results_by_trial = {}
    # trials = 1
    # while trials <= objs.shape[1]:
    #     results_by_trial[trials] = {
    #         "objs": objs.mean(0)[trials - 1],
    #         "penalties": penalties.mean(0)[trials - 1],
    #         "runtimes": runtimes.sum(0)[trials - 1],
    #     }
    #     trials *= 10

    results_ = {
        "objs": objs,
        "penalties": penalties,
        "runtimes": runtimes,
        # "results_by_trial": results_by_trial,
    }
    return results_, solutions
