# Neural Ruin Recreate

This is the repository accompanying our paper 

"Too Big, so Fail? -- Enabling Neural Construction Methods to Solve Large-Scale Routing Problems". 

A pre-print of our paper can be found on [arxiv](https://arxiv.org/abs/2309.17089).


## Environment setup
```
conda env create -f environment.yml
```
To install verypy, directly install from the [repo](https://github.com/yorak/VeRyPy):
```
pip install https://github.com/yorak/VeRyPy/zipball/master
```

for the "learning to delegate" baseline a different environment is needed,
please also see 
[this github issue](https://github.com/mit-wu-lab/learning-to-delegate/issues/2) 
of the original repo:
```
conda env create -f l2d_environment.yml
```



## Running NRR
The specific parameters are defined in the config files at [config/nrr_config](./config/nrr_config).
The different configurations can be specified on the command line. 
E.g. for a standard run on the CVRP dataset of size 500 using the 

- SGBS sub-solver, 
- knn construction, 
- greedy SG selection,
- initial solution via savings, 
- random scoring
- in debug mode

```
python run_nrr.py meta=debug solver=sgbs constr_mthd=knn select_mthd=greedy init_mthd=savings score_mthd=rnd problem=cvrp500_mixed
```

An overview can be displayed via
```
python run_nrr.py -h
```


## Learning the neural scoring function (NSF)

1) First some general dataset has to be generated via the [create_cvrp_data notebook](create_cvrp_data.ipynb).

2) Then, we need to run a solver (e.g. SGBS or LKH3) on some subgraphs of the RR procedure. 
The respective command to run it for the NRR configuration using 
   - savings initial solutions, 
   - sweep based SG selection,
   - the SGBS sub-solver
   - on uniform CVRP data of problem size 500
    ```
    python create_scoring_data.py method=score_savings_sweep_all solver=sgbs problem=cvrp500_unf
    ```
    Some predefined configurations for the data scoring can be found in 
    [config/nrr_config/method](./config/nrr_config/method).
3) Next, the NSF has to be trained 
    ```
    python train_model.py meta=train problem=sgbs_merged_500_unf
    ```
   The corresponding configuration can be found in [config/nsf_config](./config/nsf_config)


## Running baseline models
We have a unified interface for all baselines (but the "learning to delegate" aka L2D model), which can be invoked via
```
python run_baseline.py -m savings -d data/CVRP/benchmark/uchoa/n3.dat --n_seeds 3
```
where 

- -m is the baseline model
- -d is the path to the respective dataset
- --n_seeds is the number of random seeds for which to rerun the baseline

Further flags can be found in the run file [run_baseline.py](./run_baseline.py)
and the registry at [baselines/methods_registry.py](./baselines/methods_registry.py)

For the installation of required packages for some baselines (e.g. LKH3 and HGS) 
please see the respective readmes in the baselines directory 
