#
import os
from typing import List, Optional
from copy import deepcopy
from argparse import ArgumentParser
from inspect import cleandoc
from warnings import warn
import time

#from lib.utils.hp_opt import load_cfg, expand_grid

PATH = os.getcwd()
PARTITIONS = ['TEST', 'GPU', 'NGPU', 'PGPU', 'CPU', 'CPU0', 'CPU1', 'CPU2', '_jonas']
GPU_PARTITIONS = ['TEST', 'GPU', 'NGPU', '_jonas', 'MULTI']
GPU_RUN_PARTITIONS = ['GPU', 'NGPU']


def write_cpu_sbatch(cfg):
    script = cleandoc(F"""
        #!/bin/sh
        #SBATCH --job-name={cfg['job_name']}
        #SBATCH --ntasks={cfg['ntasks']}
        #SBATCH --cpus-per-task={cfg['cpus']}
        #SBATCH --partition={cfg['partition']}
        #SBATCH --output=/home/jonas/logs/out/{cfg['job_name']}_%A.out
        #SBATCH --error=/home/jonas/logs/error/{cfg['job_name']}_%A.err
        #SBATCH --mail-type={cfg['mail_type']}
        #SBATCH --mail-user={cfg['mail_user']}
        #SBATCH --export=PYTHONPATH=$PYTHONPATH:{cfg['proj_dir']}
        set -e
        echo "Starting python script..."
        echo "python path:" $PYTHONPATH
        cd {cfg['proj_dir']}
        srun /home/jonas/miniconda3/envs/{cfg['venv']}/bin/python {cfg['run_file']} {cfg['cmdargs']}

        wait
        echo "job finished."
        """)
    return script


def write_gpu_sbatch(cfg):
    """Write the SLURM sbatch bash script to run on GPU."""
    script = cleandoc(F"""
        #!/bin/sh
        #SBATCH --job-name={cfg['job_name']}
        #SBATCH --ntasks={cfg['ntasks']}
        #SBATCH --gres=gpu:{cfg['gpu']}{cfg['num_gpus']}
        #SBATCH --cpus-per-task={cfg['cpus']}
        #SBATCH --partition={cfg['partition']}
        #SBATCH --exclude=gpu-[015-018]
        #SBATCH --output=/home/jonas/logs/out/{cfg['job_name']}_%A.out
        #SBATCH --error=/home/jonas/logs/error/{cfg['job_name']}_%A.err
        #SBATCH --mail-type={cfg['mail_type']}
        #SBATCH --mail-user={cfg['mail_user']}
        #SBATCH --export=PYTHONPATH=$PYTHONPATH:{cfg['proj_dir']}
        set -e
        echo "Starting python script..."
        echo "python path:" $PYTHONPATH
        cd {cfg['proj_dir']}
        srun /home/jonas/miniconda3/envs/{cfg['venv']}/bin/python {cfg['run_file']} {cfg['cmdargs']}

        wait
        echo "job finished."
        """)
    return script


def get_args():
    """Read in cmd arguments"""
    parser = ArgumentParser(description='Run experiments on ISMLL SLURM cluster')
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 2.0')
    parser.add_argument('--partition', '-p', nargs='+', default='ALL',
                        help='SLURM partition.')
    parser.add_argument('--venv', type=str, default='nrr',
                        help='virtual conda environment name to activate.')
    parser.add_argument('--job_name', '-n', type=str, default='run',
                        help='job name.')
    parser.add_argument('--run_file', '-r', type=str, default='run.py',
                        help='experiment run file.')
    parser.add_argument('--proj_dir', '-d', type=str, default='./',
                        help='project directory on master node.')
    parser.add_argument('--ntasks', type=int, default=1,
                        help='number of tasks.')
    parser.add_argument('--cpus', '-c', type=int, default=8,
                        help='number of CPUs.')
    parser.add_argument('--gpu', '-g', type=str, default="",
                        help='type of GPU. (e.g. -g 3090 to use only 3090 GPUs)')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of GPUs.')
    parser.add_argument('--problem', type=str, default=None,
                        help='routing problem cfg file name.')
    parser.add_argument('--mail_type', type=str, default='ALL',
                        help='SLURM mailing type: [ALL, FAIL, BEGIN, END, REQUEUE].')
    parser.add_argument('--mail_user', type=str, default='falkner@ismll.de',
                        help='email address for mail event updates.')
    parser.add_argument('--hp_cfg', type=str, default=None,
                        help='path to cfg yaml file for HP search.')
    parser.add_argument('--hpo', action='store_true',
                        help='execute HP search')
    parser.add_argument('--dry_run', action='store_true',
                        help='only test sbatch without running job')
    parser.add_argument('--args', type=str, default='',
                        help="additional command line arguments as one string.")
    args = vars(parser.parse_args())  # parse to dict
    return format_args(args)


def format_args(args):
    """Directly validate some parameters and create directory."""
    part = args['partition']
    if isinstance(part, str):
        if part == 'ALL':
            args['partition'] = ",".join(GPU_RUN_PARTITIONS)
        elif part not in PARTITIONS:
            raise ValueError(F"Partition {args['partition']} not found.")
    elif isinstance(part, list):
        args['partition'] = ",".join([p for p in part if p in PARTITIONS])
    else:
        raise ValueError(F"Partition {args['partition']} not found.")
    if not os.path.exists(args['proj_dir']):
        raise NotADirectoryError(f"Project directory {args['proj_dir']} does not exist!")
    if args['problem'] is None:
        warn("no problem specified!")

    if len(args['gpu']) > 0:
        args['gpu'] = str(args['gpu']) + ':'

    print(f"Current working directory: {PATH}")
    path = os.path.join(PATH, f"run_log/")
    os.makedirs(path, exist_ok=True)

    return args, path


def format_name(name_parts, params):
    """Format the file name including selected hyper parameters."""
    s = ""
    for n, p in zip(*[name_parts, params]):
        if isinstance(p, bool):
            if p:
                s += f"_{str(n)}"
        else:
            s += f"_{n}={p}"
    return s


def execute_sbatch(args, path, job_id):
    """Create and execute the SLURM sbatch script."""
    if args['problem'] is not None:
        cmdargs = f"problem={args['problem']} "    # set corresponding problem environment
    else:
        cmdargs = ""
    cmdargs += args['args']     # add args to executed string
    name = args['job_name']

    fpath = os.path.join(path, f"submit_job_{name}.sh")
    if os.path.exists(fpath):
        print(f"Bash file exists already: {fpath} \nOverwrite file? (y/n)")
        a = input()
        if a != 'y':
            print('Could not write to configuration file.')
            return

    slurm_cfg = args.copy()
    slurm_cfg['job_name'] = name
    slurm_cfg['path'] = fpath
    slurm_cfg['cmdargs'] = cmdargs

    # Write SBATCH bash script
    if slurm_cfg['partition'] == ",".join(GPU_RUN_PARTITIONS) or slurm_cfg['partition'] in GPU_PARTITIONS:
        sbatch_script = write_gpu_sbatch(slurm_cfg)
    else:
        sbatch_script = write_cpu_sbatch(slurm_cfg)

    with open(F"{slurm_cfg['path']}", "w") as file:
        file.write(sbatch_script)

    # execute bash
    print(f"Submitting Job {job_id}...")
    if not args['dry_run']:
        os.system(F"sbatch {slurm_cfg['path']}")
        time.sleep(1.01)  # give slurm some time to process the request
    print(f" done.")


def sbatch_full(args, path, exp_args: Optional[List[str]] = None):
    """
    Execute a sequence of sbatch runs with
    additional arguments provided via exp_args
    ### or over a HP grid defined by a hp nsf_config.
    """
    hpo = args.get("hpo", False)
    if hpo:
        raise NotImplementedError
        # assert args["hp_cfg"] is not None, f"need to provide hp_cfg for hpo."
        # assert exp_args is None, f"can only do hpo without additional exp_args"
        # cfg = load_cfg(args["hp_cfg"])
        # grid = expand_grid(cfg)
        # exp_args = [''.join([f"{k}={v} " for k, v in cmb.items()]) for cmb in grid]

    for i, expa in enumerate(exp_args):
        args_ = deepcopy(args)
        args_str = ""
        if expa[0] != " " and len(args_['args']) > 0:
            if args_['args'][-1] != " ":
                args_['args'] += " "
            args_str = args_['args']
            args_str = f"{args_str.replace(' ', '+')}"
            args_str = f"{args_str.replace('--', '')}"
        args_['args'] += expa
        expa_str = "" if "/" in expa else expa.replace(' ', '+')
        args_['job_name'] += f"_{i+1}_{args['problem']}_{args_str}{'HPO' if hpo else expa_str}"
        print(f"query model with extra args: {expa}")
        execute_sbatch(args_, path, i+1)
