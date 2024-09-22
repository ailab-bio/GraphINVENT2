"""
Example submission script for a GraphINVENT2 training job (unconditional generation,
 not fine-tuning/optimization job). This can be used to pre-train a model before
 a reinforcement learning (fine-tuning) job.

To run, type:
 user@cluster GraphINVENT2$ python submit.py
"""
# load general packages and functions
import csv
import sys
import os
from pathlib import Path
import subprocess
import time


class Config:
    """
    Configuration for running GraphINVENT2 jobs. Modify these parameters as necessary.
    """
    def __init__(self):
        # set paths here
        self.python_path      = "apptainer exec docker/graphinvent.sif /opt/conda/envs/graphinvent/bin/python"
        self.graphinvent_path = "./graphinvent/"
        self.data_path        = "./data/pre-training/"

        # define what you want to do for the specified job(s)
        self.dataset          = "gdb13-debug"  # dataset name in "./data/pre-training/"
        self.job_type         = "train"        # "preprocess", "train", "generate", "fine-tune", or "test"
        self.jobdir_start_idx = 0              # where to start indexing job dirs
        self.n_jobs           = 1              # number of jobs to run per model
        self.restart          = False          # whether or not this is a restart job
        self.force_overwrite  = True           # overwrite job directories which already exist
        self.jobname          = self.job_type  # label used to create a job sub directory (can be anything)

        # set SLURM params here (if using SLURM)
        self.use_slurm        = True               # use SLURM or not
        self.run_time         = "0-06:00:00"       # d-hh:mm:ss
        self.account          = "XXXXXXXXXX"       # if cluster requires specific allocation/account, use here

        # define dataset-specific parameters
        self.params = {
            "atom_types"     : ["C", "N", "O", "S", "Cl"],
            "formal_charge"  : [-1, 0, +1],
            "max_n_nodes"    : 13,
            "job_type"       : self.job_type,
            "dataset_dir"    : f"{self.data_path}{self.dataset}/",
            "restart"        : self.restart,
            "sample_every"   : 50,
            "init_lr"        : 1e-4,
            "epochs"         : 1000,
            "batch_size"     : 50,
            "block_size"     : 1000,
            "device"         : "cuda",  # or "cpu" if no CUDA
            "n_samples"      : 100,
            # additional paramaters can be defined here, if different from the "defaults"
            # for instance, for "generate" jobs, don't forget to specify "generation_epoch"
            # and "n_samples"
        }

    def update_paths(self, job_dir, tensorboard_dir):
        """
        Update dynamic paths for each job in the configuration.
        """
        self.params['job_dir'] = str(job_dir)+"/"
        self.params['tensorboard_dir'] = str(tensorboard_dir)+"/"

def submit(config):
    """
    Creates and submits submission scripts based on the provided configuration.

    Args:
        config: Configuration object containing all settings and parameters.
    """
    dataset_output_path, tensorboard_path = create_output_directories(config)
    submit_jobs(config, dataset_output_path, tensorboard_path)

def create_output_directories(config):
    """
    Creates output and tensorboard directories.

    Args:
        config: Configuration object containing dataset and jobname.

    Returns:
        A tuple of dataset_output_path and tensorboard_path.
    """
    base_path = Path(f"./output/output_{config.dataset}")
    dataset_output_path = base_path / config.jobname if config.jobname else base_path
    tensorboard_path = dataset_output_path / "tensorboard"

    dataset_output_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    print(f"* Creating dataset directory {dataset_output_path}/", flush=True)

    return dataset_output_path, tensorboard_path

def submit_jobs(config, dataset_output_path, tensorboard_path):
    """
    Submits the specified number of jobs by creating subdirectories and submission scripts.

    Args:
        config: Configuration object containing job and execution details.
        dataset_output_path: Path object for the dataset output directory.
        tensorboard_path: Path object for the tensorboard directory.
    """
    jobdir_end_idx = config.jobdir_start_idx + config.n_jobs
    for job_idx in range(config.jobdir_start_idx, jobdir_end_idx):
        job_dir = dataset_output_path / f"job_{job_idx}/"
        tensorboard_dir = tensorboard_path / f"job_{job_idx}/"

        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        create_job_directory(job_dir, config)

        config.update_paths(job_dir, tensorboard_dir)
        write_input_csv(config, job_dir, filename="input.csv")
        submit_job(config, job_dir)

        print("-- Sleeping 2 seconds.")
        time.sleep(2)

def create_job_directory(job_dir, config):
    """
    Creates a job directory, handles overwriting based on configuration.

    Args:
        job_dir: Path object for the job directory.
        config: Configuration object with job type and overwrite settings.
    """
    try:
        job_dir.mkdir(parents=True, exist_ok=config.force_overwrite or config.job_type in ["generate", "test"])
        print(f"* Creating model subdirectory {job_dir}/", flush=True)
    except FileExistsError:
        print(f"-- Model subdirectory {job_dir} already exists.", flush=True)
        if not config.restart:
            return

def submit_job(config, job_dir):
    """
    Writes and submits a job based on the SLURM configuration or runs directly.

    Args:
        config: Configuration object with paths and SLURM settings.
        job_dir: Path object for the job directory.
    """
    if config.use_slurm:
        print("* Writing submission script.", flush=True)
        write_submission_script(config, job_dir)

        print("* Submitting job to SLURM.", flush=True)
        subprocess.run(["sbatch", str(job_dir / "submit.sh")], check=True)
    else:
        print("* Running job as a normal process.", flush=True)
        subprocess.run([config.python_path, config.graphinvent_path + "main.py", "--job-dir", str(job_dir)+"/"], check=True)

def write_input_csv(config, job_dir, filename="params.csv") -> None:
    """
    Writes job parameters/hyperparameters from the config object to a CSV file.
    Args:
        config: Configuration object containing all settings and parameters.
        job_dir (Path): The directory where the job will run.
        filename (str): Filename for the CSV output, default is "params.csv".
    """
    dict_path = job_dir / filename

    try:
        # open the file at dict_path in write mode
        with dict_path.open(mode="w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            for key, value in config.params.items():
                writer.writerow([key, value])
    except IOError as e:
        # exception handling for any I/O errors
        print(f"Failed to write file {dict_path}: {e}")

def write_submission_script(config, job_dir) -> None:
    """
    Writes a submission script (`submit.sh`) using settings from the config object.
    Args:
        config: Configuration object containing all settings and parameters.
        job_dir (Path): The directory where the job will run.
    """
    submit_filename = job_dir / "submit.sh"
    output_filename = job_dir / f"output.o${{SLURM_JOB_ID}}"
    main_py_path = Path(config.graphinvent_path) / "main.py"

    with submit_filename.open("w") as submit_file:
        submit_file.write("#!/bin/bash\n")
        submit_file.write(f"#SBATCH -A {config.account}\n")
        submit_file.write(f"#SBATCH --job-name={config.job_type}\n")
        submit_file.write(f"#SBATCH --time={config.run_time}\n")
        submit_file.write("#SBATCH --gpus-per-node=T4:1\n")
        submit_file.write("hostname\n")
        submit_file.write("export QT_QPA_PLATFORM='offscreen'\n")
        submit_file.write(f"({config.python_path} {main_py_path} --job-dir {job_dir} > {output_filename})\n")

if __name__ == "__main__":
    config = Config()  # create an instance of the Config class
    submit(config)     # pass the config object to the submit function
