## Reinforcement learning
In this tutorial, you will be guided through the steps of using reinforcement learning for fine-tuning a pre-trained model in GraphINVENT2.

### Preparing a fine-tuning job
Once you have pre-trained a model according to [2_using_a_new_dataset.md](2_using_a_new_dataset.md) and/or [4_transfer_learning.md](4_transfer_learning.md), you must prepare a submission script that can find your pre-train model and run a reinforcement learning job for fine-tuning it according to a specific scoring model.

A sample submission script [../submit.py](../submit.py) has been provided. Begin by modifying the submission script to specify that you want to run a reinforcement learning job. This is done with the "fine-tune" flag. For example, you can use the settings below, substituting in your own values where necessary:

```
submit.py >
        # define what you want to do for the specified job(s)
        self.dataset          = "your_dataset_name"  # this is the dataset name, which corresponds to the directory containing the data, located in ./data/pre-training/ unless otherwise specified
        self.job_type         = "fine-tune"    # this tells the code that this is a reinforcement learning job
        self.jobdir_start_idx = 0              # where to start indexing job dirs
        self.n_jobs           = 1              # number of jobs to run per model
        self.restart          = False          # this is not a restart job
        self.force_overwrite  = True           # ok to overwrite job directories which already exist
        self.jobname          = self.job_type  # just a label used to create a job sub directory (can be anything you want, e.g., the score name)
```

Then, specify whether you want the job to run using [SLURM](https://slurm.schedmd.com/overview.html). In the example below, we specify that we want the job to run as a regular process (i.e., no SLURM). In such cases, any specified run time and memory requirements will be ignored by the script. Note: if you want to use a different scheduler, this can be easily changed in the submission script (search for "sbatch" and change it to your scheduler's submission command).

```
submit.py >
        # set SLURM params here (if using SLURM)
        self.use_slurm        = False              # this tells the code to NOT use SLURM
        self.run_time         = "0-06:00:00"       # run for 6 hrs max
        self.account          = "XXXXXXXXXX"       # if cluster requires specific allocation/account, use here
```

Then, specify the path to the Python binary in the GraphINVENT2 virtual environment. You probably won't need to change *graphinvent_path* or *data_path*, unless you want to run the code from a different directory. Generally you can keep these parameters the same as for any prior training jobs used to learn the prior:

```
submit.py >
        # set paths here
        self.python_path      = "apptainer exec docker/graphinvent.sif /opt/conda/envs/graphinvent/bin/python"  # this is the path to the Python binary to use (change to your own)
        self.graphinvent_path = "./graphinvent/"        # this is the directory containing the source code
        self.data_path        = "./data/pre-training/"  # this is the directory where all datasets are found
```

Finally, details regarding the specific dataset you want to use need to be entered. You will need to specify all dataset-specific parameters that you want to use for training in Config as shown in the previous tutorials. Generally you can keep these parameters the same as for any prior training jobs used to learn the prior.

To these parameters, we can add specific parameters indicating the reinforcement learning settings:

```
submit.py >
        # define dataset-specific parameters
        self.params = {
            "atom_types"       : ["C", "N", "O", "S", "Cl"],# your dataset's atom types
            "formal_charge"    : [-1, 0, +1],               # your dataset's formal charges
            "max_n_nodes"      : 13,                        # your dataset's value
            "job_type"         : self.job_type,
            "dataset_dir"      : f"{self.data_path}{self.dataset}/",
            "sample_every"     : 50,                        # how often you want to sample/evaluate your model during training (for larger datasets, we recommend sampling more often)
            "init_lr"          : 1e-4,                      # tune the initial learning rate if needed
            "epochs"           : 100,                       # how many RL steps you want to train for (you can experiment with this)
            "batch_size"       : 50,                        # tune the batch size if needed
            "block_size"       : 1000,                      # tune the block size if needed
			"generation_epoch" : 100,                       # <-- epoch corresponding to pre-trained model to use
    		"n_samples"        : 2000,                      # <-- how many samples to generate when evaluating the model
            "sigma"            : 20,                        # <-- how much to weight the scoring function in the loss
            "alpha"            : 0.5,                       # <-- how much to weight the best agent reminder component in the loss (0.5 weighs them equally)
			"score_components" : ["QED", "drd2_activity", "target_size=13"],          # <-- list of all components to use in RL scoring fn **see note below**
    		"score_thresholds" : [0.5, 0.5, 0.0],                                     # <-- acceptable thresholds for above score components (order matters)
    		"score_type"  : "binary",                                                 # <-- can be "binary" or "continuous"
    		"qsar_models" : {"drd2_activity": "data/fine-tuning/qsar_model.pickle"},  # <-- dictionary containing the path to each activity model specified in `score_components`
        }
```

\*\*Some score components are predefined in the code (see [graphinvent/ScoringFunction.py](graphinvent/ScoringFunction.py) and [graphinvent/parameters/defaults.py](graphinvent/parameters/defaults.py) for a list of options you can use). If you would like to define a new scoring function to use in GraphINVENT2, please read on below.

If any parameters are not specified in *submit.py* before running, the model will use the default values in [../graphinvent/parameters/defaults.py](../graphinvent/parameters/defaults.py), but it is not always the case that the "default" values will work well for your dataset. For instance, the parameters related to the learning rate decay are strongly dependent on the dataset used, and you might have to tune them to get optimal performance using your dataset. Depending on your system, you might also need to tune the mini-batch and/or block size so as to reduce/increase the memory requirement for training jobs.

You can then run a GraphINVENT2 reinforcement learning job from the terminal using the following command with your updated submission script:

```
$ python submit.py
```

As the models are training, you should see the progress bar updating on the terminal every epoch.

### Generating structures using the newly trained models
Once you have trained a model, you can use a saved state (e.g., *model_restart_70.pth*) to generate molecules. To do this, *submit.py* needs to be updated to specify a generation job. The first setting that needs to be changed is the *job_type*; all other settings here should be kept fixed so that the program can find the correct job directory:

```
submit.py >
        # define what you want to do for the specified job(s)
        self.dataset          = "your_dataset_name"  # dataset name in "./data/pre-training/"
        self.job_type         = "generate"           # this tells the code that this is a generation job
        self.jobdir_start_idx = 0                    # where to start indexing job dirs
        self.n_jobs           = 1                    # number of jobs to run per model
        self.restart          = False                # whether or not this is a restart job
        self.force_overwrite  = True                 # overwrite job directories which already exist
        self.jobname          = "train"              # sub-directory where the saved model can be found
```

You will then need to update the *generation_epoch* and *n_samples* parameter in *submit.py*:

```
submit.py >
        # define dataset-specific parameters
        self.params = {
            "generation_epoch": 70,   # <-- specify which saved model (i.e., at which epoch) to use for training)
            "n_samples": 30000,       # <-- specify how many structures you want to generate
        }
```

The *generation_epoch* should correspond to the saved model state that you want to use for generation, and *n_samples* tells the program how many structures you want to generate. In the example above, the parameters specify that the model saved at Epoch 70 should be used to generate 30,000 structures. All other parameters should be kept the same (if they are related to training, such as *epochs* or *init_lr*, they will be ignored during generation jobs).

Structures will be generated in batches of size *batch_size*. If you encounter memory problems during generation jobs, reducing the batch size should once again solve them. Generated structures, along with their corresponding metadata, will be written to the *generation/* directory within the existing job directory. These files are:

* *epochGEN{generation_epoch}_{batch}.smi*, containing molecules generated at the epoch specified
* *epochGEN{generation_epoch}_{batch}.nll*, containing their respective NLLs
* *epochGEN{generation_epoch}_{batch}.valid*, containing their respective validity (0: invalid, 1: valid)

Additionally, the *generation.log* file will be updated with the various evaluation metrics for the generated structures.

If you've followed the tutorial up to here, it means you can successfully create new molecules using a GNN-based model fine-tuned using reinforcement learning.

### Creating a new loss function
To define a new scoring function, you can update the class [graphinvent/ScoringFunction.py](graphinvent/ScoringFunction.py). Within this class, you will want to modify the `get_contributions_to_score` function to add your new function to the for loop, as illustrated by the following code snippet:

```
    def get_contributions_to_score(self, graphs : list) -> list:
        """
        Returns the different elements of the score.

        Args:
        ----
            graphs (list) : Contains molecular graphs to evaluate.

        Returns:
        -------
            contributions_to_score (list) : Contains elements of the score due to
                                            each scoring function component.
        """
        contributions_to_score = []

        for score_component in self.score_components:
            if "target_size" in score_component:
                .
				.
				.

            elif score_component == "QED":
                .
				.
				.

            elif "activity" in score_component:
                .
				.
				.

			elif "MY_NEW_FUNCTION" in score_component:  # <-- new function added as an elif statement here
                .
				.  # you can write code for what you want your function to do here
				.

            else:
                raise NotImplementedError("The score component is not defined. "
                                          "You can define it in "
                                          "`ScoringFunction.py`.")

        return contributions_to_score
```

Above, we are adding a new function called "MY_NEW_FUNCTION" to the for loop.

Hopefully it is straightforward, if not, please make an "issue" so we can address it ASAP.

#### (Optional) Postprocessing

To make things more convenient for any subsequent analyses, you can concatenate all structures generated in different batches into one file using:

```
for i in epochGEN{generation_epoch}_*.smi; do cat $i >> epochGEN{generation_epoch}.smi; done
```

Above, *{generation_epoch}* should be replaced with a number corresponding to a valid epoch. You can do similar things for the NLL and validity files, as the rows in those files correspond to the rows in the SMILES files.

Note that "Xe" and empty graphs may appear in the generated structures, even if the models are well-trained, as there is always a small probability of sampling invalid actions. If you do not want to include invalid entries in your analysis, these can be filtered out by typing:

```
sed -i "/Xe/d" path/to/file.smi          # remove "Xe" placeholders from file
sed -i "/^ [0-9]\+$/d" path/to/file.smi  # remove empty graphs from file
```

See [3_visualizing_molecules](./3_visualizing_molecules.md) for examples on how to draw grids of molecules.

### Summary
Hopefully you are now able to fine-tune models using reinforcement learning using GraphINVENT2 pre-trained on custom datasets. If anything is unclear in this tutorial, or if you have any questions that have not been addressed by this guide, feel free to contact the authors for assistance (best via a GitHub issue in the GraphINVENT2 repository).

We look forward to seeing the molecules you've generated using GraphINVENT2.
