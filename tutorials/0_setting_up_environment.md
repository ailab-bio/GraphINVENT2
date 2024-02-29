## Setting up the environment
Before doing anything with GraphINVENT, you will need to configure the GraphINVENT virtual environment, as the code is dependent on very specific versions of packages. You can use [conda](https://docs.conda.io/en/latest/) for this. If you do not have conda installed, please check out the installation instructions at [this link](https://docs.anaconda.com/free/miniconda/).

GraphINVENT requires PyTorch, RDKit, and XXX to run. A virtual environment can be easily created using conda by typing into the terminal:

```
conda create -n graphinvent python=3.9 pytorch::pytorch torchvision torchaudio -c pytorch
```

The above command might be Mac specific, in which case you might need to change the syntax a bit to install PyTorch on a different OS (see the [PyTorch documentation](https://pytorch.org/get-started/locally/)).

Then, to activate the environment:

```
conda activate graphinvent
```

To install additional packages to the virtual environment, like RDKit, tqdm, h5py, scikit-learn, matplotlib, and tensorboard, use:

```
conda install -n graphinvent conda-forge::rdkit conda-forge::tqdm anaconda::h5py anaconda::scikit-learn matplotlib tensorboard
```

To install additional packages to the virtual environment, should the need arise, use:

```
conda install -n graphinvent {package_name}
```

And that's it! To learn how to start training models, go to [1_introduction](1_introduction.md).


### Possible issues
If you are getting the following error message when you try to run the submission script:

*Fatal Python error: config_get_locale_encoding: failed to get the locale encoding: nl_langinfo(CODESET) failed*

try setting LANG and LC_CTYPE with a locale:
```
export LANG="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
```
