# GraphINVENT2

![cover image](./cover-image.png)

## Description
GraphINVENT2 is a platform for graph-based molecular generation and optimization. It uses a tiered deep neural network architecture to probabilistically generate new molecules a single bond at a time, and reinforcement learning to guide the model towards better molecules. GraphINVENT2 captures the core functionalities of GraphINVENT in a smaller, more user-friendly package.

This is not only a newer version of GraphINVENT, but it is also the only version being actively maintained. Please use this version in all new uses of GraphINVENT.

The methods used in GraphINVENT are described in the original publication, [*Graph Networks for Molecular Design*](https://iopscience.iop.org/article/10.1088/2632-2153/abcf91) by Mercado et al. (2021).

## Prerequisites
* Anaconda or Miniconda with Python 3.6 or 3.8.
* (for GPU-training only) CUDA-enabled GPU.

## Instructions and tutorials
For detailed guides on how to use GraphINVENT, see the [tutorials](./tutorials/).

## Examples
An example training set is available in [./data/gdb13_1K/](./data/gdb13_1K/). It is a small (1K) subset of GDB-13 and is already preprocessed.

## Contributions
Contributions are welcome in the form of issues or pull requests. To report a bug, please submit an issue. Thank you to everyone who has used the code and provided feedback thus far.

## References
If you use GraphINVENT2 in your research, please reference our original [publication](https://doi.org/10.1088/2632-2153/abcf91).

Additional details related to the development of GraphINVENT are available in our [technical note](https://doi.org/10.1002/ail2.18). You might find this note useful if you're interested in either exploring different hyperparameters or developing your own generative models.

The references in BibTex format are available below:

```
@article{mercado2020graph,
  author = "Rocío Mercado and Tobias Rastemo and Edvard Lindelöf and Günter Klambauer and Ola Engkvist and Hongming Chen and Esben Jannik Bjerrum",
  title = "{Graph Networks for Molecular Design}",
  journal = {Machine Learning: Science and Technology},
  year = {2020},
  publisher = {IOP Publishing},
  doi = "10.1088/2632-2153/abcf91"
}

@article{mercado2020practical,
  author = "Rocío Mercado and Tobias Rastemo and Edvard Lindelöf and Günter Klambauer and Ola Engkvist and Hongming Chen and Esben Jannik Bjerrum",
  title = "{Practical Notes on Building Molecular Graph Generative Models}",
  journal = {Applied AI Letters},
  year = {2020},
  publisher = {Wiley Online Library},
  doi = "10.1002/ail2.18"
}
```

## License

GraphINVENT is licensed under the MIT license and is free and provided as-is.

## Link
https://github.com/ailab-bio/GraphINVENT2
