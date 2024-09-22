## Benchmarking

### Benchmarking models using the sample efficiency (recommended)
For benchmarking molecular generative models, we recommend the sample efficiency benchmark proposed by Gao et al. in [Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization](https://arxiv.org/abs/2206.12411), code for which is also available [here](https://github.com/wenhao-gao/mol_opt).

### Benchmarking models using chemical space coverage
Another option for benchmarking models is using the chemical space coverage. The idea behind this metric is simple: with a subset of an enumerated database, like GDB-13 (975 million molecules), how much of the full database can be sampled given that a model has only been trained on a small fraction of structures from that database (e.g., 0.1%). This concept is illustrated in the following publications:
* Arús-Pous et al. (2019) [Exploring the GDB-13 Chemical Space Using Deep Generative Models](https://doi.org/10.1186/s13321-019-0341-z),
* and Zhang et al. (2021) [Comparative Study of Deep Generative Models on Chemical Space Coverage](https://doi.org/10.1021/acs.jcim.0c01328).

### Benchmarking models with MOSES
Models can also be easily benchmarked using MOSES, though this is a bit of an "outdated" benchmark and is not recommended (though it was nice when it came out). To do this, we recommend reading the MOSES documentation, available at https://github.com/molecularsets/moses. If you want to compare to previously benchmarked models, you will need to train models using the MOSES datasets, available [here](https://github.com/molecularsets/moses/tree/master/data).

Once you have a satisfactorily trained model, you can run a Generation job to create 30,000 new structures (see [2_using_a_new_dataset](./2_using_a_new_dataset.md) and follow the instructions using the MOSES dataset). The generated structures can then be used as the \<generated dataset\> in MOSES evaluation jobs.

From our experience, MOSES benchmarking jobs require c.a. 30 GB RAM and are done in about an hour.
