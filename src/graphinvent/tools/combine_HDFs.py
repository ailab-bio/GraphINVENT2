"""
Combines preprocessed HDF files. Useful when preprocessing large datasets, as
one can split the `{split}.smi` into multiple files (and directories), preprocess
them separately, and then combine using this script.

To use script, modify the variables below to automatically create a list of
paths **assuming** HDFs were created with the following directory structure:
 data/
  |-- {dataset}_1/
  |-- {dataset}_2/
  |-- {dataset}_3/
  |...
  |-- {dataset}_{n_dirs}/

The variables are also used in setting the dimensions of the HDF datasets later on.

If directories were not named as above, then simply replace `path_list` below
with a list of the paths to all the HDFs to combine.

Then, run:
python combine_HDFs.py
"""

import csv
from typing import Union

import h5py
import numpy as np
import torch


def load_ts_properties_from_csv(csv_path: str) -> Union[dict, None]:
    """
    Loads CSV file containing training set properties and returns contents as a dictionary.
    """
    print("* Loading training set properties.", flush=True)

    # read dictionaries from csv
    try:
        with open(csv_path, "r") as csv_file:
            reader = csv.reader(csv_file, delimiter=";")
            csv_dict = dict(reader)
    except Exception:
        return None

    # fix file types within dict in going from `csv_dict` --> `properties_dict`
    properties_dict = {}
    for key, value in csv_dict.items():

        # first determine if key is a tuple
        key = eval(key)
        if len(key) > 1:
            tuple_key = (str(key[0]), str(key[1]))
        else:
            tuple_key = key

        # then convert the values to the correct data type
        try:
            properties_dict[tuple_key] = eval(value)
        except (SyntaxError, NameError):
            properties_dict[tuple_key] = value

        # convert any `list`s to `torch.Tensor`s (for consistency)
        if isinstance(properties_dict[tuple_key], list):
            properties_dict[tuple_key] = torch.Tensor(properties_dict[tuple_key])

    return properties_dict


def write_ts_properties_to_csv(ts_properties_dict: dict) -> None:
    """
    Writes the training set properties in `ts_properties_dict` to a CSV file.
    """
    dict_path = f"data/{dataset}/{split}.csv"

    with open(dict_path, "w") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=";")
        for key, value in ts_properties_dict.items():
            if "validity_tensor" in key:
                continue  # skip writing the validity tensor because it is really long
            elif isinstance(value, np.ndarray):
                csv_writer.writerow([key, list(value)])
            elif isinstance(value, torch.Tensor):
                try:
                    csv_writer.writerow([key, float(value)])
                except ValueError:
                    csv_writer.writerow([key, [float(i) for i in value]])
            else:
                csv_writer.writerow([key, value])


def get_dims(path: str) -> dict:
    """
    Reads the per-subgraph dimensions of each dataset from an existing HDF file.

    Recomputing them from hand-set atom/charge/bond counts, as this used to,
    silently omitted the implicit-hydrogen and chirality segments of the node
    feature vector, which are present under every shipped configuration; the
    resulting datasets had the wrong width and writing real blocks into them
    raised a broadcast error.  Reading the shapes back makes the tool correct
    for any feature vocabulary, with no parameters to keep in sync.
    """
    dims = {}
    with h5py.File(path, "r") as hdf_file:
        for name in hdf_file.keys():
            dims[name] = list(hdf_file[name].shape[1:])
    return dims


def get_total_n_subgraphs(paths: list) -> int:
    """
    Gets the total number of subgraphs saved in all the HDF files in the `paths`,
    where `paths` is a list of strings containing the path to each HDF file we want
    to combine.
    """
    total_n_subgraphs = 0
    for path in paths:
        print("path:", path)
        hdf_file = h5py.File(path, "r")
        nodes = hdf_file.get("nodes")
        n_subgraphs = nodes.shape[0]
        total_n_subgraphs += n_subgraphs
        hdf_file.close()

    return total_n_subgraphs


def main(paths: list, training_set: bool) -> None:
    """
    Combine many small HDF files (their paths defined in `paths`) into one large HDF file.
    """
    total_n_subgraphs = get_total_n_subgraphs(paths)
    dims = get_dims(paths[0])

    def _dtype(name: str) -> np.dtype:
        # Must match DataProcessor._dataset_dtype: action counts need int16 and
        # condition vectors are float32.
        if name == "condition_vector":
            return np.dtype("float32")
        if name == "action_probs":
            return np.dtype("int16")
        return np.dtype("int8")

    print(f"* Creating HDF file to contain {total_n_subgraphs} subgraphs")
    new_hdf_file = h5py.File(f"data/datasets/{dataset}/{split}.h5", "a")
    new_datasets = {
        name: new_hdf_file.create_dataset(
            name, (total_n_subgraphs, *shape), dtype=_dtype(name)
        )
        for name, shape in dims.items()
    }
    new_dataset_nodes = new_datasets["nodes"]
    new_dataset_edges = new_datasets["edges"]
    new_dataset_action_probs = new_datasets["action_probs"]

    print("* Combining data from smaller HDFs into a new larger HDF.")
    init_index = 0
    for path in paths:
        print("path:", path)
        hdf_file = h5py.File(path, "r")

        nodes = hdf_file.get("nodes")
        edges = hdf_file.get("edges")
        action_probs_data = hdf_file.get("action_probs")

        n_subgraphs = nodes.shape[0]

        new_dataset_nodes[init_index : (init_index + n_subgraphs)] = nodes
        new_dataset_edges[init_index : (init_index + n_subgraphs)] = edges
        new_dataset_action_probs[init_index : (init_index + n_subgraphs)] = (
            action_probs_data
        )

        init_index += n_subgraphs
        hdf_file.close()

    new_hdf_file.close()

    if training_set:
        print(f"* Combining data from respective `{split}.csv` files into one.")
        csv_list = [f"{path[:-2]}csv" for path in paths]

        ts_properties_old = None
        csv_files_processed = 0
        for path in csv_list:
            ts_properties = load_ts_properties_from_csv(csv_path=path)
            ts_properties_new = {}
            if ts_properties_old and ts_properties:
                for key, value in ts_properties_old.items():
                    if isinstance(value, float):
                        ts_properties_new[key] = (
                            value * csv_files_processed + ts_properties[key]
                        ) / (csv_files_processed + 1)
                    else:
                        new_list = []
                        for i, value_i in enumerate(value):
                            new_list.append(
                                float(
                                    value_i * csv_files_processed
                                    + ts_properties[key][i]
                                )
                                / (csv_files_processed + 1)
                            )
                        ts_properties_new[key] = new_list
            else:
                ts_properties_new = ts_properties
            ts_properties_old = ts_properties_new
            csv_files_processed += 1

        write_ts_properties_to_csv(ts_properties_dict=ts_properties_new)


if __name__ == "__main__":
    # combine the HDFs defined in `path_list`

    # set variables
    # Feature dimensions are read from the input files, so nothing needs to be
    # kept in sync with the preprocessing configuration here.
    dataset = "ChEMBL"

    # combine the training files
    n_dirs = 12  # how many times was `{split}.smi` split?
    split = "train"  # train, test, or valid
    path_list = [f"data/datasets/{dataset}_{i}/{split}.h5" for i in range(0, n_dirs)]
    main(path_list, training_set=True)

    # combine the test files
    n_dirs = 4  # how many times was `{split}.smi` split?
    split = "test"  # train, test, or valid
    path_list = [f"data/datasets/{dataset}_{i}/{split}.h5" for i in range(0, n_dirs)]
    main(path_list, training_set=False)

    # combine the validation files
    n_dirs = 2  # how many times was `{split}.smi` split?
    split = "valid"  # train, test, or valid
    path_list = [f"data/datasets/{dataset}_{i}/{split}.h5" for i in range(0, n_dirs)]
    main(path_list, training_set=False)

    print("Done.", flush=True)
