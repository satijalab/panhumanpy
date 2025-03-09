"""
This module contains functions to read single cell transcriptomic data from 
a h5ad file. 

Functions contained herein are:
- h5ad_to_indexed_vector
- subset_sparse_matrix
- data_filepath
- training_genes
- data_csr_matrix
- ref_labels

ToDo: Add docstrings to the functions. 

"""


import os

import anndata
import h5py
import numpy as np


def h5ad_to_indexed_vector(file_path, array_key, indices_key):
    with h5py.File(file_path, "r") as f:
        array_data = f[array_key][:]
        indices = f[indices_key][:]
    ordered_data = array_data[indices]
    return ordered_data


def subset_sparse_matrix(adata, custom_genes):
    decoded_genes = [
        gene.decode("utf-8") for gene in custom_genes if isinstance(gene, bytes)
    ]
    gene_indices = [
        adata.var_names.get_loc(gene)
        for gene in decoded_genes
        if gene in adata.var_names
    ]
    subset_adata = adata[:, gene_indices]

    return subset_adata


def data_filepath(data_dir, dataset_h5ad):
    """
    New data_filepath fn pointing to objects in Sourav's workspace.
    Args:
        -data_dir (str): filepath to the dir where the data is stored in h5ad format.
    """

    ref_h5ad = os.path.join(data_dir, dataset_h5ad)

    return ref_h5ad


def load_training_genes(data_dir, features_txt):
    """
    write, we switch to this after adding pulmonary ionocytes
    """

    file_path = os.path.join(data_dir, features_txt)
    with open(file_path, "r") as f:
        training_genes = f.read().splitlines()
    training_genes = [gene.encode("utf-8") for gene in training_genes]
    training_genes = np.unique(np.array(training_genes))

    return training_genes


def data_csr_matrix(data_dir, dataset_h5ad, features_txt):
    """
    Returns scRNA-seq data as a csr matrix of shape (number of cells, number of training genes).

    Args:
        -data_dir (str): filepath to the dir where the data is stored in h5ad format.
    """

    ref_h5ad = data_filepath(data_dir, dataset_h5ad)
    training_genes = load_training_genes(data_dir, features_txt)

    ad = anndata.read_h5ad(ref_h5ad)
    subset_ad = subset_sparse_matrix(ad, custom_genes=training_genes)
    data_matrix = subset_ad.X

    return data_matrix


def ref_labels_fn(data_dir, dataset_h5ad, delimiter):
    """
    Args:
        -data_dir (str): filepath to the dir where the data is stored in h5ad format.
    """

    ref_h5ad = data_filepath(data_dir, dataset_h5ad)

    cell_type_labels = "parent_hierarchy"
    ct_categories = f"/obs/{cell_type_labels}/categories"
    ct_codes = f"/obs/{cell_type_labels}/codes"

    ref_labels = h5ad_to_indexed_vector(
        ref_h5ad, array_key=ct_categories, indices_key=ct_codes
    )
    ref_labels = [s.decode("utf-8") for s in ref_labels]
    if delimiter == "-":
        ref_labels = [s.replace(" - ", "|") for s in ref_labels]

    return ref_labels


if __name__ == "__main__":
    pass
