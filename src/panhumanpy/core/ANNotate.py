"""
Inference script using Azimuth Neural Network trained on
annotated panhuman scRNA-seq data.
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
from data_prep import *
from loss_fn import *
from postprocessing import *
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model, load_model

from data.kfold_data.read_data import load_training_genes

warnings.filterwarnings("ignore")


def give_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


script_dir = give_script_dir()
sys.path.append(script_dir)


def configure():
    """
    Configures TensorFlow GPU settings to optimize memory usage and
      performance.

    - Limits default memory allocation on the GPU to prevent excessive
    consumption.
    - Enables memory growth, allowing TensorFlow to allocate GPU memory
    as needed.
    - Sets the JIT (Just-In-Time) compilation flag to True for potential
      performance improvements.

    If GPUs are available, it prints the number of physical and logical
    GPUs detected.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs \n")
        except RuntimeError as e:
            print(e)
            print("\n")

    tf.config.optimizer.set_jit(True)


def data_dir(data_source, seed, data_split):
    """
    Construct the absolute directory path and dataset name for the data
    files.

    The function assumes that the data is stored in a directory relative
    to the directory of the training script (as returned by
    `give_script_dir()`) under an "experiments" folder. The final
    directory path is built from the base name of the data_source, the
    provided seed, and the data split values (train/valid/test).

    Args:
        data_source (str): Relative directory where the k-fold data is
            stored. It should not end with a trailing slash ('/').
        seed (int): The seed used for the train/validation/test split.
        data_split (list): A list of three integers representing the
            train, validation, and test splits.

    Returns:
        tuple[str, str]: A tuple where the first element is the complete
             directory path to read data from and the second element is
             the immediate sub-directory name (dataset name) containing
             the data files.

    Raises:
        ValueError: If `data_source` ends with a slash or if the
            extracted base name is empty, or if `data_split` does not
            have exactly three elements.
    """

    cwd = give_script_dir()
    data_source_id = data_source.rstrip("/").split("/")[-1]

    if not data_source_id:
        raise ValueError(
            "The data_source should be a non-empty string with no trailing '/'."
        )
    if len(data_split) != 3:
        raise ValueError("data_split must be a list of exactly three integers.")

    data_split_str = f"{data_split[0]}_" f"{data_split[1]}_" f"{data_split[2]}"
    dir_path = os.path.join(cwd, "experiments", data_source_id)

    dir_path = os.path.join(dir_path, f"data_{seed}_{data_split_str}")

    dataset_name = f"data_{seed}_{data_split_str}"

    return dir_path, dataset_name


def return_match(contents, templates):
    """
    Returns the first content string that starts with any of the
    provided template prefixes.

    This function iterates over the given list of templates. For each
    template, it checks each content in the 'contents' list to see if
    the content begins with the template. The first content that matches
      any template is returned.

    Args:
        contents (list of str): List of strings representing content
            (e.g., directory names) to search through.
        templates (list of str): List of template prefixes to match
            against the beginning of each content string.

    Returns:
        str: The first content string that matches one of the templates.

    Raises:
        ValueError: If the 'contents' list is empty or if no content
            matches any of the provided templates.
    """

    if len(contents) == 0:
        raise ValueError("No contents in the path provided, maybe somewhere else?")
    else:
        match = None
        for template in templates:
            template_len = len(template)

            for content in contents:
                if content[:template_len] == template:
                    match = content
                    break
        if not match:
            raise ValueError(
                "No matching directories found in the path provided, "
                "maybe somewhere else?"
            )

    return match


def give_exp_dir(
    split_mode,
    model,
    epochs,
    train_seed,
    loss_name,
    data_seed,
    data_source,
    data_split,
    mask_seed,
    tm_frac,
    lm_frac,
    batch_size,
    optimizer_name,
    lr,
    l1,
    l2,
    dropout,
    save,
    exist=False,
):
    """
    Returns the directory path with the saved model based on the
    arguments provided.
    """

    data_source_id = data_source.split("/")[-1]
    cwd = give_script_dir()
    exp_dir = os.path.join(cwd, "experiments/" + data_source_id)
    exp_dir = os.path.join(exp_dir, split_mode)
    exp_dir = os.path.join(exp_dir, model)
    exp_dir = os.path.join(exp_dir, loss_name)
    exp_data_source, dataset_name = data_dir(data_source, data_seed, data_split)
    exp_dir = os.path.join(exp_dir, dataset_name)

    if mask_seed:
        if if_val(tm_frac) + if_val(lm_frac) != 1:
            raise ValueError(
                "One and exactly one of tail_mask or single_level_mask"
                "needs to be provided."
            )
        if tm_frac:
            mask_mode = "tail"
            mask_subdir = f"tail_mask/tm_frac{tm_frac[0]}_{tm_frac[1]}"
        elif lm_frac:
            mask_mode = "single_level"
            mask_subdir = f"single_level_mask/lm_frac{lm_frac}"
        exp_dir = os.path.join(exp_dir, mask_subdir, f"/mask_seed_{mask_seed}")
        mask_dir = exp_dir
    else:
        mask_mode = None
        mask_subdir = "no_mask"
        exp_dir = os.path.join(exp_dir, mask_subdir)
        mask_dir = None

    if save:
        exp_dir = os.path.join(exp_dir, "saved_models")

    exp_name = f"TS{train_seed}_BS{batch_size}_optim_{optimizer_name}_lr{lr}"

    exp_name_0 = f"{exp_name}_l2{l2}_dropout{dropout}_epochs{epochs}"
    exp_name_1 = f"{exp_name}_l1{l1}_l2{l2}_dropout{dropout}_epochs{epochs}"

    exp_names = [exp_name_0, exp_name_1]  # updatable

    if exist:
        contents = os.listdir(path=exp_dir)
        matching_dirname = return_match(contents, exp_names)
        exp_dir = os.path.join(exp_dir, matching_dirname)
    else:
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M")

        exp_name = exp_names[-1]
        exp_name = exp_name + "_" + date_time

        exp_dir = os.path.join(exp_dir, exp_name)

    return (exp_dir, exp_data_source, dataset_name, mask_mode, mask_dir, mask_subdir)


def read_obj(query_path, feature_names_col):
    """
    Reads an h5ad file and return a CSR matrix of the counts data, a
    list of gene names, a cell metadata DataFrame, and the full AnnData
    object.

    This function reads an AnnData (.h5ad) file from the given path,
    extracts the feature (gene) names from a specified column in the
    'var' slot (or uses the default var_names if no column is
    specified), and converts the main data matrix (X) into a CSR matrix
    if necessary. It also verifies that the cell metadata in 'obs' is
    available as a pandas DataFrame.

    Args:
        query_path (str): Path to the input .h5ad file.
        feature_names_col (str or None): Column name in query.var to
            extract feature names. If None, query.var_names is used.

    Returns:
        tuple: A tuple containing:
            - X_query (csr_matrix): The feature matrix in CSR format.
            - query_features (list): A list of gene/feature names.
            - cells_df (pandas.DataFrame): The cell metadata from
                    query.obs.
            - query (anndata.AnnData): The full AnnData object read
                    from query_path.

    Raises:
        ValueError: If the file extension is not 'h5ad', or if the cell
                metadata is not a pandas DataFrame.
    """

    filetype = query_path.split(".")[-1]
    if filetype != "h5ad":
        raise ValueError("File type should be h5ad.")

    query = anndata.read_h5ad(query_path)

    if feature_names_col:
        query_features = query.var[feature_names_col].tolist()
    else:
        query_features = query.var_names.tolist()

    cells_df = query.obs
    assert (
        str(type(cells_df)) == "<class 'pandas.core.frame.DataFrame'>"
    ), "Query cell meta is not available as dataframe, fix this."

    X_query = query.X
    if str(type(X_query)).split(".")[-1][:-2] == "coo_matrix":
        X_query = X_query.tocsr()
    elif str(type(X_query)).split(".")[-1][:-2] == "ndarray":
        X_query = csr_matrix(X_query)

    return X_query, query_features, cells_df, query


def reorder_subset_h5ad(X_query, query_features, training_genes):
    """
    Reorder and subset the query feature matrix to match the training
    genes list.

    This function adjusts the query feature matrix (X_query) so that its
      columns are ordered according to the training gene list. It first
      converts any byte-encoded gene names in the training set to UTF-8
      strings and creates a 'template' list. It then determines the
      common features between the query and the training genes. For any
      training gene missing from the query, a zero-filled column is
      appended to the query matrix. Finally, the matrix columns are
      reordered to match the order in the template, and diagnostic
      information is printed.

    Args:
        X_query (scipy.sparse matrix): Sparse matrix (e.g., CSR matrix)
            of query data, where each row corresponds to a cell and each
              column to a feature.
        query_features (list of str): List of feature names
            corresponding to the columns of X_query.
        training_genes (iterable): Iterable of gene identifiers
            (bytes or str) for the training set. Byte strings will be
            decoded to UTF-8.

    Returns:
        tuple: A tuple containing:
            - reordered_X_query (scipy.sparse matrix): The query matrix
                    with columns reordered (and zero-padded for missing
                    features) to match training_genes.
            - num_cells (int): The number of cells (rows) in X_query.
            - template (list of str): The training gene names as
                    strings, in the order specified.
            - list(common_features): The list of gene names found in
                    both query_features and training_genes.

    Raises:
        AssertionError: If the number of columns in the reordered matrix
             does not equal the length of the training gene list.
    """

    template = [
        gene.decode("utf-8") if isinstance(gene, bytes) else gene
        for gene in training_genes
    ]
    common_features = set(query_features).intersection(set(template))
    extra_features = set(template) - common_features

    num_cells = X_query.shape[0]

    zero_columns = csr_matrix((num_cells, len(extra_features)))
    X_query = hstack([X_query, zero_columns])
    query_features.extend(extra_features)

    reordered_query_indices = [query_features.index(name) for name in template]
    reordered_X_query = X_query[:, reordered_query_indices]

    assert reordered_X_query.shape[1] == len(
        template
    ), "Feature dimension mismatch after reordering and subsetting the query."

    print("Query object:")
    print(f"    Total number of cells: {num_cells}")
    print(f"    Total number of features: {len(set(query_features))}")
    print(
        f"    Overlap w. feature reference: {len(common_features)} "
        f"({int(len(common_features)*100/len(set(template)))}%)\n"
    )

    return reordered_X_query, num_cells, template, list(common_features)


def extract_X(X_query, query_features, source_data_dir, features_txt):
    """
    Extracts scRNA-seq data as a compressed sparse row (CSR) matrix with
    dimensions (number of cells, number of training genes), ensuring
    alignment between query features and training genes.

    Args:
        X_query (csr_matrix or ndarray): The gene expression matrix of
            query cells.
        query_features (list): A list of gene names corresponding to the
             features in X_query.
        source_data_dir (str): Directory path where the reference
            training data is stored.
        features_txt (str): Filename of the text file containing the
            list of training genes.

    Returns:
        csr_matrix: The processed gene expression matrix aligned with
            training genes.
        int: The number of cells in the dataset.
        list: The ordered list of template (training) gene features.
        list: The list of common features between query and training
            data.
    """

    training_genes = load_training_genes(source_data_dir, features_txt)
    data_matrix, num_cells, template_features, common_features = reorder_subset_h5ad(
        X_query, query_features, training_genes
    )

    assert (
        data_matrix.shape[1] == training_genes.shape[0]
    ), "Shape of X is not consistent with the number of features."

    return data_matrix, num_cells, template_features, common_features


def check_normalization(matrix, normalization_override, norm_check_batch_size):
    """
    write docstring
    """

    mat = matrix[:norm_check_batch_size, :]

    if normalization_override:
        return True
    else:
        mat = mat.toarray()
        mat_floor = np.floor(mat)

        if np.any((mat_floor - mat) != 0.0):
            return True

        else:
            return False


def normalize(mat, normalization_override, norm_check_batch_size=100):
    """
    Normalizes a gene expression count matrix using log1p transformation.

    This function first checks whether the provided matrix is already
    normalized by inspecting a batch of cells (specified by
    norm_check_batch_size). If the matrix is not normalized and
    normalization_override is False, the function scales each cell to
    10,000 total counts and applies a log1p transformation to the data.
    The input matrix is assumed to be a sparse matrix (e.g., CSR format)
    that supports the operations .sum(), .multiply(), and .tocsr().

    Args:
        mat (scipy.sparse matrix): Gene expression count matrix with
            shape (num_cells, num_genes).
        normalization_override (bool): If True, assumes that the matrix
            is already normalized and bypasses the scaling and log1p
            transformation.
        norm_check_batch_size (int, optional): The number of cells used
            to determine whether the matrix is normalized. Default is
            100.

    Returns:
        scipy.sparse.csr_matrix: The normalized gene expression matrix.
        boolen: Whether the provided matrix was normalized or not
            intially.
    """

    check_norm = check_normalization(mat, normalization_override, norm_check_batch_size)

    if not check_norm:
        total_counts = mat.sum(axis=1)
        total_counts = np.array(total_counts).reshape(-1, 1)

        total_counts[total_counts == 0] = 1
        if 0 in total_counts.flatten():
            print("Warning: Cells with 0 counts across the entire gene " "panel found.")
        scaled_mat = mat.multiply(10000 / total_counts)
        scaled_mat.data = np.log1p(scaled_mat.data)
        mat = scaled_mat.tocsr()

    return mat, check_norm


def labels_array(X_query, model, label_encoders, eval_batch_size, max_depth_given=None):
    """
    Generates predicted labels and probabilities for scRNA-seq data
    using the trained Azimuth Neural Network model.

    This function processes the query dataset in batches, predicts
    hierarchical labels, and retrieves their corresponding softmax
    probabilities. The function handles multi-depth label prediction,
    supporting both predefined and automatic depth detection.

    Args:
        X_query (scipy.sparse matrix or numpy.ndarray): Query gene
            expression data.
        model (tensorflow/keras model): Trained model for predicting
            hierarchical labels.
        label_encoders (list of sklearn.preprocessing.LabelEncoder): A
            list of encoders to convert predicted indices back to
            categorical string labels at each level of hierarchy.
        eval_batch_size (int): Number of cells to process per batch
            during inference.
        max_depth_given (int or str, optional): The number of
            hierarchical levels to predict. If set to an integer, that
            depth is used. If set to 'autocorrect_for_depth',
            the function determines depth automatically by ignoring the
            depth outputs at the final level.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Predicted string labels of shape
                (num_cells, max_depth).
            - np.ndarray: Predicted integer label indices of shape
                (num_cells, max_depth).
            - np.ndarray: Prediction confidence scores of shape
                (num_cells, max_depth).
            - int: Maximum depth of the hierarchical classification.
            - list of np.ndarray: Softmax outputs at each hierarchy
                level.

    Raises:
        AssertionError: If the shape of predicted outputs does not match max_depth.
    """

    pred_batches = []
    prob_batches = []
    num_cells = X_query.shape[0]
    eval_steps = num_cells // eval_batch_size

    if isinstance(max_depth_given, int):
        max_depth = max_depth_given
    elif max_depth_given == "autocorrect_for_depth":
        max_depth = len(model.predict(X_query[:2].toarray(), verbose=0)) - 1
    else:
        max_depth = len(model.predict(X_query[:2].toarray(), verbose=0))
    softmax_outputs = [[] for _ in range(max_depth)]

    print(f"Splitting query data into {eval_steps+1} evaluation batches.\n")

    print("Running model:")
    for i in range(eval_steps):
        pred_batches_cache = []
        prob_batches_cache = []
        X_batch = X_query[i * eval_batch_size : (i + 1) * eval_batch_size].toarray()
        y_pred_batch = model.predict(X_batch)

        for i in range(max_depth):
            level = y_pred_batch[i]
            softmax_outputs[i].append(level)
            pred_batch_level = tf.expand_dims(
                tf.cast(tf.argmax(level, axis=-1), tf.int32), axis=1
            )
            prob_batch_level = tf.reduce_max(level, axis=-1, keepdims=True)
            pred_batches_cache.append(pred_batch_level)
            prob_batches_cache.append(prob_batch_level)
        pred_batch = tf.concat(pred_batches_cache, axis=1)
        prob_batch = tf.concat(prob_batches_cache, axis=1)
        pred_batches.append(pred_batch)
        prob_batches.append(prob_batch)

    last_batch = X_query[eval_steps * eval_batch_size :].toarray()
    y_pred_final_batch = model.predict(last_batch)
    pred_batches_cache = []
    prob_batches_cache = []
    for i in range(max_depth):
        level = y_pred_final_batch[i]
        softmax_outputs[i].append(level)
        pred_batch_level = tf.expand_dims(
            tf.cast(tf.argmax(level, axis=-1), tf.int32), axis=1
        )
        prob_batch_level = tf.reduce_max(level, axis=-1, keepdims=True)
        pred_batches_cache.append(pred_batch_level)
        prob_batches_cache.append(prob_batch_level)
    pred_batch = tf.concat(pred_batches_cache, axis=1)
    prob_batch = tf.concat(prob_batches_cache, axis=1)
    pred_batches.append(pred_batch)
    prob_batches.append(prob_batch)

    pred_all = tf.concat(pred_batches, axis=0)
    prob_all = tf.concat(prob_batches, axis=0)

    assert max_depth == pred_all.shape[1], (
        "The array of all predictions should have dimensions consistent "
        "with max_depth."
    )
    if pred_all.shape[0] != num_cells:
        print("Warning: number of predicted labels != number of cells.")

    softmax_outputs = [np.concatenate(level, axis=0) for level in softmax_outputs]
    for level in softmax_outputs:
        assert level.shape[0] == num_cells, (
            "number of cells in softmax_outputs does not match the "
            "number of cells in dataset."
        )

    string_labels_out = []

    for i in range(max_depth):
        string_labels_out.append(label_encoders[i].inverse_transform(pred_all[:, i]))

    return (
        np.array(string_labels_out).T,
        np.array(pred_all),
        np.array(prob_all),
        max_depth,
        softmax_outputs,
    )


def embedding_extractor_model(model, embedding_layer):
    """
    Auxilliary model to extract embeddings from a specified intermediate
    layer of the Azimuth Neural Network model.
    """
    embedding = model.get_layer(embedding_layer).output
    embedding_model = Model(inputs=model.input, outputs=embedding)

    return embedding_model


def embeddings(X_query, model, embedding_layer, eval_batch_size):
    """
    Returns embeddings from the specified intermediate layer of the
    Azimuth Neural Network model using the auxilliary model provided
    by the function embedding_extractor_model(model, embedding_layer).
    """
    embedding_batches = []
    eval_steps = X_query.shape[0] // eval_batch_size

    embedding_model = embedding_extractor_model(model, embedding_layer)

    for i in range(eval_steps):
        X_batch = X_query[i * eval_batch_size : (i + 1) * eval_batch_size].toarray()
        embeddings_batch = embedding_model.predict(X_batch)
        embedding_batches.append(embeddings_batch)

    embeddings_final_batch = embedding_model.predict(
        X_query[eval_steps * eval_batch_size :].toarray()
    )
    embedding_batches.append(embeddings_final_batch)

    embeddings = tf.concat(embedding_batches, axis=0)

    return embeddings


def comb_label(array_label, depth, max_depth):
    """
    Combine hierarchical label components from a list of label strings.

    This function takes a list of label strings (each expected to
    contain hierarchical components separated by the '|' character) and
    produces a single concatenated label. For each hierarchy level
    (from 0 to depth-1), the function looks for the first non-empty
    component among a subset of the labels in `array_label`. The search
    for each level starts at index i and considers up to (max_depth - i)
      subsequent elements. Once a non-empty component is found, it is
      appended to the output string, followed by a '|' delimiter.
      After processing all levels, the trailing delimiter is removed.

    Note:
        - Some discordant cases may not be flagged and will be merged
            into the output label.
        - This function assumes that each element in array_label
            contains the same number of '|' delimiters (i.e. a
            consistent hierarchical structure).

    Args:
        array_label (list of str): List of hierarchical label strings.
            Each string should contain components separated by '|'.
        depth (int): Number of hierarchical levels to combine.
        max_depth (int): Maximum number of label levels available in
            array_label.

    Returns:
        str: A concatenated label string built from the appropriate
            components.
    """
    out = ""
    for i in range(depth):
        buffer = max_depth - i
        for j in range(buffer):
            add = array_label[i + j].split("|")[i]
            if add != "":
                out += add
                out += "|"
                break
    out = out[:-1]

    return out


def load_cell_types():
    """
    Loads list of all cell types stored in cell_types.txt.
    """
    cwd = give_script_dir()
    cell_types_file = os.path.join(cwd, "cell_types.txt")
    with open(cell_types_file, "r") as f:
        cell_types_enc = f.read()
        if isinstance(cell_types_enc, bytes):
            cell_types_enc.decode("utf-8")

    cell_types_all = cell_types_enc.split("\n")[:-1]

    return cell_types_all


def give_embedding_layers(embeddings_mode, model):
    """
    Provides dictionary with embedding layer name(s).
    """
    if embeddings_mode == "shallow":
        if model == "M0.1_cumul":
            embedding_layers = {"shallow_embedding_layer": "dense_2"}
        elif model == "M0.2":
            embedding_layers = {"shallow_embedding_layer": "dense_3"}
        elif model == "M0.2_depth":
            embedding_layers = {"shallow_embedding_layer": "dense_3"}
        # else: add other models
    elif embeddings_mode == "deep":
        if model == "M0.1_cumul":
            embedding_layers = {"deep_embedding_layer": "dense_17"}
        elif model == "M0.2":
            embedding_layers = {"deep_embedding_layer": "dense_18"}
        elif model == "M0.2_depth":
            embedding_layers = {"deep_embedding_layer": "dense_22"}
        # else: add other models
    elif embeddings_mode == "both":
        if model == "M0.1_cumul":
            embedding_layers = {
                "shallow_embedding_layer": "dense_2",
                "deep_embedding_layer": "dense_17",
            }
        elif model == "M0.2":
            embedding_layers = {
                "shallow_embedding_layer": "dense_3",
                "deep_embedding_layer": "dense_18",
            }
        elif model == "M0.2_depth":
            embedding_layers = {
                "shallow_embedding_layer": "dense_3",
                "deep_embedding_layer": "dense_22",
            }
        # else: add other models

    return embedding_layers


def abs_labels(hierarchical_labels_array, max_depth):
    """
    Compute absolute hierarchical labels for each cell up to each level.

    This function processes an array of hierarchical labels (one per
    cell) and computes an "absolute" label for each hierarchy level from
      1 up to max_depth. For each level, the function combines label
      components using the comb_label function. If the combined label
      for a cell does not contain enough components (i.e. the number of
      '|' delimiters is less than the expected level), the label is
      replaced with "NA".

    Args:
        hierarchical_labels_array (list): A list of hierarchical labels,
             where each element represents the label of a cell. Each
             cell label should be in a format that is compatible with
             the comb_label function (e.g., a list or string with
             components separated by '|').
        max_depth (int): The maximum number of hierarchical levels
            available in the labels.

    Returns:
        list: A list of length max_depth. The i-th element is a list of
            absolute labels (strings) for each cell at hierarchical
            level i+1.
    """

    abs_labels_upto_level = [[] for i in range(max_depth)]
    for i in range(max_depth):
        for cell_label in hierarchical_labels_array:
            combined_label = comb_label(cell_label, i + 1, max_depth)
            if len(combined_label.split("|")) < i + 1:
                combined_label = "NA"
            abs_labels_upto_level[i].append(combined_label)

    return abs_labels_upto_level


def split_labels_w_final_level(hierarchical_labels_array, max_depth):
    """
    Split hierarchical labels and extract final level annotations.

    This function processes an array of hierarchical labels for cells,
    where each cell's label is expressed as a series of hierarchical
    components separated by the '|' character. It performs the following
      steps:

      1. Uses the `abs_labels` function to generate absolute labels for
            each hierarchical level (from 1 to max_depth) for each cell.
      2. Constructs a 2D array of these labels (cells x levels) and
            determines the deepest level (final level) for each cell by
            counting non-'NA' entries.
      3. Appends the computed final level (1-indexed) as a new row to
            the list of absolute labels.
      4. For each hierarchical level, extracts only the final component
            (i.e., the substring after the last '|' delimiter).
      5. Extracts the final level label for each cell using the computed
             final level indices and appends this as an additional
             element.

    Args:
        hierarchical_labels_array (list): A list (or iterable) where
            each element corresponds to a cell's hierarchical label.
            Each cell's label is itself a list (or similar iterable) of
            label strings for each level.
        max_depth (int): The maximum number of hierarchical levels
            present in the labels.

    Returns:
        tuple: A tuple containing:
            - abs_labels_list (list): A list where:
                * The first max_depth elements are lists of the final
                    label
                  components for each level (i.e., the part after the
                    last '|' delimiter).
                * The (max_depth+1)-th element is a list of the computed
                     final levels (1-indexed) for each cell.
                * The (max_depth+2)-th element is a list of the final
                    level labels for each cell.
            - final_levels_arr (numpy.ndarray): A 1D numpy array of
                shape (num_cells,) containing the zero-indexed final
                level for each cell, computed as the number of non-'NA'
                levels minus one.
    """
    abs_labels_list = abs_labels(hierarchical_labels_array, max_depth)
    abs_labels_array = np.array(abs_labels_list).T

    na_mask = abs_labels_array != "NA"
    final_levels_arr = np.sum(na_mask, axis=1) - 1
    final_levels_list = list(final_levels_arr + 1)

    abs_labels_list.append(final_levels_list)

    for i in range(max_depth):
        abs_labels_list[i] = [label.split("|")[-1] for label in abs_labels_list[i]]

    final_level_labels = np.array(abs_labels_list).T[
        np.arange(len(final_levels_list)), final_levels_arr
    ]
    final_level_labels = list(final_level_labels)

    abs_labels_list.append(final_level_labels)

    return abs_labels_list, final_levels_arr


def compute_scores(
    cell_idx,
    indices,
    distances,
    max_depth_query,
    max_depth,
    labels_by_level_w_final_level,
    weight=True,
    sd=10,
):
    """
    Compute consensus scores for a cell's hierarchical labels based on
    its neighbors.

    For a given cell (specified by cell_idx), this function computes a
    score for each hierarchical level (from level 1 up to
    max_depth_query) by comparing the cell's predicted label at that
    level to the labels of its nearest neighbors. The neighbors'
    contributions are weighted either by a Gaussian function of their
    distances (if weight=True) or uniformly otherwise. If a cell's
    predicted label at a level is 'NA', the score for that level is set
    to NaN. Finally, the function sets a final score (at index
    max_depth_query) based on the cell's final level annotation.

    The neighbor indices and distances arrays are assumed to contain,
    for each cell, a list of its nearest neighbor indices and
    corresponding distances. The first neighbor (index 0) is assumed to
    be the cell itself and is skipped.

    Args:
        cell_idx (int): The index of the cell for which to compute the
            scores.
        indices (array-like): An array (or list of arrays) of neighbor
            indices for each cell.
        distances (array-like): An array (or list of arrays) of
            distances corresponding to the neighbor indices.
        max_depth_query (int): The number of hierarchical levels for
            which to compute scores.
        max_depth (int): The maximum depth available in
            labels_by_level_w_final_level. The final level labels are
            stored at this index.
        labels_by_level_w_final_level (list): A list where each element
            is an array-like of predicted labels for a given
            hierarchical level for all cells, with an additional element
              for the final level.
        weight (bool, optional): If True, apply a Gaussian weighting
            based on neighbor distances; otherwise, assign equal weight
            to each neighbor. Defaults to True.
        sd (float, optional): The standard deviation used in the
            Gaussian weighting function. Defaults to 10.

    Returns:
        numpy.ndarray: An array of consensus scores with shape
            (max_depth_query + 1,). The first max_depth_query elements
            represent scores for each hierarchical level, and the last
            element is the score corresponding to the cell's final level
              (derived from the final level label).

    Note:
        - If a cell's predicted label at a given level is 'NA', that
            level's score is set to NaN.
    """
    cell_neighbors = indices[cell_idx][1:]
    cell_distances = distances[cell_idx][1:]
    scores = np.zeros(max_depth_query + 1)

    min_dist = np.min(cell_distances)
    max_dist = np.max(cell_distances)

    if weight:
        if max_dist == min_dist:
            normalized_distances = np.zeros_like(cell_distances)
        else:
            normalized_distances = 1 - (
                (cell_distances - min_dist) / (max_dist - min_dist)
            )
        gaussian_distances = 1 - np.exp(-(normalized_distances**2) / (2 * sd**2))
        weight_scores = gaussian_distances / np.sum(gaussian_distances)
    else:
        weight_scores = np.full(len(cell_neighbors), 1 / len(cell_neighbors))

    for level in range(max_depth_query):
        predicted_label = labels_by_level_w_final_level[level][cell_idx]

        if predicted_label != "NA":
            neighbor_labels = np.array(
                [labels_by_level_w_final_level[level][idx] for idx in cell_neighbors]
            )
            scores[level] = np.sum((neighbor_labels == predicted_label) * weight_scores)
        else:
            scores[level] = np.nan

    last_valid_level = labels_by_level_w_final_level[max_depth][cell_idx]
    scores[max_depth_query] = scores[last_valid_level - 1]

    return scores


def create_umaps(
    em_dict,
    n_neighbors,
    n_components,
    metric,
    min_dist,
    umap_lr,
    umap_seed,
    spread,
    verbose,
    init,
):
    """
    Compute UMAP embeddings for a set of embeddings using specified
    parameters.

    This function takes a dictionary of embeddings and computes UMAP
    projections for each embedding using the umap-learn package. The
    UMAP parameters provided are typically aligned with the defaults in
    Seurat (e.g., n_neighbors=30, n_components=2, min_dist=0.3,
    metric='cosine', learning_rate=1.0, random_state=42, spread=1,
    verbose=True, and init='spectral'), but can be customized as needed.

    For each embedding in `em_dict`, if the embedding is a numpy array,
    the function prints diagnostic information about its sparsity and
    then computes the UMAP embedding. Embeddings that are not numpy
    arrays are skipped with a printed message. If `em_dict` is empty, a
    warning is issued and the function returns None.

    Args:
        em_dict (dict): A dictionary where keys are embedding names
            (str) and values are embeddings as numpy.ndarray.
        n_neighbors (int): Number of nearest neighbors to use for UMAP.
        n_components (int): Dimension of the UMAP embedding.
        metric (str): Distance metric to use in UMAP (e.g., 'cosine').
        min_dist (float): Minimum distance parameter for UMAP.
        umap_lr (float): Learning rate for UMAP optimization.
        umap_seed (int): Random seed for UMAP to ensure reproducibility.
        spread (float): Spread parameter for UMAP.
        verbose (bool): Verbosity flag for UMAP.
        init (str): Initialization method for UMAP (e.g., 'spectral').

    Returns:
        dict or None: A dictionary where each key matches the
            corresponding key in `em_dict` and the value is the UMAP
            embedding (a numpy.ndarray). Returns None if `em_dict` is
            empty.
    """

    if not em_dict:
        warnings.warn("Can't find any embeddings to calculate umaps. Skipping.")
        return
    else:
        umap_embeddings_dict = {}

        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            learning_rate=umap_lr,
            random_state=umap_seed,
            spread=spread,
            verbose=verbose,
            init=init,
        )
        for em_name in em_dict:
            em = em_dict[em_name]
            if not isinstance(em, np.ndarray):
                print(
                    f"{em_name} must be in np.ndarray format, but they aren't, "
                    "so skipping the umap calculation."
                )
            else:
                sparsity = 1 - (np.count_nonzero(em) / em.size)
                print(f"Sparsity on embeddings is {sparsity}.\n")

                umap_embeddings = umap_model.fit_transform(em)
                umap_embeddings_dict[em_name] = umap_embeddings

        return umap_embeddings_dict


def create_outdir(
    query_filepath, model, epochs, split_mode, batch_size, loss_name, feature_names_col
):
    """
    Creates output directory to save results obtained from inference
    at the same path as the query object.
    """

    query_filename = query_filepath.split("/")[-1].split(".")[:-1]
    query_filename = ".".join(query_filename)

    if feature_names_col:
        out_subdir = (
            f"ANNotations_{query_filename}_{model}_"
            f"{epochs}epochs_{split_mode}_bs{batch_size}"
            f"_fn_{feature_names_col}"
        )
    else:
        out_subdir = (
            f"ANNotations_{query_filename}_{model}_"
            f"{epochs}epochs_{split_mode}_bs{batch_size}"
        )

    if loss_name != "focal_loss":
        out_subdir = f"{out_subdir}_{loss_name}"

    outpath = os.path.join(str(Path(query_filepath).parent), out_subdir)

    os.makedirs(outpath, exist_ok=True)

    return outpath


def save_outputs_to_disk(
    out_file_disk,
    object_disk,
    outpath,
    split_mode,
    model_import,
    softmax_outputs,
    embeddings_mode,
    embeddings_dict,
    labels_pred,
    combined_labels,
    new_frac,
    discordant_frac,
    new_unique,
    query_cells_df,
    if_umap_embeddings,
    umap_embeddings_dict,
    template_features,
    common_features,
    query_obj,
    feature_names_col,
    query_filepath,
):
    """
    Save various outputs from the annotation pipeline to disk.

    This function writes a variety of outputs to disk based on the
    provided flags. It can save model summaries, softmax outputs,
    embeddings, predicted labels, cell type labels, feature lists, cell
    metadata, UMAP embeddings, and the final annotated query object. The
    outputs are stored in the directory specified by `outpath` and are
    named using parameters such as the split mode and query file name.

    When `out_file_disk` is True, the following outputs are written:
        - A model summary text file.
        - A .npz file containing softmax outputs.
        - (If `embeddings_mode` is True) A pickle file containing
            embeddings.
        - A text file containing split labels.
        - A text file containing cell type labels.
        - A text file with statistics on discordant and new cell type
            fractions, plus a list of unique new cell type labels.
        - A features.txt file detailing the number of features used for
            training and the overlapping features in the query.
        - A CSV file of cell metadata (from `query_cells_df`).
        - (If `if_umap_embeddings` is True) A pickle file containing
            UMAP embeddings.

    When `object_disk` is True, the annotated query object (`query_obj`)
      is updated with the cell metadata and any embeddings (both
      annotation embeddings and UMAP embeddings) and saved to disk as an
        .h5ad file. The output filename is derived from the original
        query file name with an '_ANN.h5ad' suffix.

    Args:
        out_file_disk (bool): Flag indicating whether to write
            individual outputs to disk.
        object_disk (bool): Flag indicating whether to save the
            annotated query object.
        outpath (str): Directory path where all outputs will be written.
        split_mode (str): Identifier for the current split mode; used in
             naming output files.
        model_import (tf.keras.Model): The trained model whose summary
            is saved to disk.
        softmax_outputs (list): List of softmax output arrays from model
             predictions.
        embeddings_mode (bool): Flag indicating whether (and which)
            layer embeddings should be saved.
        embeddings_dict (dict): Dictionary of annotation embeddings to
            be saved (if applicable).
        labels_pred (np.ndarray): Array of predicted labels; saved to a
            text file.
        combined_labels (list of str): List of cell type labels to be
            written to disk.
        new_frac (float): Fraction of cells annotated as "new" cell
            types that is are outside of the hierarchy used in training.
        discordant_frac (float): Fraction of cells with discordant model
             predictions.
        new_unique (list of str): List of unique new cell type labels
            predicted.
        query_cells_df (pandas.DataFrame): DataFrame containing cell
            metadata.
        if_umap_embeddings (bool): Flag indicating whether UMAP
            embeddings should be saved.
        umap_embeddings_dict (dict): Dictionary of UMAP embeddings to
            be saved.
        template_features (list of str): List of features used during
            training.
        common_features (list of str): List of features overlapping
            between query and training.
        query_obj (anndata.AnnData): The query AnnData object to be
            annotated and saved.
        feature_names_col (str or None): Column in query_obj.var to set
            as index for features.
        query_filepath (str): File path of the original query file,
            used for naming output file.

    Returns:
        None

    """

    if out_file_disk:
        print("Writing some outputs to dir " + outpath + "... \n")

        model_summary_filename = f"{split_mode}_model_summary.txt"
        model_summary_filepath = os.path.join(outpath, model_summary_filename)
        with open(model_summary_filepath, "w") as f:
            model_import.summary(print_fn=lambda x: f.write(x + "\n"))

        softmax_outputs_filepath = os.path.join(outpath, "softmax_outputs.npz")
        np.savez(softmax_outputs_filepath, *softmax_outputs)

        if embeddings_mode:
            embeddings_file = "embeddings.pkl"
            embedding_outpath = os.path.join(outpath, embeddings_file)
            with open(embedding_outpath, "wb") as f:
                pickle.dump(embeddings_dict, f)

        array_outpath = os.path.join(outpath, f"{split_mode}_split_labels.txt")
        np.savetxt(array_outpath, labels_pred, delimiter=",\n", fmt="%s")

        labels_outpath = os.path.join(outpath, "cell_type_labels.txt")
        with open(labels_outpath, "w") as f:
            for label in combined_labels:
                f.write(label + "\n")

        new_labels_outpath = os.path.join(outpath, "new_cell_type_labels.txt")
        with open(new_labels_outpath, "w") as f:
            f.write(
                f"{discordant_frac} fraction of all cells had discordant "
                "predictions from the model.\n"
            )
            f.write(
                f"{new_frac} fraction of all cells are annotated as "
                "cell types outside of database.\n"
            )
            f.write("\n")
            f.write("The set of new cell type labels predicted is:\n")
            f.write("\n")
            for new_label in new_unique:
                f.write(new_label + "\n")
                f.write("\n")

        features_outpath = os.path.join(outpath, "features.txt")
        with open(features_outpath, "w") as f:
            f.write(f"# features used for training: {len(template_features)}" " \n")
            f.write(
                "# overlapping features in query object: " f"{len(common_features)} \n"
            )
            print("\n")
            f.write("The overlapping features are:\n")
            f.write("\n")
            for gene in common_features:
                f.write(gene + "\n")
            f.write("\n")
            f.write("\n")
            f.write("The set of all features used in training are: \n")
            f.write("\n")
            for gene in template_features:
                f.write(gene + "\n")

        query_cells_df.to_csv(os.path.join(outpath, "cell_dataframe.csv"))

        if if_umap_embeddings:
            umap_filename = "umaps.pkl"
            umap_filepath = os.path.join(outpath, umap_filename)
            with open(umap_filepath, "wb") as f:
                pickle.dump(umap_embeddings_dict, f)

    if object_disk:
        print(
            "Writing object with annotations and embeddings, if any, "
            f"to disk at {outpath} ... \n "
        )
        query_obj.obs = query_cells_df

        if feature_names_col:
            query_obj.var.set_index(feature_names_col, inplace=True)

        if embeddings_mode:
            for em_name in embeddings_dict:
                em = embeddings_dict[em_name]
                query_obj.obsm["ann_" + em_name] = em

        if if_umap_embeddings:
            for em_name in umap_embeddings_dict:
                em = umap_embeddings_dict[em_name]
                query_obj.obsm["umap_ann_" + em_name] = em

        query_name = query_filepath.split("/")[-1]
        query_name = query_name[:-5]
        query_ANN_name = query_name + "_ANN.h5ad"
        query_obj.write(os.path.join(outpath, query_ANN_name))


def insert_col(df, loc, col_name, col_vals):
    """
    Inserts a column in a pandas dataframe, if the column name
    exists already, it is overwritten.
    """

    if col_name in df.columns:
        df.drop(col_name, axis=1, inplace=True)
    df.insert(loc, col_name, col_vals)

    return df


def if_concordant(cell_label, max_depth):
    """
    Returns a boolean indicating whether the hierarchical
    predictions returned by the model for a given cell are
    internally concordant or not.
    """
    for i in range(max_depth - 1):
        res = cell_label[i].split("|")[-1] == (cell_label[i + 1].split("|")[-2])

        if res == False:
            break

    return res


def arg_parse_in():
    """
    Arg parser to collect arguments to pass to ANNotate.py
    """

    print("Parsing arguments... \n")
    print("\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        help=("enter abs file path to the query." " Query should be in h5ad format."),
        type=str,
    )

    parser.add_argument(
        "-m",
        "--mode",
        default="cumulative",
        help=("enter the label mode: cumulative" " or independent."),
        type=str,
    )

    parser.add_argument(
        "-md", "--model", default="M0.2", help="enter the model name.", type=str
    )

    parser.add_argument(
        "-sdd",
        "--source_data_dir",
        default=(
            "/brahms/sarkars/AzimuthNN_clone/" "AzimuthNN/sarkars/data/dataset_main"
        ),
        help=("enter the dir path where the dataset" " object is stored"),
        type=str,
    )

    parser.add_argument(
        "-ft",
        "--features_txt",
        default="features_02_26_25_17_50.txt",
        help=(
            "features txt file with feature selection, "
            "ensure model chosen is trained on "
            "the same set"
        ),
        type=str,
    )

    parser.add_argument(
        "-fn",
        "--feature_names_col",
        default=None,
        help=(
            "enter the column name where the "
            "feature names are stored in query.var"
            " where query is the anndata object read "
            "from the h5ad."
        ),
        type=str,
    )

    parser.add_argument(
        "--epochs",
        default=55,
        help=("enter the number of epochs the " "model has been trained for"),
        type=int,
    )

    parser.add_argument(
        "-ts",
        "--train_seed",
        default=100,
        help="enter the training seed used",
        type=int,
    )

    parser.add_argument(
        "-l",
        "--loss",
        default="level_wt_focal_loss",
        help=("enter the loss function used for " "optimization"),
        type=str,
    )
    parser.add_argument(
        "-ds",
        "--data_seed",
        default=414,
        help="enter the data prep seed that was used",
        type=int,
    )

    parser.add_argument(
        "-dso",
        "--data_source",
        default=("data/kfold_data/datasets/" "fold10_02_26_2025_17_53_139"),
        help=("enter the k-fold source dataset with " "no / at either end"),
        type=str,
    )

    parser.add_argument(
        "-dsp",
        "--data_split",
        nargs=3,
        default=[7, 1, 2],
        help=("enter the train:valid:test split as ints " "separated by a space"),
        type=int,
    )

    parser.add_argument(
        "-ms",
        "--mask_seed",
        default=None,
        help="enter the seed used in the masking processes",
        type=int,
    )

    parser.add_argument(
        "-tm",
        "--tail_mask",
        nargs=2,
        default=None,
        help=(
            "enter the tail masking parameters as floats "
            "separated by a space: [frac of cells masked, "
            "max frac of tail depth masked]"
        ),
        type=float,
    )

    parser.add_argument(
        "-slm",
        "--single_level_mask",
        default=None,
        help=(
            "enter the fraction of cells in which a " "single random level was masked"
        ),
        type=float,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        default=256,
        help="enter the batch size used in training",
        type=int,
    )

    parser.add_argument(
        "-ebs",
        "--eval_batch_size",
        default=40960,
        help=(
            "enter the evaluation batch size suitable to "
            "your hardware, defaults to 40960"
        ),
        type=int,
    )

    parser.add_argument(
        "-opt",
        "--optimizer",
        default="adam",
        help="enter the name of the optimizer used",
        type=str,
    )
    parser.add_argument(
        "-lr", "--lr", default=None, help="enter the learning rate used", type=float
    )

    parser.add_argument(
        "-l1",
        "--l1",
        default=None,
        help="enter L1 reg strength used if any",
        type=float,
    )
    parser.add_argument(
        "-l2",
        "--l2",
        default=None,
        help="enter L2 reg strength used if any",
        type=float,
    )
    parser.add_argument(
        "-dp",
        "--dropout",
        default=None,
        help="enter dropout rate used if any",
        type=float,
    )

    parser.add_argument(
        "-norm",
        "--normalization_override",
        default=False,
        help=(
            "is the counts data lop1p normalized after "
            "scaling to 10k? defaults to False"
        ),
        type=bool,
    )

    parser.add_argument(
        "-em",
        "--embeddings",
        default=None,
        help=(
            "extract embeddings? defaults to None, other "
            "options: ['shallow', 'deep', 'both']"
        ),
        type=str,
    )

    parser.add_argument(
        "-knn",
        "--knn_scores",
        default=False,
        help=(
            "specify if you want scores based on k "
            "nearest neighbours, defaults to False"
        ),
        type=bool,
    )

    parser.add_argument(
        "-umap",
        "--umap_embeddings",
        default=False,
        help=("specify if you want umap embeddings, " "defaults to False"),
        type=bool,
    )

    parser.add_argument(
        "-rf",
        "--refine_labels",
        default=True,
        help=("do you want to refine annotations? " "default is True"),
        type=bool,
    )

    parser.add_argument(
        "-nnbrs",
        "--n_neighbors",
        default=30,
        help=("n_neighbors param for umaps, defaults " "to Seurat default 30"),
        type=int,
    )

    parser.add_argument(
        "-nc",
        "--n_components",
        default=2,
        help=("n_components param for umaps, defaults " "to Seurat default 2"),
        type=int,
    )

    parser.add_argument(
        "-me",
        "--metric",
        default="cosine",
        help=("metric param for umaps, defaults to " "Seurat default 'cosine'"),
        type=str,
    )

    parser.add_argument(
        "-mdt",
        "--min_dist",
        default=0.3,
        help=("min_dist param for umaps, defaults to " "Seurat default 0.3"),
        type=float,
    )

    parser.add_argument(
        "-ulr",
        "--umap_lr",
        default=1.0,
        help=("learning_rate param for umaps, defaults " "to Seurat default 1.0"),
        type=float,
    )

    parser.add_argument(
        "-useed",
        "--umap_seed",
        default=42,
        help=(
            "random_state param for reproducibility of "
            "umaps, defaults to Seurat default 42"
        ),
        type=int,
    )

    parser.add_argument(
        "-sp",
        "--spread",
        default=1.0,
        help=("spread param for umaps, defaults to " "Seurat default 1.0"),
        type=float,
    )

    parser.add_argument(
        "-uv",
        "--umap_verbose",
        default=True,
        help=("verbose param for umaps, " "defaults to True"),
        type=bool,
    )

    parser.add_argument(
        "-uin",
        "--umap_init",
        default="spectral",
        help=(
            "init param for umaps, defaults to "
            "'spectral', the other option is 'random'"
        ),
        type=str,
    )

    parser.add_argument(
        "-objd",
        "--object_disk",
        default=True,
        help=("do you want to write object to disk? " "default is True"),
        type=bool,
    )

    parser.add_argument(
        "-ofd",
        "--out_file_disk",
        default=True,
        help=("do you want to write separate files to disk?" " default is True"),
        type=bool,
    )

    args = parser.parse_args()

    return args


def arg_parse_out(args):
    """
    Reading the arguments passed to the arg parser for ANNotate.py.
    """
    print("Reading arguments... \n")
    print("\n")

    query_filepath = args.filepath
    source_data_dir = args.source_data_dir
    features_txt = args.features_txt

    feature_names_col = args.feature_names_col
    if args.mode == "independent" or args.mode == "cumulative":
        split_mode = args.mode
    else:
        raise ValueError("Mode should either be 'independent' or 'cumulative'")
    model = args.model

    loss_name = args.loss
    epochs = args.epochs
    train_seed = args.train_seed

    data_seed = args.data_seed
    data_source = args.data_source
    data_split = args.data_split
    mask_seed = args.mask_seed
    tm_frac = args.tail_mask
    lm_frac = args.single_level_mask
    save = True
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    optimizer_name = args.optimizer
    lr = args.lr
    l1 = args.l1
    l2 = args.l2
    dropout = args.dropout
    normalization_override = args.normalization_override

    embeddings_mode = args.embeddings
    if_knn_scores = args.knn_scores
    if_umap_embeddings = args.umap_embeddings
    if_refine_labels = args.refine_labels
    n_neighbors = args.n_neighbors
    n_components = args.n_components
    metric = args.metric
    min_dist = args.min_dist
    umap_lr = args.umap_lr
    umap_seed = args.umap_seed
    spread = args.spread
    verbose = args.umap_verbose
    init = args.umap_init

    object_disk = args.object_disk
    out_file_disk = args.out_file_disk

    arguments = [
        query_filepath,
        source_data_dir,
        features_txt,
        feature_names_col,
        split_mode,
        model,
        loss_name,
        epochs,
        train_seed,
        data_seed,
        data_source,
        data_split,
        mask_seed,
        tm_frac,
        lm_frac,
        save,
        batch_size,
        eval_batch_size,
        optimizer_name,
        lr,
        l1,
        l2,
        dropout,
        normalization_override,
        embeddings_mode,
        if_knn_scores,
        if_umap_embeddings,
        if_refine_labels,
        n_neighbors,
        n_components,
        metric,
        min_dist,
        umap_lr,
        umap_seed,
        spread,
        verbose,
        init,
        object_disk,
        out_file_disk,
    ]

    return arguments


def annotate_core(
    X_query,
    query_features,
    source_data_dir,
    features_txt,
    split_mode,
    model,
    epochs,
    train_seed,
    loss_name,
    data_seed,
    data_source,
    data_split,
    mask_seed,
    tm_frac,
    lm_frac,
    batch_size,
    optimizer_name,
    lr,
    l1,
    l2,
    dropout,
    save,
    eval_batch_size,
    normalization_override,
    embeddings_mode,
    query_cells_df,
    if_knn_scores,
    if_umap_embeddings,
    if_refine_labels,
    n_neighbors,
    n_components,
    metric,
    min_dist,
    umap_lr,
    umap_seed,
    spread,
    verbose,
    init,
):
    configure()

    print("Reference model and parameters:")
    print(f"    Model name: {model}")
    print(f"    Evaluation batch size: {eval_batch_size}")
    print(f"    Extract embeddings: {embeddings_mode}")
    print(f"    Run umap: {if_umap_embeddings}")
    print(f"    Refine labels in postprocessing: {if_refine_labels}")
    print(f"    Compute knn scores: {if_knn_scores}\n")

    X_query, check_norm = normalize(X_query, normalization_override)

    (X_query, num_cells, template_features, common_features) = extract_X(
        X_query, query_features, source_data_dir, features_txt
    )

    print("Normalization:")
    print(f"    Integer expression values detected: {not check_norm}")
    if check_norm:
        print("    Assuming data has been log-normalized.\n")
    else:
        print("    Performing log-normalization.\n")

    ###############################################################################

    depth_aug = model[-5:] == "depth"

    exp_dir, _, _, _, _, _ = give_exp_dir(
        split_mode,
        model,
        epochs,
        train_seed,
        loss_name,
        data_seed,
        data_source,
        data_split,
        mask_seed,
        tm_frac,
        lm_frac,
        batch_size,
        optimizer_name,
        lr,
        l1,
        l2,
        dropout,
        save,
        exist=True,
    )

    model_path = os.path.join(exp_dir, "saved_model.keras")

    model_import = load_model(model_path)

    enc_path = os.path.join(exp_dir, "encoders.pkl")
    with open(enc_path, "rb") as f:
        label_encoders = pickle.load(f)

    ##################################################################################

    if depth_aug:
        max_depth_given = "autocorrect_for_depth"
    else:
        max_depth_given = None

    (labels_pred, _, labels_prob, max_depth, softmax_outputs) = labels_array(
        X_query,
        model_import,
        label_encoders,
        eval_batch_size,
        max_depth_given=max_depth_given,
    )

    cell_types_all = load_cell_types()

    combined_labels = []
    new_labels = []
    concordant = []
    for cell_label in labels_pred:
        concordant.append(if_concordant(cell_label, max_depth))
        combined_label = comb_label(cell_label, max_depth, max_depth)
        combined_labels.append(combined_label)
        if combined_label not in cell_types_all:
            new_labels.append(combined_label)

    discordant_frac = 1 - np.sum(np.array(concordant)) / len(concordant)

    new_frac = len(new_labels) / len(combined_labels)
    new_unique = list(set(new_labels))

    ################################################################################

    embeddings_dict = {}

    if embeddings_mode:
        print("")
        print("Extracting model embeddings:")
        embedding_layers = give_embedding_layers(embeddings_mode, model)

        for em_layer in embedding_layers:
            embeddings_name = em_layer[:-6] + "s"
            embedding_outputs = embeddings(
                X_query, model_import, embedding_layers[em_layer], eval_batch_size
            )
            embedding_outputs = embedding_outputs.numpy()
            embeddings_dict[embeddings_name] = embedding_outputs

    #################################################################################

    (
        abs_split_labels_list_w_final_layer,
        final_levels_arr,
    ) = split_labels_w_final_level(labels_pred, max_depth)
    final_level_prob = labels_prob[np.arange(num_cells), final_levels_arr]

    assert str(type(query_cells_df)) == (
        "<class 'pandas.core.frame.DataFrame'>"
    ), "Query cell meta is not available as dataframe."

    query_cells_df = insert_col(
        query_cells_df, 0, "abs_cell_type_label", combined_labels
    )
    query_cells_df = insert_col(
        query_cells_df,
        1,
        "cell_type_in_database",
        [label in cell_types_all for label in combined_labels],
    )
    query_cells_df = insert_col(
        query_cells_df,
        2,
        "abs_cell_type_label_with_flag",
        [f"{label}_{label in cell_types_all}" for label in combined_labels],
    )
    for i in range(max_depth):
        query_cells_df = insert_col(
            query_cells_df,
            3 + i,
            f"cell_type_label_level_{i+1}",
            abs_split_labels_list_w_final_layer[i],
        )
    for i in range(max_depth):
        query_cells_df = insert_col(
            query_cells_df,
            3 + max_depth + i,
            f"prob_level_{i+1}",
            list(labels_prob[:, i]),
        )
    query_cells_df = insert_col(
        query_cells_df,
        3 + 2 * max_depth,
        "final_nonblank_level",
        abs_split_labels_list_w_final_layer[max_depth],
    )
    query_cells_df = insert_col(
        query_cells_df,
        4 + 2 * max_depth,
        "final_level_label",
        abs_split_labels_list_w_final_layer[max_depth + 1],
    )
    query_cells_df = insert_col(
        query_cells_df, 5 + 2 * max_depth, "final_level_prob", list(final_level_prob)
    )
    query_cells_df = insert_col(
        query_cells_df, 6 + 2 * max_depth, "concordance", concordant
    )

    #########################################################################################
    ######## postprocessing: refining annotate labels #######################################

    if if_refine_labels:
        ann_level = "fine"
        print("")
        print(f"Finalizing label predictions for consistent granularity.\n")
        enc_map_path = os.path.join(exp_dir, "encoder_maps.json")
        with open(enc_map_path, "r") as file:
            encoder_map = json.load(file)
        fine_annot = pd.read_csv(
            "/brahms/sarkars/Panhuman_AzimuthNN_public/data/"
            f"postprocessing_files/panhuman_annotate_{ann_level}.csv",
            index_col=None,
        )
        refined_results = refine_annotations(
            encoder_map, fine_annot, softmax_outputs, query_cells_df, ann_level
        )
        query_cells_df = insert_col(
            query_cells_df,
            7 + 2 * max_depth,
            f"annotate_{ann_level}",
            refined_results[f"annotate_{ann_level}"],
        )
        query_cells_df = insert_col(
            query_cells_df,
            8 + 2 * max_depth,
            f"annotate_{ann_level}_prob",
            refined_results[f"annotate_{ann_level}_prob"],
        )
        query_cells_df = insert_col(
            query_cells_df,
            9 + 2 * max_depth,
            f"refine_type",
            refined_results["result_type"],
        )

    #####################################################################################
    # knn based scores

    if if_knn_scores:
        print("Computing knn scores.\n")
        if embeddings_mode is None:
            print(
                "No embeddings have been generated to calculate ", "knn based scores.\n"
            )
        else:
            neighbors = 10

            max_depth_query = max(abs_split_labels_list_w_final_layer[max_depth])

            em_layer_counter = 0
            for em_layer in embedding_layers:
                embeddings_name = em_layer[:-6] + "s"
                if embeddings_name not in embeddings_dict:
                    raise KeyError(
                        f"Key {embeddings_name} not found in the "
                        "embeddings dictionary."
                    )
                embeddings_vals = embeddings_dict[embeddings_name]

                nbrs = NearestNeighbors(
                    n_neighbors=neighbors + 1, algorithm="auto"
                ).fit(embeddings_vals)
                distances, indices = nbrs.kneighbors(embeddings_vals)

                result = [
                    compute_scores(
                        i,
                        indices,
                        distances,
                        max_depth_query,
                        max_depth,
                        abs_split_labels_list_w_final_layer,
                    )
                    for i in range(num_cells)
                ]
                result = np.array(result)

                assert (
                    result.shape[0] == num_cells
                ), f"{embeddings_name}: dimension mismatch."
                assert (
                    result.shape[1] == max_depth_query + 1
                ), f"{embeddings_name}: dimension mismatch."

                columns = 10 if if_refine_labels else 7
                base_index = (
                    columns + 2 * max_depth + (em_layer_counter * (max_depth_query + 2))
                )
                for i in range(max_depth_query):
                    query_cells_df = insert_col(
                        query_cells_df,
                        base_index + i,
                        embeddings_name + f"_level_{i+1}_knn_score",
                        list(result[:, i]),
                    )
                query_cells_df = insert_col(
                    query_cells_df,
                    base_index + max_depth_query,
                    embeddings_name + "_final_level_knn_score",
                    list(result[:, max_depth_query]),
                )
                query_cells_df = insert_col(
                    query_cells_df,
                    base_index + max_depth_query + 1,
                    embeddings_name + "_diff_prob_knn",
                    list(final_level_prob - result[:, max_depth_query]),
                )

                em_layer_counter += 1

    ##############################################################################################################
    ### calculating umaps ###################################################################################

    if if_umap_embeddings:
        print("Running UMAP:")
        embedding_details = {
            "shallow": "shallow_embeddings, 128 dimensions",
            "deep": "deep_embeddings, 256 dimensions",
            "both": (
                "shallow and deep embeddings," " 128 and 256 dimensions respectively"
            ),
        }
        print(f"    UMAP input: {embedding_details[embeddings_mode]}, ")
        print("")

        umap_embeddings_dict = create_umaps(
            embeddings_dict,
            n_neighbors,
            n_components,
            metric,
            min_dist,
            umap_lr,
            umap_seed,
            spread,
            verbose,
            init,
        )
    else:
        umap_embeddings_dict = {}

    core_outputs = [
        model_import,
        softmax_outputs,
        embeddings_mode,
        embeddings_dict,
        labels_pred,
        combined_labels,
        new_frac,
        discordant_frac,
        new_unique,
        query_cells_df,
        if_umap_embeddings,
        umap_embeddings_dict,
        template_features,
        common_features,
    ]

    return core_outputs


def annotate():
    ######### parsing arguments
    args = arg_parse_in()

    #### Reading arg parse arguments

    arguments = arg_parse_out(args)

    query_filepath = arguments[0]
    source_data_dir = arguments[1]
    features_txt = arguments[2]
    feature_names_col = arguments[3]
    split_mode = arguments[4]
    model = arguments[5]
    loss_name = arguments[6]
    epochs = arguments[7]
    train_seed = arguments[8]
    data_seed = arguments[9]
    data_source = arguments[10]
    data_split = arguments[11]
    mask_seed = arguments[12]
    tm_frac = arguments[13]
    lm_frac = arguments[14]
    save = arguments[15]
    batch_size = arguments[16]
    eval_batch_size = arguments[17]
    optimizer_name = arguments[18]
    lr = arguments[19]
    l1 = arguments[20]
    l2 = arguments[21]
    dropout = arguments[22]
    normalization_override = arguments[23]
    embeddings_mode = arguments[24]
    if_knn_scores = arguments[25]
    if_umap_embeddings = arguments[26]
    if_refine_labels = arguments[27]
    n_neighbors = arguments[28]
    n_components = arguments[29]
    metric = arguments[30]
    min_dist = arguments[31]
    umap_lr = arguments[32]
    umap_seed = arguments[33]
    spread = arguments[34]
    verbose = arguments[35]
    init = arguments[36]
    object_disk = arguments[37]
    out_file_disk = arguments[38]

    ########################################################
    ###### creating output directory

    outpath = create_outdir(
        query_filepath,
        model,
        epochs,
        split_mode,
        batch_size,
        loss_name,
        feature_names_col,
    )

    ##########################################################################

    X_query, query_features, query_cells_df, query_obj = read_obj(
        query_filepath, feature_names_col=feature_names_col
    )

    #################################################################################
    ####### annotate python core

    core_outputs = annotate_core(
        X_query,
        query_features,
        source_data_dir,
        features_txt,
        split_mode,
        model,
        epochs,
        train_seed,
        loss_name,
        data_seed,
        data_source,
        data_split,
        mask_seed,
        tm_frac,
        lm_frac,
        batch_size,
        optimizer_name,
        lr,
        l1,
        l2,
        dropout,
        save,
        eval_batch_size,
        normalization_override,
        embeddings_mode,
        query_cells_df,
        if_knn_scores,
        if_umap_embeddings,
        if_refine_labels,
        n_neighbors,
        n_components,
        metric,
        min_dist,
        umap_lr,
        umap_seed,
        spread,
        verbose,
        init,
    )

    ###### saving outputs

    save_outputs_to_disk(
        out_file_disk,
        object_disk,
        outpath,
        split_mode,
        *core_outputs,
        query_obj,
        feature_names_col,
        query_filepath,
    )


if __name__ == "__main__":
    annotate()
