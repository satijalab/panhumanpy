""" 
Module with inference tools using Azimuth Neural Network trained on 
annotated panhuman scRNA-seq data.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import pickle
import anndata
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.neighbors import NearestNeighbors
import umap
import warnings
from datetime import datetime

import sys
import importlib
from importlib.resources import files
from panhumanpy._tools import inference_model, inference_encoders
from panhumanpy._tools.inference_model import model_meta
from panhumanpy._tools import inference_feature_panel
from panhumanpy._tools import postprocessing
from panhumanpy.loss_fn import * 

import warnings
import gc
#warnings.filterwarnings("ignore")  make this optional in script




#######################################################################
################## functions ##########################################
#######################################################################
#######################################################################


def configure():
    """
    Configures TensorFlow GPU settings to optimize memory usage and 
    performance.

    - Limits default memory allocation on GPU to prevent excessive 
      consumption
    - Enables memory growth, allowing TensorFlow to allocate GPU memory 
      as needed
    - Sets JIT compilation flag to True for potential performance 
      improvements
    
    If GPUs are available, prints the number of physical and logical GPUs
    detected.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(
                len(gpus), 
                "Physical GPUs,", 
                len(logical_gpus), 
                "Logical GPUs \n"
                )
        except RuntimeError as e:
            print(e)
            print("\n")

    tf.config.optimizer.set_jit(True)


def is_valid_anndata_obj(obj):
    """
    write
    """
    return (
        isinstance(obj, anndata.AnnData) and
        hasattr(obj, "X") and obj.X is not None and
        hasattr(obj, "obs") and isinstance(obj.obs, pd.DataFrame) and
        hasattr(obj, "var") and isinstance(obj.var, pd.DataFrame)
    )


def check_normalization(
        matrix, 
        normalization_override, 
        norm_check_batch_size
        ):
    '''
    write docstring
    '''
    if matrix.shape[0] > norm_check_batch_size:
        mat = matrix[:norm_check_batch_size,:]
    else:
        mat = matrix

    if normalization_override:
        return True
    else:
        mat = mat.toarray()
        mat_floor = np.floor(mat)

        if np.any((mat_floor-mat) != 0.):
            return True
        
        else:
            return False
        

def normalize(
        matrix, 
        normalization_override, 
        norm_check_batch_size
        ):
    """
    Normalizes a gene expression count matrix using log1p transformation.

    This function first checks whether the provided matrix is already
    normalized by inspecting a batch of cells (specified by
    norm_check_batch_size). If the matrix is not normalized and 
    normalization_override is False, the function scales each cell to 
    10,000 total counts and applies a log1p transformation to the data.
    The input matrix is assumed to be a sparse matrix (e.g., CSR format)
    that supports the operations .sum(), .multiply(), and .tocsr().

    A caveat of the current formulation of this function is that it
    merely checks if the a certain batch of cells (as specified) has 
    integer counts data or not, and if not it assumes that the data is 
    log1p normalized, which need not be the case.

    Args:
        matrix (scipy.sparse matrix): Gene expression count matrix with 
            shape (num_cells, num_genes).
        normalization_override (bool): If True, assumes that the matrix 
            is already normalized and bypasses the scaling and log1p 
            transformation.
        norm_check_batch_size (int, optional): The number of cells used 
            to determine whether the matrix is normalized.

    Returns:
        scipy.sparse.csr_matrix: The normalized gene expression matrix.
        boolen: Whether the provided matrix was normalized or not 
            intially.
    """

    check_norm = check_normalization(
        matrix, 
        normalization_override, 
        norm_check_batch_size
        )

    if not check_norm:
        total_counts = matrix.sum(axis=1)  
        total_counts = np.array(total_counts).reshape(-1, 1)

        if 0 in total_counts.flatten():
            warnings.warn(
                "Cells with 0 counts across the entire feature "
                "panel found."
                )
        total_counts[total_counts == 0] = 1
        
        scaled_matrix = matrix.multiply(10000 / total_counts)
        scaled_matrix.data = np.log1p(scaled_matrix.data)
        matrix = scaled_matrix.tocsr()

        
    return matrix, check_norm


def reorder_subset_data_matrix(
        data_matrix,
        query_features,
        feature_panel_template,
        common_features=None
        ):
    """
    write
    """

    if not common_features:
        common_features = set(
            query_features
            ).intersection(set(feature_panel_template))
        
    extra_features = set(feature_panel_template)-common_features
        
    zero_columns = csr_matrix((data_matrix.shape[0], len(extra_features)))
    data_matrix = hstack([data_matrix, zero_columns])
    query_features_extended = query_features.copy()
    query_features_extended.extend(extra_features)

    reordered_query_indices = [
        query_features_extended.index(name) 
        for name in feature_panel_template
        ]
    reordered_data_matrix = data_matrix[:,reordered_query_indices]

    return reordered_data_matrix



def if_full_consistent_hierarchy(cell_label, max_depth):
    '''
    Returns a boolean indicating whether the hierarchical 
    predictions returned by the model for a given cell form a 
    consistent hierarchy or not.
    '''
    for i in range(max_depth-1):
        res = (
            cell_label[i].split("|")[-1] == (
                cell_label[i+1].split("|")[-2]
            )
        )

        if res==False:
            break
    
    return res



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
    out = ''
    for i in range(depth): 
        buffer=max_depth-i
        for j in range(buffer):
            add = array_label[i+j].split('|')[i]
            if add != '':
                out+= add
                out +='|'
                break
    out = out[:-1]

    return out



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

    abs_labels_upto_level=[[] for i in range(max_depth)]
    for i in range(max_depth):
        for cell_label in hierarchical_labels_array:
            combined_label = comb_label(cell_label, i+1, max_depth)
            if len(combined_label.split('|'))<i+1:
                combined_label = 'NA'
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
            present in the predicted labels.
    
    Returns:
        tuple: A tuple containing:
            - abs_labels_list (list): A list where:
                * The first max_depth number of elements are lists of 
                    the final label components for each level (i.e., the
                      part after the last '|' delimiter in the abs label
                      up to the corresponding level).
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

    na_mask = abs_labels_array!='NA'
    final_levels_arr = np.sum(na_mask, axis=1) -1 
    final_levels_list = list(final_levels_arr + 1)

    abs_labels_list.append(final_levels_list)

    for i in range(max_depth):
        abs_labels_list[i] = [
            label.split('|')[-1] for label in abs_labels_list[i]
            ]

    final_level_labels = np.array(abs_labels_list).T[
        np.arange(len(final_levels_list)), 
        final_levels_arr
        ]
    final_level_labels = list(final_level_labels)

    abs_labels_list.append(final_level_labels)

    return abs_labels_list, final_levels_arr


def categorize_refinement_type(value):
        if isinstance(value, list):
            return "Further"
        elif value == "False":
            return "No-match"
        elif isinstance(value, str):
            return "Match"
        return "Unknown"


def create_anndata(
    X, 
    cell_meta_df, 
    feature_df, 
    feature_names=None, 
    cell_ids=None,
    embeddings=None
    ):
    """
    Create an AnnData object from expression matrix and metadata.
    
    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Expression matrix of shape (n_cells, n_features)
    cell_meta_df : pandas.DataFrame
        DataFrame containing cell metadata, with rows corresponding to 
        cells
    feature_df : pandas.DataFrame
        DataFrame containing feature metadata, with rows corresponding 
        to features
    feature_names : list or None, optional
        Feature names to use as var_names. If None, uses 
        feature_df.index
    cell_ids : list or None, optional
        Cell IDs to use as obs_names. If None, uses cell_meta_df.index
    embeddings : dict or None, optional
        Dictionary of embeddings to store in obsm. Keys are embedding 
        names (e.g., 'X_umap', 'X_pca'), and values are numpy arrays of 
        shape (n_cells, n_embedding_dims)
    
    Returns
    -------
    anndata.AnnData
        AnnData object containing the expression matrix and metadata
    
    Raises
    ------
    ValueError
        If dimensions don't match between inputs
    """
    
    n_cells, n_features = X.shape
    
    if cell_meta_df.shape[0] != n_cells:
        raise ValueError(
            f"cell_meta_df has {cell_meta_df.shape[0]} rows but X "
            f"has {n_cells} rows (cells)"
            )
    
    if feature_df.shape[0] != n_features:
        raise ValueError(
            f"feature_df has {feature_df.shape[0]} rows but X has "
            f"{n_features} columns (features)"
            )
    
    if cell_ids is not None:
        cell_meta_df.index = cell_ids
    
    if feature_names is not None:
        feature_df.index = feature_names
    
    adata = anndata.AnnData(
        X=X,
        obs=cell_meta_df,
        var=feature_df
    )
    
    if embeddings is not None:
        if not isinstance(embeddings, dict):
            raise ValueError(
                "embeddings must be a dictionary mapping names to arrays"
                )
        
        for embedding_name, embedding_matrix in embeddings.items():
            if embedding_matrix.shape[0] != n_cells:
                raise ValueError(
                    f"Embedding '{embedding_name}' has "
                    f"{embedding_matrix.shape[0]} rows but should have "
                    f"{n_cells} rows (cells)."
                )
            adata.obsm[embedding_name] = embedding_matrix
    
    assert adata.n_obs == n_cells, (
        "Created AnnData object has incorrect number of observations"
    )
    assert adata.n_vars == n_features, (
        "Created AnnData object has incorrect number of variables"
    )
    
    return adata




def insert_col(df, loc, col_name, col_vals):
    '''
    Inserts a column in a pandas dataframe, if the column name 
    exists already, it is overwritten.
    '''

    if col_name in df.columns:
        df.drop(col_name, axis=1, inplace=True)
    df.insert(loc, col_name, col_vals)

    return df
        
    




        








########################################################################
################### class objects ######################################
########################################################################
########################################################################



class MemoryContext():
    def __init__(self, description="Memory-intensive operation"):
        self.description = description
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect() 
        return None



class Inference():
    """
    write: supervised inference run
    we want a confidence calibrated,
    supervised model running here
    """
    def __init__(
            self, 
            X, 
            model, 
            label_encoders, 
            eval_batch_size, 
            max_depth
            ):
        self._X = X
        self._model = model
        self._label_encoders = label_encoders
        self._eval_batch_size = eval_batch_size
        self._eval_steps = X.shape[0]//self._eval_batch_size
        self._max_depth = max_depth

    def run_on_minibatch(self, minibatch):
        """
        write
        """
        y_mb = self._model.predict(minibatch)

        return y_mb
    
    def process_minibatch(
            self, 
            minibatch, 
            softmax_mb_levels_cache,
            class_preds_mb_cache,
            max_probs_mb_cache
            ):
        """
        write
        """
        class_levels_cache = []
        prob_levels_cache = []

        y_mb = self.run_on_minibatch(minibatch)

        for i in range(self._max_depth):
            softmax_mb_levels_cache[i].append(y_mb[i])

            class_outs = tf.expand_dims(
                tf.cast(tf.argmax(y_mb[i], axis=-1), tf.int32), 
                axis=1
                )
            class_levels_cache.append(class_outs)

            max_probs = tf.reduce_max(y_mb[i], axis=-1, keepdims=True)
            prob_levels_cache.append(max_probs)

        class_preds_mb = tf.concat(class_levels_cache, axis=1)
        max_probs_mb = tf.concat(prob_levels_cache, axis=1)

        class_preds_mb_cache.append(class_preds_mb)
        max_probs_mb_cache.append(max_probs_mb)

        return (
            softmax_mb_levels_cache, 
            class_preds_mb_cache, 
            max_probs_mb_cache
        )
    
    def run_on_X(self):
        """
        Run inference on the entire input matrix in batches.
        """
        if self._X.shape[0] == 0:
            raise ValueError("Input matrix is empty")
        
        softmax_mb_levels_cache = [[] for _ in range(self._max_depth)]
        class_preds_mb_cache = []
        max_probs_mb_cache = []

        print(
            f"Splitting query data into {self._eval_steps+1} "
            "evaluation batches.\n"
            )
        
        print("Running model:")

        for i in range(self._eval_steps):
            start_idx = i*self._eval_batch_size
            end_idx = (i+1)*self._eval_batch_size
            if start_idx >= self._X.shape[0]:
                break
            
            minibatch = self._X[start_idx:end_idx].toarray()
            if minibatch.size == 0:
                continue
            
            (
                softmax_mb_levels_cache,
                class_preds_mb_cache,
                max_probs_mb_cache
            ) = self.process_minibatch(
                minibatch,
                softmax_mb_levels_cache,
                class_preds_mb_cache,
                max_probs_mb_cache
            )
            del minibatch

        final_start = self._eval_steps*self._eval_batch_size
        if final_start < self._X.shape[0]:
            final_batch = self._X[final_start:].toarray()
            if final_batch.size > 0:
                (
                    softmax_mb_levels_cache,
                    class_preds_mb_cache,
                    max_probs_mb_cache
                ) = self.process_minibatch(
                    final_batch,
                    softmax_mb_levels_cache,
                    class_preds_mb_cache,
                    max_probs_mb_cache
                )
                del final_batch

        if not class_preds_mb_cache:
            raise RuntimeError("No predictions were generated")

        softmax_vals_all = [
            np.concatenate(level, axis=0) for level in softmax_mb_levels_cache
        ]
        class_preds = tf.concat(class_preds_mb_cache, axis=0)
        max_probs = tf.concat(max_probs_mb_cache, axis=0)

        assert self._max_depth==class_preds.shape[1], (
            "The array of all predictions should have dimensions consistent "
            "with max_depth."
        )
        assert self._X.shape[0] == class_preds.shape[0], (
            "The number of predictions != the number of cells in X_query"
        )
        for level in softmax_vals_all:
            assert level.shape[0]==self._X.shape[0], (
                "Number of cells in softmax_outputs != numbers of cells in query "
            )

        return (
            np.array(class_preds),
            np.array(max_probs),
            softmax_vals_all
        )
    
    def run_inference(self):
        """
        write
        """
        string_labels_out = []

        with MemoryContext():
            class_preds, max_probs, softmax_vals_all = self.run_on_X()
        for i in range(self._max_depth):
            string_labels_out.append(
            self._label_encoders[i].inverse_transform(class_preds[:,i])
            )

        return (
            {
            'hierarchical_label_preds':np.array(string_labels_out).T, 
            'class_preds':class_preds, 
            'probability_of_preds':max_probs, 
            'softmax_vals_all':softmax_vals_all
            }
        )
    

class InferenceTools():
    """
    write: reading the tools needed for inference
    each tool in it's own directory identified by name
    each directory should have unique contents
    write a test based on this(?)
    """

    def __init__(
            self, 
            annotation_pipeline,
            inference_model_filename='inference_model.keras',
            model_meta= model_meta, 
            inference_encoders_filename='inference_encoders.pkl',
            inference_feature_panel_filename='inference_feature_panel.txt'
            ):
        
        self._inference_model_filename = inference_model_filename
        self._model_meta = model_meta
        self._inference_encoders_filename = inference_encoders_filename
        self._inference_feature_panel_filename = (
            inference_feature_panel_filename
        )

        self._annotation_pipeline = annotation_pipeline

    def load_inference_model(self):
        """
        to load the model,
        inference model must be save in inference_model dir
        """
        model_dir_path = files(inference_model)
        model_path = model_dir_path / self._inference_model_filename

        model= load_model(model_path)

        return model
    
    def load_model_meta(self):
        """
        read the meta corresponding to the model,
        max_depth, depth_aug, etc
        """

        meta_dict = self._model_meta      

        for meta_key in meta_dict.keys():
            assert isinstance(meta_key, str), (
                "keys in the model metadata dict must be provided as strings."
            )

        assert 'inference_model_name' in meta_dict.keys(),(
            "model metadata dict must have a "
            "key 'inference_model_name'"
        )

        assert 'inference_model_loss_function' in meta_dict.keys(),(
            "model metadata dict must have a key "
            "'inference_model_loss_function'"
        )

        assert 'max_depth' in meta_dict.keys(), (
            "model metadata dict must have a key 'max_depth'."
        )

        assert isinstance(meta_dict['max_depth'], int),(
            "max_depth must be an integer."
        )

        assert 'inference_model_embedding_layer' in meta_dict.keys(), (
            "model metadata dict must have a key "
            "'inference_model_embedding_layer'"
        )

        assert isinstance(meta_dict['inference_model_embedding_layer'], str),(
            "inference_model_embedding_layer name must be a string."
        )

        if self._annotation_pipeline == 'supervised':

            assert 'feature_panel_size' in meta_dict.keys(), (
                "model metadata dict must have a key 'feature_panel_size'."
            )

            assert isinstance(meta_dict['feature_panel_size'], int),(
                "feature_panel_size must be an integer."
            )
        # enforce existence of certain key:vals here

        return meta_dict

    def load_inference_encoders(self):
        """
        encoders for the inference model
        """ 
        encoders_dir_path = files(inference_encoders)
        encoders_path = encoders_dir_path / self._inference_encoders_filename

        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)

        return encoders
    
    def load_inference_feature_panel(self):
        """
        feature panel for subsetting the query for the inference model
        """
        if self._annotation_pipeline == 'supervised':
            feat_dir_path = files(inference_feature_panel)
            feat_path = feat_dir_path / self._inference_feature_panel_filename

            with open(feat_path, "r") as f:
                feat_list = f.read().splitlines()

            feat_list = [
                gene.decode("utf-8") if isinstance(gene, bytes) else gene 
                for gene in feat_list
                ]
            
            assert len(feat_list) == len(set(feat_list)), (
                "There are possible duplicates in the inference feature panel."
            )
            
            if 'feature_panel_size' in self.load_model_meta().keys():
                assert len(feat_list) == (
                    self.load_model_meta()['feature_panel_size']
                ), (
                    "Number of features in inference feature panel provided \n"
                    " does not match with the number provided in inference \n"
                    "model metadata."
                )
                
            return feat_list
        else:
            return None


class AutoloadInferenceTools(InferenceTools):
    """
    Automatically loads all inference tools on initialization
    and stores them as attributes based on method names.
    """
    
    def __init__(self, annotation_pipeline, *args, **kwargs):
        super().__init__(annotation_pipeline, *args, **kwargs)
        
        methods = [method for method in dir(self) 
                  if callable(getattr(self, method)) and 
                  (method.startswith('load_'))]
        
        for method_name in methods:
            attr_name = method_name[5:]
            # everything other than 'load_'
            method = getattr(self, method_name)
            result = method()
            setattr(self, attr_name, result)
    

  

class QueryObj():
    """
    write: query should be an anndata object
    """
    def __init__(self, query):
        if not is_valid_anndata_obj(query):
            raise ValueError(
                "Object should be a valid anndata.AnnData object w. \n"
                "a data slot X, cell meta and gene meta in obs and var resp."
            )
        
        self.query = query

    def X_query(self, format='csr_matrix'):
        """
        convert type to csr_matrix
        """

        X_query = self.query.X
        if format=='csr_matrix':
            if not isinstance(X_query, csr_matrix):
                X_query = csr_matrix(X_query)
        else:
            raise ValueError(
                "Inference process is supported on the csr_matrix only. \n"
                "We may incorporate more flexibility in the future.")
        
        return X_query
    
    def query_features(self, feature_names_col=None):
        """
        write
        """
        if feature_names_col:
            features = self.query.var[feature_names_col].tolist()
        else:
            features = self.query.var_names.tolist()

        return features

    def features_meta(self):
        """
        write
        """

        features_meta_df = self.query.var
        assert isinstance(features_meta_df, pd.DataFrame), (
            "Query features meta should be a pandas DataFrame."
            )
        
        return features_meta_df
    

    def cells_meta(self):
        """
        write
        """

        cells_meta_df = self.query.obs
        assert isinstance(cells_meta_df, pd.DataFrame), (
            "Query cells meta should be a pandas DataFrame."
            )
        
        return cells_meta_df
        

class ReadQueryObj(QueryObj):
    """
    write: to read query from file directly
    """

    def __init__(self, query_path):
        query_filetype = query_path.split('.')[-1]
        if query_filetype!='h5ad':
            raise ValueError("Query filetype should be h5ad.")        

        self.query_path = query_path
        query = anndata.read_h5ad(self.query_path)

        super().__init__(query)

        

class InferenceInputData():
    """
    class for processing a counts data matrix to input for inference model
    """
    def __init__(
            self, 
            X_query, 
            query_features, 
            feature_panel_template,
            normalization_override = False,
            norm_check_batch_size = 1000
            ):
        
        assert isinstance(X_query, csr_matrix), (
            "X_query should be in the scipy.sparse.csr_matrix format."
        )
        self.X_query, _ = normalize(
            X_query,
            normalization_override=normalization_override,
            norm_check_batch_size=norm_check_batch_size
            )
        self.num_cells = X_query.shape[0]

        assert isinstance(query_features, list), (
            "query_features must be a list."
        ) 
        assert all(isinstance(item, str) for item in query_features), (
            "Each element of the list of query_features must be a string."
        )
        self.query_features = query_features

        if feature_panel_template:
            assert isinstance(feature_panel_template, list), (
                "query_features must be a list."
            ) 
            assert all(
                isinstance(item, str) for item in feature_panel_template
                ), (
                "Each element of the list of query_features must be a string."
            )
        self.feature_panel_template = feature_panel_template
        if self.feature_panel_template:
            self.template_size = len(set(self.feature_panel_template))
            assert self.template_size == len(self.feature_panel_template), (
                "There are possible duplicates in the feature_panel_template."
            )

            self.common_features = set(
                self.query_features
                ).intersection(set(self.feature_panel_template))
            
            self.overlap_percent = (
                len(self.common_features)*100
            )/self.template_size

            if self.overlap_percent < 20:
                warnings.warn(
                    f"Low feature overlap ({self.overlap_percent:.1f}%) may "
                    "lead to suboptimal results.",
                    UserWarning
                )


    def reorder_subset_on_feature_template(self):
        """
        write
        """
        if self.feature_panel_template:
            reordered_X_query = reorder_subset_data_matrix(
                self.X_query,
                self.query_features,
                self.feature_panel_template,
                common_features=self.common_features
            )   

            assert reordered_X_query.shape[1]==self.template_size, (
                "Feature dimension mismatch after reordering "
                "and subsetting the query."
                )

            
            print("Query object:")
            print(f"    Total number of cells: {self.num_cells}")
            print("    Total number of features: "
                  f"{len(set(self.query_features))}")
            print("    Overlap w. feature reference: "
                  f"{len(self.common_features)} "
                  f"(~{int(self.overlap_percent)}%)\n")
            
            return reordered_X_query
        else:
            warnings.warn(
                "A reference feature panel has not been provided, returning "
                "query data on all query features."
            )
            return self.X_query
        
    def inference_input(self, annotation_pipeline='supervised'):
        """
        write: to produce input matrix for passing to inference model
        """

        if annotation_pipeline=='supervised':
            input_matrix = self.reorder_subset_on_feature_template()
        else:
            raise ValueError("annotation_pipeline not recognized.")
        
        return input_matrix
    


class OutputLabels():
    """
    class for processing inference output labels, annotate fine/medium
    not included here.
    """
    def __init__(self, labels_pred, labels_prob, max_depth, num_cells):

        assert labels_pred.shape[0] == num_cells, (
            "Number of labels predicted != number of cells in query."
        )

        self._labels_predicted = labels_pred
        self._labels_prob = labels_prob
        self._max_depth = max_depth
        self._num_cells = num_cells

        combined_labels = []
        for cell_label in self._labels_predicted:
            combined_label = comb_label(
                cell_label, 
                self._max_depth, 
                self._max_depth
                )
            combined_labels.append(combined_label)
        self.combined_labels = combined_labels

        (
            self._level_specific_labels_and_final_level,
            self._final_levels_array
        ) = split_labels_w_final_level(
            self._labels_predicted, 
            self._max_depth
        )

        self.level_zero_labels = self._level_specific_labels_and_final_level[0]
        self.final_level_labels = self._level_specific_labels_and_final_level[
            self._max_depth+1
        ]

        self.level_zero_softmax_prob = self._labels_prob[:,0]
        self.final_level_softmax_prob = self._labels_prob[
            np.arange(self._num_cells), 
            self._final_levels_array
            ]

        full_consistent_hierarchy = []
        for cell_label in self._labels_predicted:
            full_consistent_hierarchy.append(
                if_full_consistent_hierarchy(cell_label, self._max_depth)
                )
        self.full_consistent_hierarchy = full_consistent_hierarchy

       
    def level_specific_softmax_prob(self, level):
        """
        Get softmax probabilities for a specific hierarchical level.
        
        Args:
            level (int): Level to get probabilities for (1-indexed)
            
        Returns:
            numpy.ndarray: Softmax probabilities for specified level
            
        Raises:
            TypeError: If level is not an integer
            ValueError: If level is out of bounds
        """
        if not isinstance(level, int):
            raise TypeError("level must be an integer")
        if level < 1 or level > self._max_depth:
            raise ValueError(f"level must be between 1 and {self._max_depth}")
        
        softmax_probs = self._labels_prob[:,level-1]
        return softmax_probs
        

    def level_specific_labels(self, level):
        """
        level should be 1-indexed
        """
        labels = self._level_specific_labels_and_final_level[level-1]
        return labels

    def all_level_labels(self):
        """
        write: labels for all levels
        """
        all_level_labels = [
            self.level_specific_labels(level) 
            for level in range(self._max_depth)
            ]

        return all_level_labels

    
class PostprocessingAzimuthLabels(OutputLabels):
    """
    subclassing OutputLabels for postprocessing
    """
    VALID_REFINE_LEVELS = ['broad', 'medium', 'fine']
    
    def __init__(
        self, 
        labels_pred, 
        labels_prob, 
        max_depth, 
        num_cells,
        softmax_probs,
        encoders,
        refine_level
        ):
        
        if refine_level not in self.VALID_REFINE_LEVELS:
            raise ValueError(
                f"refine_level must be one of {self.VALID_REFINE_LEVELS}"
            )
            
        super().__init__(
            labels_pred, 
            labels_prob, 
            max_depth, 
            num_cells
            )

        self.softmax_probs = softmax_probs
        self.refine_level = refine_level
        self.encoders = encoders

        try:
            postprocessing_dir_path = files(postprocessing)
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                "postprocessing module not found. Please ensure it is installed"
            ) from e

        self.refined_annotations_file_name = (
            f'panhuman_annotate_{refine_level}.csv'
        )
        self._refined_annotations_df_path = (
            postprocessing_dir_path / self.refined_annotations_file_name
        )
        
        try:
            self.annotations_df = pd.read_csv(
                self._refined_annotations_df_path,
                index_col=None
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Annotation file {self.refined_annotations_file_name} not "
                f"found in postprocessing directory"
            ) from e
            
        required_columns = ['Orig_Label', f'Annotate_{refine_level}']
        missing_cols = [col for col in required_columns 
                       if col not in self.annotations_df.columns]
        if missing_cols:
            raise ValueError(
                f"Annotation file missing required columns: {missing_cols}"
            )

    def __combined_labels_w_consistency(self):

        labels_w_consistency_flag = [
            f"{self.combined_labels[i]}_{self.full_consistent_hierarchy[i]}"
            for i in range(self._num_cells)
        ]

        return labels_w_consistency_flag

    def __encoders_dict(self):

        enc_dicts = {}
        for i, encoder in enumerate(self.encoders):
            level_name = int(i)
            enc_dicts[level_name] = {
                label: index for index, label in enumerate(encoder.classes_)
                }

        return enc_dicts

    def __annotations_dict(self):

        annot_dict = dict(
            zip(
            self.annotations_df['Orig_Label'], 
            self.annotations_df[f'Annotate_{self.refine_level}']
            )
            )
        for fine_label in self.annotations_df[f'Annotate_{self.refine_level}']:
            annot_dict[fine_label] = fine_label

        return annot_dict

    def __softmax_arrays_dict(self):

        softmax = {
            f"arr_{i}": np.array(row) for i, row 
            in enumerate(self.softmax_probs)
            }

        return softmax

    def __map_to_annotate_refine(self, label): 
        label_split = label.split("_")
        flag = label_split[-1]
        label = '_'.join(label_split[:-1])
        annot_dict = self.__annotations_dict()

        if flag == "True":  
            refined_label = annot_dict.get(label)
            if pd.isna(refined_label):
                matching_df = self.annotations_df[
                    self.annotations_df['Orig_Label'].str.startswith(
                        label, na=False
                        )
                ]
                matching_refined_labels = matching_df[
                    f'Annotate_{self.refine_level}'
                ].tolist()
                matching_refined_labels = list(set(matching_refined_labels))
                return matching_refined_labels
            if refined_label == label:
                return refined_label
            elif refined_label in label:
                return refined_label
        return "False"

    def __refinement_type(self):

        labels_w_consistency_flag = self.__combined_labels_w_consistency()

        refine_type = pd.Series(labels_w_consistency_flag).apply(
            self.__map_to_annotate_refine
        )

        refine_type = refine_type.apply(categorize_refinement_type).tolist()

        return refine_type

    def __softmax_classes_dict(self):

        precomputed_softmax = {}

        levels_classes_dict = self.__encoders_dict()
        for level, level_dict in levels_classes_dict.items():
            arr_key = f"arr_{level}"
            softmax = self.__softmax_arrays_dict()
            if arr_key in softmax:
                softmax_array = softmax[arr_key]
                precomputed_softmax[level] = {
                    key: softmax_array[:, idx] for key, idx 
                    in level_dict.items() 
                    if idx < softmax_array.shape[1]
            }

        return precomputed_softmax

    def refine_labels(self):
        
        labels_w_consistency_flag = self.__combined_labels_w_consistency()
        labels_options = pd.Series(labels_w_consistency_flag).apply(
            self.__map_to_annotate_refine
        ).tolist()
        
        refined_labels = []

        refine_types = self.__refinement_type()
        precomputed_softmax = self.__softmax_classes_dict()

        for i in range(self._num_cells):
            if refine_types[i] == "Further":
                annotations = labels_options[i]
                max_value = -np.inf
                best_prediction = None

                for pred in annotations:
                    levels = len(pred.split("|")) - 1

                    if pred in precomputed_softmax[levels]:
                        value = precomputed_softmax[levels][pred][i]  
                        if value > max_value:
                            max_value = value
                            best_prediction = pred

            else:
                best_prediction = labels_options[i]

            refined_label = (
                best_prediction.split("|")[-1] if isinstance(
                    best_prediction, str
                    ) else best_prediction
            )

            refined_labels.append(refined_label)

        return refined_labels



class Embeddings():
    """
    write
    """
    def __init__(self, model, embedding_layer_name):
        self.model = model
        layer_names = [layer.name for layer in model.layers]
        assert embedding_layer_name in layer_names, (
            f"Layer '{embedding_layer_name}' not found in model."
        )
        self.embedding_layer = embedding_layer_name

        embedding = self.model.get_layer(self.embedding_layer).output
        self.embedding_model = Model(
            inputs=self.model.input, 
            outputs=embedding
            )

    def embeddings(self, X_query, batch_size):
        """
        Returns embeddings from the specified intermediate layer of the 
        Azimuth Neural Network model using the auxilliary model provided 
        by the function embedding_extractor_model(model, embedding_layer).
        """
        embedding_batches=[]
        eval_steps = X_query.shape[0]//batch_size

        for i in range(eval_steps):  
            with MemoryContext():
                X_batch = X_query[i*batch_size:(i+1)*batch_size].toarray()
                embeddings_batch = self.embedding_model.predict(X_batch)
                embedding_batches.append(embeddings_batch)
                del X_batch

        with MemoryContext(): 
            X_final_batch = X_query[eval_steps*batch_size:].toarray()  
            embeddings_final_batch = self.embedding_model.predict(
                X_final_batch
                )
            embedding_batches.append(embeddings_final_batch)
            del X_final_batch

        with MemoryContext():    
            embeddings = tf.concat(embedding_batches, axis=0)
            embedding_batches.clear()

        return embeddings.numpy()



class Umaps():
    """
    for creating umaps
    """
    def __init__(
        self, 
        n_neighbors,
        n_components,
        metric,
        min_dist,
        umap_lr,
        umap_seed,
        spread,
        verbose,
        init
        ):

        self._n_neighbors = n_neighbors
        self._n_components = n_components
        self._metric = metric
        self._min_dist = min_dist
        self._umap_lr = umap_lr
        self._umap_seed = umap_seed
        self._spread = spread
        self._verbose = verbose
        self._init = init 

        self.umap_model = umap.UMAP(
                                    n_neighbors=n_neighbors,   
                                    n_components=n_components,   
                                    metric=metric,     
                                    min_dist=min_dist,   
                                    learning_rate=umap_lr,    
                                    random_state=umap_seed,    
                                    spread=spread,
                                    verbose=verbose,
                                    init=init
                                    )

    def create_umap(self, input_data):
        """
        write
        """
        
        with MemoryContext():
            umap_embeddings = self.umap_model.fit_transform(input_data)

        return umap_embeddings


class EmbeddingsAndUmap(Embeddings, Umaps):
    """
    write
    """
    def __init__(
        self,
        model,
        embedding_layer_name,
        n_neighbors,
        n_components,
        metric,
        min_dist,
        umap_lr,
        umap_seed,
        spread,
        verbose,
        init
        ):

        Embeddings.__init__(self, model, embedding_layer_name)
        Umaps.__init__(
            self,
            n_neighbors,
            n_components,
            metric,
            min_dist,
            umap_lr,
            umap_seed,
            spread,
            verbose,
            init
        )

        

    def create_embeddings_and_umap(self, X_query, batch_size):
        """
        write
        """
        
        em = self.embeddings(X_query, batch_size)
        input_dim = em.shape[1] 

        print("Running UMAP:")
        print(
            f"    UMAP input: {self.embedding_layer}, "
            f"{input_dim} dimensions"
            )
        
        sparsity = 1 - (np.count_nonzero(em)/ em.size)
        print(f"Sparsity on embeddings is {sparsity}.\n")

        umap_embeddings = self.create_umap(em)

        return em, umap_embeddings














            








    




    






    

      
    
    












        
     

    
    








            


        




