# the code here is bloated, simplify

import argparse
import os
import pickle
import random

import scipy.sparse as sp

# import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def configure():
    """
    Limits default memory allocation on the GPU
    and allows for memory growth as needed. Sets
    the jit compilation flag to True.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tf.config.optimizer.set_jit(True)


def if_val(x):
    """
    Returns 0 if the arg has a NoneType value or a Boolean False value,
    returns 1 otherwise.
    """

    if x:
        return 1
    else:
        return 0


def cumulative_split(string, delimiter="|"):
    """
    Gives a cumulative split of a string separated with a delimiter.
    For example, if the string is "The|sun|is|out." and if the
    delimiter is '|', the output will be:
    ['The', 'The|sun', 'The|sun|is', 'The|sun|is|out.']

    Args:
        string (str): The string to split.
        delimiter (str): The delimiter to split w.r.t.

    Returns:
        list: A list of strings as described above.
    """
    split = []
    current = ""
    for item in string.split(delimiter):
        current += item
        split.append(current)
        current += delimiter

    return split


def shuffle_lists(list_a, list_b):
    """
    Shuffles two lists parallely.

    Args:
        list_a (list): first list.
        list_b (list): second list.

    Returns:
        list, list: list_a and list_b shuffled parallely.
                    This operation is not in place.
    """

    list_c = list(zip(list_a, list_b))
    random.shuffle(list_c)

    list_as, list_bs = zip(*list_c)

    return list_as, list_bs


def shuffle_folds(X_folds, y_folds):
    """
    Shuffles the list of X folds and the list of y folds parallely.

    Args:
        X_folds (list): list of folds of X in csr format.
        y_folds (list): list of folds of y as strings.
    Returns:
        list, list: list of folds of X and folds of y shuffled
        parallely.
    """

    return shuffle_lists(X_folds, y_folds)


def load_X(filepath):
    """
    Loads X as a csr matrix from the provided filepath.
    """
    with open(filepath, "rb") as f:
        X = pickle.load(f)

    return X


def load_labels(filepath):
    """
    Loads the labels as a list of strings from the provided filepath.
    """
    with open(filepath, "r") as f:
        labels = f.read()
        labels_list = labels.split("\n")
        labels_list = labels_list[:-1]  # dropping the last empty line

    return labels_list


def load_data(data_dir_path):
    """
    Loads the X and y from the kfold datasets as lists of folds.

    Args:
        data_dir_path (str): absolute path to the k-fold dataset.

    Returns:
        list, list: list of folds of X, list of folds of y.
    """
    data_dir = data_dir_path
    folds_X = []
    folds_labels = []
    for k in range(10):
        X_file = os.path.join(data_dir, "fold_" + str(k) + "/X_fold.pkl")
        labels_file = os.path.join(data_dir, "fold_" + str(k) + "/labels.txt")

        with open(labels_file, "r") as f:
            labels_fold = f.read()
            labels_fold_list = labels_fold.split("\n")
            labels_fold_list = labels_fold_list[
                :-1
            ]  # there's an empty string entry as the last
            # element which we want to drop
        with open(X_file, "rb") as f:
            X_fold = pickle.load(f)

        folds_X.append(X_fold)
        folds_labels.append(labels_fold_list)

    return folds_X, folds_labels


def mask_tail(hierarchical_labels, p):
    """
    Takes hierarchical labels for one cell and masks a max of 1/p number
    of levels (and a min of 1 unless the max amounts to 0) from the tail
    end. For example, if the hierarchical labels for a cell are
    ['Immune cell', 'Myeloid cell', 'Dendritic cell', 'cDC'], and p=0.5,
    then passing it through this function will mask or delete either one
    or two (randomly chosen) of the levels from the tail. So the output
    will either be ['Immune cell', 'Myeloid cell', 'Dendritic cell'] or
    ['Immune cell', 'Myeloid cell']

    Args:
        hierarchical_labels (list): Hierarchical labels for a single cell.
        p (float): Fraction of levels to be masked from the tail end.

    Returns:
        list: Hierarchical labels for a cell with tail masked.
    """
    if p >= 1.0 or p < 0.0:
        raise ValueError("Value of p should be in the range [0,1).")

    depth = len(hierarchical_labels)
    max_tail_length = int(depth * p)
    min_tail_lenth = min(1, max_tail_length)
    tail_length = random.randint(min_tail_lenth, max_tail_length)

    return hierarchical_labels[: (depth - tail_length)]


def mask_single_level(hierarchical_labels):
    """
    Takes hierarchical labels for one cell and masks a randomly chosen
    level. For example, if the hierarchical labels for a cell are
    ['Immune cell', 'Myeloid cell', 'Dendritic cell', 'cDC'], then
    passing it through this function will mask or delete any one of the
    hierarchical labels and replace it with a blank ''. So the output
    could be ['Immune cell', 'Myeloid cell', '', 'cDC'] for example.

    Args:
        hierarchical_labels (list): Hierarchical labels for a single cell.

    Returns:
        list: Hierarchical labels for a cell with one level masked.
    """

    depth = len(hierarchical_labels)
    single_level = random.randint(0, depth - 1)

    current_hierarchical_labels = (
        hierarchical_labels.copy()
    )  # ensuring that hierarchical labels
    # is not changed in place.

    current_hierarchical_labels[single_level] = ""

    return current_hierarchical_labels


class Labels:
    """
    Instantiates a Label class object to process cell type labels, produce independent or
    cumulative hierarchical labels, and output one hot encoded vectors in a chosen mode.

    Arguments:
        labels (list): Cell type labels as a list of strings.
        delimiter (string): The delimiter separating labels at different hierarchies,
                            defaults to '|'.
        max_depth_provided (int): defaults to None, in which case it is calculated. Provide
                                it manually unless you're sure that the list of labels
                                under consideration is a representative sample of the
                                entire dataset.

    Attributes:
        labels (list): returns the input list of labels.
        delimiter (str): returns the delimiter being used.
        hierarchical_labels (list): returns list of labels with each cell label now split
                                    into independent hierarchical labels. For example,
                                    'Endothelial cell|Capillary EC' is turned into
                                    ['Endothelial cell', 'Capillary EC'].
        cumulative_labels (list): returns list of labels with each cell label now split
                                    into cumulative hierarchical labels. For example,
                                    'Endothelial cell|Capillary EC' is turned into
                                    ['Endothelial cell', 'Endothelial cell|Capillary EC'].
        max_depth (int): the max number of hierarchical levels in cell type labels in the
                            dataset.

    Methods:
        encode(mode)
        get_one_hot(mode)
        enc_trans_goh(mode, label_encoders)
        tail_mask(tm_frac, seed)
        single_level_mask(lm_frac, seed)
    """

    def __init__(self, labels, delimiter="|", max_depth_provided=None):
        self.labels = labels
        self.delimiter = delimiter
        self.sample_size = len(self.labels)

        hierarchical_labels = []
        cumulative_labels = []
        max_depth = 0
        for label in self.labels:
            split_label = label.split(self.delimiter)
            hierarchical_labels.append(split_label)

            cumulative_split_label = cumulative_split(label, self.delimiter)
            cumulative_labels.append(cumulative_split_label)

            # other kinds of splitting (more modes)
            # add here as needed later

            if len(split_label) > max_depth:
                max_depth = len(split_label)

        self.hierarchical_labels = hierarchical_labels
        self.cumulative_labels = cumulative_labels

        if max_depth_provided:
            self.max_depth = max_depth_provided
        else:
            self.max_depth = max_depth

    def encode(self, mode):
        """
        Given a mode, it encodes the cell type labels at different hierarchical levels
        into numbered classes and also provides the corresponding label encoders for each
        level.

        Note that this method should only be called when the Label object in question
        represents all cell types in the dataset, as it gives an encoding that we need
        to consistently use across all samples from this dataset.

        This method should not be called independently on two different samples from the
        same dataset, otherwise we shall get non-sensical results.

        Args:
            mode (str): either 'independent' or 'cumulative'. More modes can be added later.

        Returns:
            list[sklearn.preprocessing.LabelEncoder]: a list of label encoders at each level
                                                        that have been fit to the existing
                                                        classes at the corresponding level.
            list[array]: a list of encoded labels with each row being an array corresponding
                        to a hierarchical level and each column of these arrays corresponding
                        to a cell. For example if our dataset had 10 cells, and the max
                        hierarchical depth were 6, the encoded labels could look as follows:
                                            [array([3, 2, 0, 1, 0, 2, 2, 1, 1, 2]),
                                            array([1, 6, 0, 2, 0, 4, 6, 5, 3, 6]),
                                            array([6, 1, 0, 5, 0, 3, 1, 0, 4, 2]),
                                            array([1, 4, 0, 0, 0, 2, 5, 0, 3, 0]),
                                            array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                                            array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])]
            list[int]: a list of integers giving the number of classes at each hierarchical
                        level in the chosen mode.
        """

        if (
            mode != "independent" and mode != "cumulative"
        ):  # more modes can be added later.
            raise ValueError("Mode should either be 'independent' or 'cumulative'.")

        if mode == "independent":
            split_labels = self.hierarchical_labels
        elif mode == "cumulative":
            split_labels = self.cumulative_labels
        # options to add to this conditional statement later.

        label_encoders = [LabelEncoder() for _ in range(self.max_depth)]
        encoded_labels = [[] for _ in range(self.max_depth)]

        for cell_label in split_labels:
            deepest_label = cell_label[-1]
            for i in range(len(cell_label)):
                encoded_labels[i].append(cell_label[i])
            for i in range(len(cell_label), self.max_depth):
                if mode == "independent":
                    encoded_labels[i].append("")
                elif mode == "cumulative":
                    deepest_label += "|"
                    encoded_labels[i].append(deepest_label)
                # add more options later.

        for i in range(self.max_depth):
            encoded_labels[i] = label_encoders[i].fit_transform(encoded_labels[i])

        num_classes = [max(encoded_labels[i]) + 1 for i in range(self.max_depth)]

        return label_encoders, encoded_labels, num_classes

    def get_one_hot(self, mode):
        """
        Given a mode, it performs a one-hot encoding of the cell type labels at different
        hierarchical levels.

        Note that this method should only be called when the Label object in question
        represents all cell types in the dataset, as it is implementing an class encoding
        that we need to consistently use across all samples from this dataset.

        This method should not be called independently on two different samples from the
        same dataset, otherwise we shall get non-sensical results.

        Args:
            mode (str): either 'independent' or 'cumulative'. More modes can be added later.

        Returns:
            list: a list with each row being an array corresponding to a hierarchical level
                and each column of these arrays corresponding to a cell. Each element of said
                array is a one-hot encoded vector corresponding to the level specific class.
                It's basically a one-hot hot encoding of the encoded labels produced my the
                encode(mode) method.
        """
        _, encoded_labels, _ = self.encode(mode)

        one_hot_labels = [to_categorical(labels) for labels in encoded_labels]

        return one_hot_labels

    def enc_trans_goh(self, mode, label_encoders, num_classes):
        """
        Given a mode, it performs a one-hot encoding of the cell type labels at different
        hierarchical levels, but it's different from the encode(mode) and get_one_hot(mode)
        methods in that it's using label_encoders provided to it (that have already been
        fit to the dataset) to encode say a part of the dataset in a manner consistent
        with the existing encoding.

        If the list of labels to be encoded is only a small sample of the entire dataset
        and the dataset has already been encoded, this method should be used to produce
        the one-hot encodings for hierarchical cell type labels. num_classes is also
        provided to it as a small sample from the dataset may not feature all classes
        present in the total dataset and the one-hot encoding needds to be performed w.r.t.
        the number of classes in the total dataset.

        Make sure that the mode with which the label encoders have been generated matches
        the mode in which they are being used, and similarly for the number of classes.
        """

        if (
            mode != "independent" and mode != "cumulative"
        ):  # more modes can be added later.
            raise ValueError("Mode should either be 'independent' or 'cumulative'.")
        if mode == "independent":
            split_labels = self.hierarchical_labels
        elif mode == "cumulative":
            split_labels = self.cumulative_labels
        # options to add to this conditional statement later.

        encoded_labels = [[] for _ in range(self.max_depth)]
        for cell_label in split_labels:
            deepest_label = cell_label[-1]
            for i in range(len(cell_label)):
                encoded_labels[i].append(cell_label[i])
            for i in range(len(cell_label), self.max_depth):
                if mode == "independent":
                    encoded_labels[i].append("")
                elif mode == "cumulative":
                    deepest_label += "|"
                    encoded_labels[i].append(deepest_label)
                # add more options later.

        for i in range(self.max_depth):
            encoded_labels[i] = label_encoders[i].transform(encoded_labels[i])

        one_hot_labels = [
            to_categorical(encoded_labels[i], num_classes[i])
            for i in range(self.max_depth)
        ]

        return one_hot_labels

    # add masking functions
    def tail_mask(self, tm_frac, seed):
        """
        Given a list of labels it chooses a specified fraction of all cells and deletes
        some of the hierarchies from the tail end.

        Args:
            tm_frac (list[float, float]): Each of these floats needs to be a fraction
                                            between 0 and 1. The first specifies
                                            specifies the fraction of cells to be
                                            subject to tail masking. For each cell
                                            chosen for this, the second fraction
                                            gives an upper bound on frac of hierarchical
                                            depth in the corresponding cell type label
                                            that we are allowed to snip off from the tail.
            seed (int): a seed for the random sampling and the random choice of levels
                        to delete (subject to an upper bound).

        Returns:
            list: A sorted listed of indices specifying which cells have been subject to
                    tail masking.
            list: The total list of labels only some of the cell types are now missing their
                    tails.
        """

        random.seed(seed)
        q, p = tm_frac

        if q < 0.0 or q >= 1.0:
            raise ValueError("tm_frac should have fractions in the range [0,1).")

        masked_sample_size = int(self.sample_size * q)

        masked_sample_ind = random.sample(
            list(range(self.sample_size)), masked_sample_size
        )
        masked_sample_ind.sort()

        current_hierarchical_labels = self.hierarchical_labels

        for i in masked_sample_ind:
            current_hierarchical_labels[i] = mask_tail(self.hierarchical_labels[i], p)

        masked_labels = [
            "|".join(cell_labels) for cell_labels in current_hierarchical_labels
        ]

        return masked_sample_ind, masked_labels

    def single_level_mask(self, lm_frac, seed):
        """
        Given a list of labels it chooses a specified fraction of all cells and deletes
        a single randomly chosen level from the hierarchies.

        Args:
            lm_frac (float): A fraction between 0 and 1. Specifies the fraction of cells to
                             be subject to tail masking.
            seed (int): a seed for the random level to delete.

        Returns:
            list: A sorted listed of indices specifying which cells have been subject to
                    single level masking.
            list: The total list of labels only some of the cell types are now missing a
                    level.
        """

        random.seed(seed)
        p = lm_frac
        if p >= 1.0 or p < 0.0:
            raise ValueError("Value of p should be in the range [0,1).")

        masked_sample_size = int(self.sample_size * p)

        masked_sample_ind = random.sample(
            list(range(self.sample_size)), masked_sample_size
        )
        masked_sample_ind.sort()

        current_hierarchical_labels = self.hierarchical_labels

        for i in masked_sample_ind:
            current_hierarchical_labels[i] = mask_single_level(
                self.hierarchical_labels[i]
            )

        masked_labels = [
            "|".join(cell_labels) for cell_labels in current_hierarchical_labels
        ]

        return masked_sample_ind, masked_labels


class Data_proc:
    """
    A general class for processing of single cell data comprising of X which has transcriptomic
    reads for a set of genes for each cell in the dataset, and the corresponding hierarchical
    cell type labels.

    Arguments:
        data_dir_path (str): path to directory containing X as .pkl files and labels
                            in .txt files.

    Attributes:
        dir (str): returns the directory path from where the data is being read.

    Methods:
        give_X(X_filename)
        give_labels(labels_filename)
        give_y(labels_filename, mode)
        enc_trans_y(labels_filename, mode, label_encoders)
    """

    def __init__(self, data_dir_path):
        self.dir = data_dir_path

    def give_X(self, X_filename):
        """
        Reads X from a .pkl file and returns it as a csr matrix.

        Args:
            X_filename (str): A .pkl filename, relative to the dir attribute, do not need
                                the entire filepath.
        Returns:
            csr matrix: X
        """
        filepath = os.path.join(self.dir, X_filename)

        return load_X(filepath)

    def give_labels(self, labels_filename):
        """
        Reads the cell type labels from a .txt file and returns them as a list of strings.

        Args:
            labels_filename (str): A .txt filename relative to the dir attribute, do not
                                    need the entire filepath.
        Returns:
            list: A list of strings, each string being a cell type label.
        """
        filepath = os.path.join(self.dir, labels_filename)

        return load_labels(filepath)

    def give_y(self, labels_or_filename, mode):
        """
        Given the filename with the labels and a mode in which to split the cell type labels
        into hierarchies or levels, it returns a one-hot encoding of the different classes at
        each level, returns the label encoders for each level that have been fit to the
        class labels, and the max depth of hierarchy in the dataset.

        This function should only be called on a labels file that contains a representative
        sample of the entire dataset, otherwise the label encoders fit to these labels may
        not be useful for other samples taken from the dataset.

        For one-hot encoding a small
        set of labels that may not represent all class labels at all levels, use method
        enc_trans_y which will demand as an argument label encoders that have already been fit
        to the dataset.

        Args:
            labels_or_filename (str): A list of labels or a .txt filename relative to the dir
                                    attribute, do not provide the entire filepath.
            mode (str): should be either 'independent' or 'cumulative'. In the 'independent'
                        mode, cell type labels will be split into individual hierarchical
                        levels, for example 'Endothelial cell|Capillary EC' is turned into
                        ['Endothelial cell', 'Capillary EC'] before (missing depth is filled
                        with '' and) the classes at each level are one-hot encoded. In the
                        'cumulative' mode, 'Endothelial cell|Capillary EC' is turned into
                        ['Endothelial cell', 'Endothelial cell|Capillary EC'] and the rest
                        of the algorithm remains unchanged.

        Returns:
            list: one-hot encoding of the classes at each hierarchical level. The outermost
                list has rows corresponding to the levels, and each column across these rows
                corresponds to a cell. Each element in this grid is a one-hot encoded vector.
            sklearn.preprocessing.LabelEncoder: label encoders that have been fit to the
                                                    classes at each level.
            int: max depth of hierarchy, note that not all cells have labels that go equally
                deep, the missing levels at the tail are filled in with '' that is treated as
                a label in it's own right.
            list[int]: a list of ints specifying the number of classes at each hierarchical
                        level in the chosen mode.

        """
        if mode != "independent" and mode != "cumulative":  # can add options here later
            raise ValueError("Mode should either be 'independent' or 'cumulative'.")

        if type(labels_or_filename) == str:
            labels_list = self.give_labels(labels_or_filename)
        elif type(labels_or_filename) == list:
            labels_list = labels_or_filename
        else:
            raise TypeError("labels_or_filename should be either a string or a list.")

        labels_proc = Labels(labels_list)
        max_depth = labels_proc.max_depth
        label_encoders, _, num_classes = labels_proc.encode(mode)
        oh_y = labels_proc.get_one_hot(mode)

        return oh_y, label_encoders, max_depth, num_classes

    def enc_trans_y(
        self, labels_or_filename, mode, label_encoders, num_classes, max_depth_provided
    ):
        """
        Given the filename with the labels, a mode in which to split the cell type labels
        into hierarchies or levels, and label encoders that have already been fit to class
        labels at each level in the dataset, it returns a one-hot encoding of the different
        classes at each level for the labels in the file provided.

        Note that this is meant to be used when one-hot encoding a sample from the entire
        dataset according to the encoding that has already been used for the dataset. For
        example, one can use this on the test set after training and validation have been
        done, using the label encoders from there.

        Args:
            labels_or_filename (str): A list of labels or a .txt filename relative to the dir
                                    attribute, do not provide the entire filepath.
            mode (str): should be either 'independent' or 'cumulative'. In the 'independent'
                        mode, cell type labels will be split into individual hierarchical
                        levels, for example 'Endothelial cel|Capillary EC' is turned into
                        ['Endothelial cell', 'Capillary EC'] before (missing depth is filled
                        with '' and) the classes at each level are one-hot encoded. In the
                        'cumulative' mode, 'Endothelial cell|Capillary EC' is turned into
                        ['Endothelial cell', 'Endothelial cell|Capillary EC'] and the rest
                        of the algorithm remains unchanged.
            label_encoders (list[sklearn.preprocessing.LabelEncoder]): label encoders that
                                                                    have already been fit
                                                                    to the dataset.
            num_classes (list): a list of ints specifying the number of classes at each
                                hierarchical level in the total dataset in the chosen mode.
            max_depth_provided (int): max_depth in the training data, better to provide it
                                        manually here as we may want to apply this method
                                        on a subset of labels that has a shorter max depth.


        Returns:
            list: one-hot encoding of the classes at each hierarchical level. The outermost
                list has rows corresponding to the levels, and each column across these rows
                corresponds to a cell. Each element in this grid is a one-hot encoded vector.

        """
        if type(labels_or_filename) == str:
            labels_list = self.give_labels(labels_or_filename)
        elif type(labels_or_filename) == list:
            labels_list = labels_or_filename
        else:
            raise TypeError("labels_or_filename should be either a string or a list.")

        labels_proc = Labels(labels_list, max_depth_provided=max_depth_provided)

        return labels_proc.enc_trans_goh(mode, label_encoders, num_classes)


class Train_valid_data(Data_proc):
    """
    Subclass of Data_proc class to load training and validation data, with additional
    methods for masking training data and creating test datasets meant to test the model's
    generalizability.

    Arguments:
        data_dir_path (str): path to directory containing X as .pkl files and labels
                            in .txt files.
        split_mode (str): either "cumulative" or "independent", specifies whether the cell type labels
                            are to be split into different hierarchical levels independently or
                            cumulatively.
        X_train_file (str): .pkl file with the training X, defaults to "X_train.pkl". The filepath
                             should be relative to data_dir_path.
        X_valid_file (str): .pkl file with the validation set X, defaults to "X_valid.pkl". The
                            filepath should be relative to data_dir_path.
        labels_train_file (str): .txt file with the training set labels, defaults to "labels_train.txt".
                                    Provide relative filepath as above.
        labels_valid_file (str): .txt file with the validation set labels, defaults to
                                    "labels_valid.txt". Provide relative filepath as above.
        mask_seed (int): seed to be used in the pseudo-random processes involved in masking labels
                            defaults to None.
        mask_dir (str): dir path to which the X, masked labels, and true labels (corresponding to the
                        cells whose labels are masked) are saved, defaults to None.
        mask_mode (str): either "tail" or "single_level" if masking is applied, defaults to None.
        *Note that mask_seed, mask_dir, and mask_mode need to all be None or each of them need to have
        an appropriate value, else a ValueError will be raised.
        tm_frac (list[float, float]): a list of two floats in the range [0,1) specifying the fraction
                                        of cells to which the masking of label-tails is to be applied,
                                        and the size of the tail relative to the label depth in levels.
                                        Defaults to None.
        lm_frac (float): a fraction in the range [0,1) specifying the fraction of the train dataset to
                        which single level masking is to be applied. Defaults to None.
        *Note that one and only one of tm_frac and lm_frac need to be provided if mask_seed, mask_dir,
        and mask_mode are provided. Else a ValueError will be raised.
        save_masked_data (bool): whether the data for the cells that have been subjected to masking is
                                to be saved separately. Defaults to False.

    Attributes:
        X_train_file (str): .pkl file with the training X, defaults to "X_train.pkl". The filepath
                             should be relative to data_dir_path.
        X_valid_file (str): .pkl file with the validation set X, defaults to "X_valid.pkl". The
                            filepath should be relative to data_dir_path.
        labels_train_file (str): .txt file with the training set labels, defaults to "labels_train.txt".
                                    Provide relative filepath as above.
        labels_valid_file (str): .txt file with the validation set labels, defaults to
                                    "labels_valid.txt". Provide relative filepath as above.

        X_train (csr matrix): X for the training set.
        X_valid (csr_matrix): X for the validation set.
        num_features (int): number of genes/features.
        labels_train (list): list of cell type labels in the training set.
        labels_valid (list): list of cell type labels in the validation set.
        len_train (int): number of examples in the train set.
        len_valid (int): number of examples in the valid set.
        masked_ind (list[int]): list of cell indices specifying which cells in the train set have
                                been subject to label masking.
        masked_labels (list[str]): the list of cell type labels for the train set only some of
                                them have now been subject to masking.
        labels_train_true_n_masked (list[str]): a list showing the set of all cell type labels
                                                from the train set and any new ones produced through
                                                the masking procedure.
        max_depth (int): maximum level depth of labels in the training dataset, this is unchanged by
                        the masking procedure.
        label_encoders (list[sklearn.preprocessing.LabelEncoder]): a list of LabelEncoder objects, one
                                                corresponding to each hierarchical level in the cell type
                                                labels in the training dataset. These have been fit to
                                                the set of all labels in the train set and any additional
                                                labels produced through the masking process. The encoding
                                                process is ofc sensitive to the split_mode.
        encoded_labels (list): list of all cell type labels, after the masking process if any, with the
                                classes at each level now encoded to integer class ids.
        num_classes (list[int]): number of classes at each level. Note that if a masking process has been
                                applied, this will typically be greater than the number of classes without
                                masking.
        split_mode (str): returns the split_mode argument.

    Methods:
        give_y_train_valid()
        give_y_n_depth_train_valid()
    """

    def __init__(
        self,
        data_dir_path,
        split_mode,
        X_train_file="X_train.pkl",
        X_valid_file="X_valid.pkl",
        labels_train_file="labels_train.txt",
        labels_valid_file="labels_valid.txt",
        mask_seed=None,
        mask_dir=None,
        mask_mode=None,
        tm_frac=None,
        lm_frac=None,
        save_masked_data=False,
    ):
        super().__init__(data_dir_path)

        if split_mode != "independent" and split_mode != "cumulative":
            raise ValueError(
                "split_mode should either be 'independent' or 'cumulative'."
            )

        self.mask_check = if_val(mask_seed) + if_val(mask_dir) + if_val(mask_mode)
        if self.mask_check != 0 and self.mask_check != 3:
            raise ValueError(
                "Either all three of mask_seed, mask_dir, mask_mode needs to be provided or they should all be None."
            )

        self.X_train_file = X_train_file
        self.X_valid_file = X_valid_file
        self.labels_train_file = labels_train_file
        self.labels_valid_file = labels_valid_file

        self.X_train = self.give_X(X_train_file)
        self.X_valid = self.give_X(X_valid_file)

        self.num_features = self.X_valid.shape[1]

        self.labels_train = self.give_labels(labels_train_file)
        self.labels_valid = self.give_labels(labels_valid_file)

        self.len_train = len(self.labels_train)
        self.len_valid = len(self.labels_valid)

        if self.mask_check > 0:
            if if_val(tm_frac) + if_val(lm_frac) != 1:
                raise ValueError(
                    "One and exactly one of tm_frac or lm_frac need to be provided."
                )

            labels_to_mask = Labels(self.labels_train)

            if mask_mode != "tail" and mask_mode != "single_level":
                raise ValueError("mask_mode should be either 'tail' or 'single_level")

            if mask_mode == "tail":
                self.masked_ind, self.masked_labels = labels_to_mask.tail_mask(
                    tm_frac, mask_seed
                )
            elif mask_mode == "single_level":
                self.masked_ind, self.masked_labels = labels_to_mask.single_level_mask(
                    lm_frac, mask_seed
                )

            if save_masked_data:
                masked_dir = os.path.join(mask_dir, "masked_cells_data")
                os.makedirs(masked_dir, exist_ok=True)

                X_print_path = os.path.join(masked_dir, "X.pkl")
                X_masked = self.X_train[self.masked_ind]
                with open(X_print_path, "wb") as f:
                    pickle.dump(X_masked, f)

                labels_masked_print_path = os.path.join(masked_dir, "labels_masked.txt")
                with open(labels_masked_print_path, "w") as f:
                    for i in self.masked_ind:
                        f.write(self.masked_labels[i] + "\n")

                labels_true_print_path = os.path.join(masked_dir, "labels_true.txt")
                with open(labels_true_print_path, "w") as f:
                    for i in self.masked_ind:
                        f.write(self.labels_train[i] + "\n")

            self.labels_train_true_n_masked = list(
                set(self.labels_train + self.masked_labels)
            )

        else:
            self.masked_int = None
            self.masked_labels = None
            self.labels_train_true_n_masked = list(set(self.labels_train))

        labels_set_to_encode = Labels(self.labels_train_true_n_masked)
        self.max_depth = labels_set_to_encode.max_depth

        (
            self.label_encoders,
            self.encoded_labels,
            self.num_classes,
        ) = labels_set_to_encode.encode(split_mode)

        self.split_mode = split_mode

    def give_y_train_valid(self):
        """
        Gives the cell type labels in the training and validation sets, split into hierarchies,
        masked and encoded acccording to parameters chosen for initialization, in a one-hot
        encoded form.

        Returns:
            list: one-hot encoded hierarchical cell type labels from the training set. If masking
                has been applied, then this is the training set produced after masking. For more
                details see give_y method in the Data_proc class.
            list: one-hot encoded hierarchical cell type labels from the validation set.
        """
        if self.mask_check > 0:
            training_labels = self.masked_labels
        else:
            training_labels = self.labels_train

        y_train = self.enc_trans_y(
            training_labels,
            self.split_mode,
            self.label_encoders,
            self.num_classes,
            self.max_depth,
        )
        y_valid = self.enc_trans_y(
            self.labels_valid,
            self.split_mode,
            self.label_encoders,
            self.num_classes,
            self.max_depth,
        )

        return y_train, y_valid

    def give_y_n_depth_train_valid(self):
        """
        write,
        oh labels n depth
        """
        if self.mask_check > 0:
            training_labels = self.masked_labels
        else:
            training_labels = self.labels_train

        train_depth_vals = [(len(label.split("|")) - 1) for label in training_labels]
        train_depth_oh = to_categorical(train_depth_vals, self.max_depth)

        valid_depth_vals = [(len(label.split("|")) - 1) for label in self.labels_valid]
        valid_depth_oh = to_categorical(valid_depth_vals, self.max_depth)

        y_train = self.enc_trans_y(
            training_labels,
            self.split_mode,
            self.label_encoders,
            self.num_classes,
            self.max_depth,
        )
        y_train.append(train_depth_oh)

        y_valid = self.enc_trans_y(
            self.labels_valid,
            self.split_mode,
            self.label_encoders,
            self.num_classes,
            self.max_depth,
        )
        y_valid.append(valid_depth_oh)

        return y_train, y_valid


# saving train/valid/test data


def save_data(seed, data_split, data_dir_path, out_dir):
    """
    Function for the executable part of the script. Give a [train, valid, test] split, it splits
    all of the dataset accordingly and saves the X and the labels in the out_dir provided as
    .pkl files and .txt files respectively. The data is by-default sourced from a kfold dataset
    and there is seed controlled pseudo-randomness in choosing which folds to use for the different
    sets.
    """

    random.seed(seed)
    idx_list = list(range(10))  # for 10 folds
    random.shuffle(idx_list)

    X_folds, y_folds = load_data(data_dir_path)
    X_folds = [X_folds[i] for i in idx_list]
    y_folds = [y_folds[i] for i in idx_list]

    X_train = sp.vstack(X_folds[: data_split[0]])
    X_valid = sp.vstack(X_folds[data_split[0] : data_split[0] + data_split[1]])
    X_test = sp.vstack(X_folds[data_split[0] + data_split[1] :])

    X_train_out = os.path.join(out_dir, "X_train.pkl")
    with open(X_train_out, "wb") as f:
        pickle.dump(X_train, f)

    X_valid_out = os.path.join(out_dir, "X_valid.pkl")
    with open(X_valid_out, "wb") as f:
        pickle.dump(X_valid, f)

    X_test_out = os.path.join(out_dir, "X_test.pkl")
    with open(X_test_out, "wb") as f:
        pickle.dump(X_test, f)

    y = []
    for y_fold in y_folds:
        for label in y_fold:
            y.append(label)

    labels_train = y[: X_train.shape[0]]
    labels_valid = y[X_train.shape[0] : X_train.shape[0] + X_valid.shape[0]]
    labels_test = y[X_train.shape[0] + X_valid.shape[0] :]

    labels_train_file = os.path.join(out_dir, "labels_train.txt")
    labels_valid_file = os.path.join(out_dir, "labels_valid.txt")
    labels_test_file = os.path.join(out_dir, "labels_test.txt")
    with open(labels_train_file, "w") as f:
        for label in labels_train:
            f.write(label + "\n")
    with open(labels_valid_file, "w") as f:
        for label in labels_valid:
            f.write(label + "\n")
    with open(labels_test_file, "w") as f:
        for label in labels_test:
            f.write(label + "\n")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="enter the seed", type=int)
    parser.add_argument(
        "-sp",
        "--split",
        default=[7, 1, 2],
        help="enter the train:valid:test split as a list of ints, default: [7,1,2]",
        type=list[int, int, int],
    )
    parser.add_argument(
        "-ds",
        "--data_source",
        default="data/kfold_data/datasets/fold10_02_26_2025_17_53_139",
        help="enter the source dataset with no '/' at either end",
        type=str,
    )
    args = parser.parse_args()

    seed = args.seed

    data_split = args.split

    data_split_str = (
        str(data_split[0]) + "_" + str(data_split[1]) + "_" + str(data_split[2])
    )

    data_source = args.data_source

    data_source_id = data_source.split("/")[-1]

    cwd = os.getcwd()
    data_dir_path = os.path.join(cwd, data_source)
    exp_dir = os.path.join(
        cwd,
        "Experiments/" + data_source_id + "/data_" + str(seed) + "_" + data_split_str,
    )
    os.makedirs(exp_dir, exist_ok=True)

    save_data(seed, data_split, data_dir_path, exp_dir)


if __name__ == "__main__":
    run()
