from __future__ import annotations

from collections import UserList
from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import Literal, SupportsIndex

import pandas as pd

from azimuthpy.cell_type_hierarchy import CellTypeHierarchy

__all__ = ["LabelCorrections", "LabelGroup"]


_module_root = files("azimuthpy.panhuman.postprocessing.label_correction")
_path_to_table_medium = _module_root / "correction_table_medium.csv"
_path_to_table_fine = _module_root / "correction_table_fine.csv"


@dataclass
class LabelCorrections:
    """A class to perform label corrections by mapping input labels to a
    group of candidate corrections.

    Attributes:
        label_map (dict[str, LabelGroup]):
            Mapping from an input label to a LabelGroup containing candidate
            labels.
    """

    label_map: dict[str, LabelGroup]

    @classmethod
    def load(cls, resolution: Literal["fine", "medium"]) -> LabelCorrections:
        """Load the label correction lookup table based on the specified
        resolution.

        The csv file is read into a `pandas.DataFrame`, grouped by the input
        label, and then each group is converted into a `LabelGroup`.

        Args:
            resolution (Literal["fine", "medium"]):
                Resolution for which to load corrections. Determines the csv
                file used.

        Returns:
            LabelCorrections: An instance with a populated label_map.
        """

        if resolution == "fine":
            path_to_table = _path_to_table_fine
        else:
            assert resolution == "medium"
            path_to_table = _path_to_table_medium

        with as_file(path_to_table) as fp:
            lookup_df = pd.read_csv(
                fp, names=["input_label", "output_label"], header=None
            )

        lookup = lookup_df.groupby("input_label")["output_label"].apply(list).to_dict()

        return cls({key: LabelGroup(vals) for key, vals in lookup.items()})

    def get_corrected_labels(self, azimuth_annotations: CellTypeHierarchy) -> list[str]:
        """Generate a list of corrected labels for each cell in the provided
        Azimuth annotations.

        For each cell, if a candidate correction exists in the label map, it
        will be used. If there are multiple candidates in the map then we
        assume that they are all valid encodings present in `azimuth_annotation`
        so that we can use their softmax probabilities to choose the
        appropriate correction. If no candidate correction exists the original
        label is retained.

        Args:
            azimuth_annotations (CellTypeHierarchy):
                Annotation object containing cell labels, softmax scores,
                and other metadata used for computing corrected labels.

        Returns:
            list[str]: A list of corrected label strings.
        """

        corrected_labels = []
        for cell, combined_label in zip(
            azimuth_annotations.cells, azimuth_annotations.combined_labels
        ):
            starting_label = combined_label.values.item()

            # If not corrections are available for the label, leave it as is.
            if starting_label not in self.label_map:
                corrected_labels.append(starting_label)
                continue

            candidate_labels = self.label_map[starting_label]
            # If there is only one candidate label, we can return it without
            # any further processing. In this case, labels are not constrained
            # to those already present in the dataset.
            if len(candidate_labels) == 1:
                corrected_labels.append(candidate_labels[0])
                continue

            # Determine which level in the hierarchy the candidate labels
            # correspond to.
            candidate_level = f"level{candidate_labels.label_depth}"
            # Get the softmax probabilities for the current cell at the
            # relevant level.
            softmax = azimuth_annotations.softmax[candidate_level].sel(cell=cell)
            # Encode each candidate label to align it with the "encoding"
            # dimension of the softmax array. This operation will fail if
            # the candidate labels are not present in the predicted output.
            candidate_encodings = softmax.encoder.transform(candidate_labels)
            # Get the softmax probabilities for each candidate label.
            candidate_probabilities = softmax.sel(encoding=candidate_encodings)
            # Get the encoding with the highest softmax probability.
            candidate_argmax = candidate_probabilities.argmax(dim="encoding")
            # Convert the argmax encoding back to a string label.
            corrected_label = softmax.encoder.inverse_transform(
                [candidate_argmax]
            ).item()

            corrected_labels.append(corrected_label)

        return corrected_labels


class LabelGroup(UserList):
    """A specialized list that enforces all items are strings representing
    hierarchical labels with a consistent depth. The label depth is determined
    by the number of '|' separators present in the string.
    """

    def __init__(self, *args):
        super().__init__(*args)

        for item in self.data:
            self._validate_item(item)

    def _validate_item(self, item: str):
        if not isinstance(item, str):
            raise ValueError(f"Expected item of type str, got {type(item).__name__}")
        elif self.get_label_depth(item) != self.label_depth:
            raise ValueError(
                f"Expected label with depth of {self.label_depth}, got {len(item)}"
            )

    @property
    def label_depth(self) -> int:
        return self.get_label_depth(self.data[0])

    @staticmethod
    def get_label_depth(label: str) -> int:
        return label.count("|")

    def __setitem__(self, i, item):
        self._validate_item(item)
        self.data[i] = str(item)

    def insert(self, i: SupportsIndex, item: str):
        self._validate_item(item)
        self.data.insert(i, str(item))

    def append(self, item):
        self._validate_item(item)
        self.data.append(str(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            if other.label_depth != self.label_depth:
                raise ValueError(
                    f"Expected group with a label depth of {self.label_depth}, "
                    f"got {other.label_depth}"
                )
            self.data.extend(other)
        else:
            other_list = []
            for item in other:
                self._validate_item(item)
                other_list.append(item)

            self.data.extend(other_list)
