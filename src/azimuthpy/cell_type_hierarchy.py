from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.preprocessing import LabelEncoder

from azimuthpy.datatypes import AnnotationHandler


@dataclass
class CellTypeHierarchy(AnnotationHandler):
    """Hierarchical cell type annotations derived from softmax probabilities
    and associated label encoders.

    Attributes:
        levels (list[str]):
            Names for each level in the hierarchy.
        cells (list[str | int]):
            Cell identifiers.
        softmax (xarray.Dataset):
            A dataset containing a "cell" by "encoding" array of softmax
            probabilities for each level in the hierarchy.
        argmax (xarray.Dataset):
            A dataset containing a one dimensional array of softmax encodings
            for each level in the hierarchy.
        argmax_labels (xarray.Dataset):
            A dataset containing a one dimensional array of cell type labels
            for each level in the hierarchy. Each label represents the
            predicted category name corresponding to the highest softmax
            probability at that level.
        labels_by_level (xarray.Dataset):
            A dataset similar to `argmax_labels` except with trailing
            delimiters removed.
        label_array_by_level (xarray.Dataset):
            A dataset containing a "cell" by "sublabel" array for each level
            in the hierarchy. Each value is derived from splitting the cell type
            label into its constituent parts using the '|' delimiter.
        label_array_by_cell (xarray.Dataset):
            A dataset containing a "level" by "sublabel" array for each cell.
        combined_label_array (xarray.DataArray):
            A DataArray of shape (cell, sublabel) where each entry
            represents the predicted label at each level for each cell.
        combined_labels (xarray.DataArray):
            A DataArray of strings for each cell representing the full
            hierarchical label joined by the '|' delimiter.
        strict_label_array (xarray.DataArray):
            A DataArray similar to `combined_label_array` but where labels
            after a conflict in the hierarchy are replaced with '~'.
        strict_labels (xarray.DataArray):
            A DataArray of strings for each cell representing the strict
            combined hierarchical label joined by '|' and truncated at
            the first inconsistency.
    """

    softmax: xr.Dataset

    def __post_init__(self):
        ...

    @classmethod
    def build(
        cls,
        softmax_probabilities: list[npt.NDArray],
        encoders: list[LabelEncoder],
        cells: list[str],
    ) -> CellTypeHierarchy:
        assert len(softmax_probabilities) == len(encoders)

        # Construct an xarray Dataset by creating a DataArray for each level:
        # for each softmax probability array and its corresponding encoder,
        # create a DataArray with dims ("cell", "encoding"), and attach the encoder
        # as an attribute.
        softmax = xr.Dataset(
            {
                f"level{i}": xr.DataArray(
                    array,
                    dims=("cell", "encoding"),
                    coords=dict(
                        cell=cells,
                        encoding=range(array.shape[1]),
                    ),
                    attrs=dict(encoder=encoder),
                )
                for i, (array, encoder) in enumerate(
                    zip(softmax_probabilities, encoders)
                )
            }
        )

        return cls(softmax=softmax)

    @property
    def levels(self) -> list[str]:
        return list(self.softmax.data_vars)

    @property
    def cells(self) -> list[str | int]:
        return self.softmax.coords["cell"].values.tolist()

    @cached_property
    def argmax(self) -> xr.Dataset:
        return self.softmax.argmax(dim="encoding", keep_attrs=True)

    @cached_property
    def argmax_labels(self) -> xr.Dataset:
        return self.argmax.map(lambda da: da.encoder.inverse_transform(da.data))

    @cached_property
    def labels_by_level(self) -> xr.Dataset:
        return self.argmax_labels.map(lambda da: da.str.rstrip("|"))

    @cached_property
    def label_array_by_level(self) -> xr.Dataset:
        return self.argmax_labels.map(lambda da: self._split_and_pad(da, self.levels))

    @staticmethod
    def _split_and_pad(label_array: xr.DataArray, levels: list[str]) -> xr.DataArray:
        # Split each label string on the '|' delimiter into a list of sublabels.
        da_split = label_array.str.split(dim=None, sep="|")

        # Pad each list of sublabels with '-' so that all lists have length
        # equal to the number of levels in the hierarchy.
        padded = [
            sublabels + ["-"] * (len(levels) - len(sublabels))
            for sublabels in da_split.values
        ]

        # Construct and return a DataArray of shape (cell, sublabel)
        # with coordinates for each cell and sublabel level.
        return xr.DataArray(
            padded,
            dims=["cell", "sublabel"],
            coords={
                "cell": label_array["cell"],
                "sublabel": levels,
            },
        )

    @cached_property
    def label_array_by_cell(self) -> xr.Dataset:
        return xr.Dataset(
            {
                cell: self.label_array_by_level.sel(cell=cell)
                .to_array(dim="level")
                .drop_vars("cell")
                for cell in self.cells
            }
        )

    @cached_property
    def combined_label_array(self) -> xr.DataArray:
        return self.label_array_by_cell.map(
            lambda da: self._combine_levels(da)
        ).to_array(dim="cell")

    @staticmethod
    def _combine_levels(label_array: xr.DataArray):
        # Identify the index of the first non-missing label at each sublabel
        # by taking the argmax of the mask where label_array is not '' or '-'.
        level_indices = (~label_array.isin(["", "-"])).argmax(dim="level")

        # Select and return the labels at those indices for each sublabel,
        # dropping the 'level' coordinate.
        return label_array[level_indices].drop_vars("level")

    @cached_property
    def combined_labels(self) -> xr.DataArray:
        return self.combined_label_array.str.join(dim="sublabel", sep="|").str.rstrip(
            ["|", "|-"]
        )

    @cached_property
    def strict_label_array(self) -> xr.DataArray:
        return self.label_array_by_cell.map(
            lambda da: self._combine_levels_strict(da)
        ).to_array(dim="cell")

    @staticmethod
    def _combine_levels_strict(label_array: xr.DataArray) -> xr.DataArray:
        nlevels, nlevels_ = label_array.shape
        assert nlevels == nlevels_

        # Extract the diagonal of the label array, which corresponds
        # to the label at each level for the cell.
        diagonal = np.diag(label_array).copy()
        combined_labels = diagonal.copy()

        # Extract the diagonal shifted by one (lookahead to next level)
        # to compare adjacent levels for consistency.
        diagonal_lookahead = np.diag(label_array, k=-1)

        # Determine consistency between each level and the next level:
        # True if the label at level i matches the label at level i+1.
        consistent = np.append((diagonal[:-1] == diagonal_lookahead), True)

        # Identify the indices where the hierarchy becomes inconsistent.
        conflicting_levels = np.where(~consistent)[0]

        # If any inconsistency is found, replace all labels beyond
        # the first conflict with the '~' placeholder.
        if conflicting_levels.size > 0:
            first_conflict = conflicting_levels[0] + 1
            combined_labels[first_conflict:] = "~"

        # Construct and return a DataArray of the strict combined labels.
        return xr.DataArray(
            combined_labels,
            dims=["sublabel"],
            coords={"sublabel": label_array.sublabel},
        )

    @cached_property
    def strict_labels(self) -> xr.DataArray:
        return self.strict_label_array.str.join(dim="sublabel", sep="|").str.rstrip(
            ["|", "|~"]
        )
