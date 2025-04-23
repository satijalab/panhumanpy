import itertools
import string

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.preprocessing import LabelEncoder

from azimuthpy import CellTypeHierarchy


@pytest.fixture
def expected_labels_by_level() -> dict[str, list[str]]:
    return dict(
        level0=["A0", "A0"],
        level1=["A0|B0", "A0|B0"],
        level2=["A0|B0|C0", "A0|B0|C0"],
        level3=["A0|B0|C0|D0", "A0|B0|C1|"],
        level4=["A0|B0|C0|D0|E0", "A0|B0|C1|D1|E1"],
        level5=["A0|B0|C0|D0|E0|F0", "A0|B0|C1|D1|E0|F0"],
    )


@pytest.fixture
def encoders() -> list[LabelEncoder]:
    encoders = []
    for letter in string.ascii_uppercase[:6]:
        # Generate sublabels (e.g., A0, A1, A2, A3 for letter 'A')
        sublabels = [f"{letter}{i}" for i in range(4)]
        if letter == "A":
            # Directly create an encoder for the first level using its sublabels.
            encoders.append(LabelEncoder().fit(sublabels))
            continue
        # For levels 'D' and 'E', include an empty string to account for
        # potentially empty
        elif letter in ("D", "E"):
            sublabels.append("")

        # For subsequent levels, create encodings by taking the classes from
        # previous level and calculating the cross product with the current
        # set of sublabels.
        encoders.append(
            LabelEncoder().fit(
                [
                    "|".join(label_array)
                    for label_array in itertools.product(
                        encoders[-1].classes_, sublabels
                    )
                ]
            )
        )

    return encoders


@pytest.fixture()
def softmax_probabilities(
    encoders: list[LabelEncoder], expected_labels_by_level: dict[str, list[str]]
) -> list[npt.NDArray]:
    # For each level in the hierarchy, creates an array where each row
    # is a random probability distribution.
    return [
        np.stack(
            [
                generate_softmax_probabilities(
                    # Assign a probability value for each class in the
                    # corresponding encoder.
                    len(encoder.classes_),
                    # Ensure the encoding for the expected label has the
                    # highest probability value.
                    list(encoder.classes_).index(label),
                )
                for label in expected_labels
            ]
        )
        for encoder, expected_labels in zip(encoders, expected_labels_by_level.values())
    ]


def generate_softmax_probabilities(n_classes, max_index):
    # Generate random values for each class.
    values = np.random.rand(n_classes)
    # Find the maximum value among all other classes (excluding max_index).
    other_max = np.max(np.delete(values, max_index))
    # Set the value at max_index higher than any other value by a random
    # uniform increment.
    values[max_index] = other_max + np.random.uniform(0.1, 1.0)
    # Normalize to create a probability distribution summing to 1.
    distribution = values / np.sum(values)

    return distribution


def test_label_aggregation(
    softmax_probabilities: list[npt.NDArray],
    encoders: list[LabelEncoder],
    expected_labels_by_level: dict[str, list[str]],
):
    # Generate cell identifiers based on the number of samples in the
    # first level of the expected output (e.g., "cell0", "cell1").
    cells = [f"cell{i}" for i in range(len(expected_labels_by_level["level0"]))]

    # Build the cell type hierarchy using the available test fixtures.
    test_hierarchy = CellTypeHierarchy.build(
        softmax_probabilities=softmax_probabilities,
        encoders=encoders,
        cells=cells,
    )

    # Loop over each level to verify that the predicted argmax labels match
    # the expected labels.
    for level, expected_labels in expected_labels_by_level.items():
        assert (test_hierarchy.argmax_labels[level].values == expected_labels).all()

    # ...
    assert (
        test_hierarchy.combined_labels.sel(cell="cell0").values.item()
        == "A0|B0|C0|D0|E0|F0"
    )
    assert (
        test_hierarchy.strict_labels.sel(cell="cell0").values.item()
        == "A0|B0|C0|D0|E0|F0"
    )

    # ...
    assert (
        test_hierarchy.combined_labels.sel(cell="cell1").values.item()
        == "A0|B0|C0|D1|E1|F0"
    )
    assert test_hierarchy.strict_labels.sel(cell="cell1").values.item() == "A0|B0|C0"
