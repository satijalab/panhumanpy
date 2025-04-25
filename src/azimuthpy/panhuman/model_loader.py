from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import as_file
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import (  # type: ignore [reportMissingImports]
    Model,
    load_model,
)

from azimuthpy.model_runners import KerasModelRunner

__all__ = ["PanHumanModelLoader"]


@dataclass
class PanHumanModelLoader:
    """Loader class for PanHuman models."""

    model_name: str
    model_root: Path

    embedding_layer: str

    @property
    def path_to_model(self) -> Path:
        """Return the file path to the Keras model file."""
        return self.model_root / "model.keras"

    @property
    def path_to_feature_panel(self) -> Path:
        """Return the file path to the feature panel text file."""
        return self.model_root / "feature_panel.txt"

    @property
    def pathlist_to_encoder_classes(self) -> list[Path]:
        """Return a sorted list of paths pointing to text files containing
        the class names for each level's encoded output.
        """

        def path_key(path: Path) -> str:
            match = re.search(r"encoder_classes_level(\d+)\.txt", path.name)
            if match:
                return match.group(1)

            return ""

        return sorted(
            self.model_root.glob("encoder_classes*.txt"),
            key=path_key,
        )

    def load_keras_model(self) -> Model:
        """Load the Keras model."""

        with as_file(self.path_to_model) as file:
            model = load_model(file)

        return model

    def load_feature_panel(self) -> list[str]:
        """Load the feature panel."""

        with (self.path_to_feature_panel).open() as file:
            feature_panel = file.read().splitlines()

        return feature_panel

    def load_encoder_classes(self) -> list[list[str]]:
        """Load the output labels."""

        return [
            path_to_lables.open().read().splitlines()
            for path_to_lables in self.pathlist_to_encoder_classes
        ]

    def load(self) -> KerasModelRunner:
        """Load the underlying files and construct a PanHumanModel using their
        contents.
        """

        annotation_model = self.load_keras_model()
        embedding_model = Model(
            inputs=annotation_model.input,
            outputs=annotation_model.get_layer(self.embedding_layer).output,
        )

        encoders = [
            LabelEncoder().fit(labels) for labels in self.load_encoder_classes()
        ]

        return KerasModelRunner(
            model_name=self.model_name,
            embedding_model=embedding_model,
            annotation_model=annotation_model,
            feature_panel=self.load_feature_panel(),
            encoders=encoders,
        )
