from importlib.resources import as_file, files

import azimuthpy.panhuman.models.default
from azimuthpy.panhuman.model_loader import PanHumanModelLoader
from azimuthpy.panhuman.postprocessing import PanHumanAnnotator

__all__ = [
    "PanHumanModelLoader",
    "PanHumanAnnotator",
    "available_models",
    "default_model",
    "get_model",
]


def default_model() -> PanHumanModelLoader:
    """Return the default panhuman model."""

    with as_file(files(azimuthpy.panhuman.models.default)) as fp:
        return PanHumanModelLoader(
            model_name="M0.2",
            model_root=fp,
            embedding_layer="dense_3",
        )


def available_models() -> list[PanHumanModelLoader]:
    """List all available panhuman models."""
    return [default_model()]


def get_model(name: str) -> PanHumanModelLoader:
    """Retrieve a panhuman model by name."""
    models_by_name = {loader.model_name: loader for loader in available_models()}

    if name not in models_by_name:
        raise ValueError

    return models_by_name[name]
