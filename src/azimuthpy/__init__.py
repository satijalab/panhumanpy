import azimuthpy.panhuman as panhuman
from azimuthpy.azimuth import Azimuth
from azimuthpy.cell_type_hierarchy import CellTypeHierarchy

__all__ = ["Azimuth", "CellTypeHierarchy", "panhuman", "panhuman_azimuth"]


def panhuman_azimuth() -> Azimuth[CellTypeHierarchy, panhuman.PanHumanAnnotator]:
    """The default Azimuth panhuman workflow."""

    return Azimuth(
        model=panhuman.default_model().load(),
        annotation_type=CellTypeHierarchy,
        output=panhuman.PanHumanAnnotator(label_resolution="medium"),
    )
