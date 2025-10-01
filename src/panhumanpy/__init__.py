"""
panhumanpy - A package for cell annotation using Azimuth Neural Network.
"""

from .ANNotate import AzimuthNN, AzimuthNN_base, annotate_core
from .ANNotate_tools import configure

__all__ = [
    'AzimuthNN',
    'AzimuthNN_base',
    'annotate_core',
    'configure'
]

__version__ = "0.2.3"
__version_name__ = "Andromeda"
__version_full__ = f'{__version__} ({__version_name__})'
__author__ = 'SatijaLab'


def get_version():
    """Get version information"""
    return {
        'version': __version__,
        'name': __version_name__, 
        'full': __version_full__
    }