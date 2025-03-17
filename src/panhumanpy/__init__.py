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

__version__ = '0.1.0'
__author__ = 'SatijaLab'