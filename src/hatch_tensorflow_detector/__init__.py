"""
Hatch build hook for automatic TensorFlow GPU detection.

This module provides a custom build hook that dynamically selects the appropriate 
TensorFlow installation based on GPU availability during package installation.

Key Features:
- Automatically detects NVIDIA GPU using nvidia-smi
- Installs TensorFlow with CUDA support when GPU is available
- Falls back to CPU-only TensorFlow when no GPU is detected
- Seamlessly integrates with pip and hatch installation processes

Usage:
    # Auto-detect and install appropriate TensorFlow version
    pip install .

    # Force CPU or GPU installation
    pip install .[cpu]
    pip install .[gpu]

Detection Logic:
1. Runs nvidia-smi to check for GPU presence
2. If GPU detected, installs TensorFlow with CUDA support
3. If no GPU or nvidia-smi not found, installs CPU-only TensorFlow

Requirements:
- hatchling
- nvidia-smi (for GPU detection, optional)
"""

import subprocess
from hatchling.buildhook.plugin.interface import BuildHookInterface

class TensorFlowDetectorHook(BuildHookInterface):
    def initialize(self, version, build_data):
        """
        Detect GPU and modify dependencies accordingly.

        Args:
            version (str): Package version
            build_data (dict): Build configuration data

        Side Effects:
            Appends appropriate TensorFlow dependency to build_data
            Prints detection status to console
        """
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(
                ['nvidia-smi'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            if result.returncode == 0:
                # GPU detected
                build_data['dependencies'].append('tensorflow[and-cuda]==2.17.0')
                print('GPU detected. Installing TensorFlow with CUDA support.')
            else:
                # No GPU
                build_data['dependencies'].append('tensorflow-cpu==2.17.0')
                print('No GPU detected. Installing CPU-only TensorFlow.')
        except FileNotFoundError:
            # nvidia-smi not found
            build_data['dependencies'].append('tensorflow-cpu==2.17.0')
            print('nvidia-smi not found. Installing CPU-only TensorFlow.')

def plugin():
    """
    Provide the TensorFlow detector hook for Hatch.

    Returns:
        TensorFlowDetectorHook: The build hook for TensorFlow detection
    """
    return TensorFlowDetectorHook