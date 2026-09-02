"""
System and hardware environment detection utility.
Captures comprehensive hardware and software telemetry for reproducibility.
"""

import platform
import os
import sys
from typing import Dict, Any
import numpy as np
import torch


def collect_system_info() -> Dict[str, Any]:
    """
    Collect exhaustive system, hardware, and environment telemetry.

    Returns:
        Dictionary containing OS, CPU, RAM, GPU, and Python framework metadata.
    """
    info: Dict[str, Any] = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "frameworks": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "hardware": {
            "cpu_count_logical": os.cpu_count(),
        },
    }

    # Detect CUDA / GPU details
    if torch.cuda.is_available():
        info["hardware"]["gpu_count"] = torch.cuda.device_count()
        info["hardware"]["gpu_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["hardware"]["gpu_devices"].append(
                {
                    "device_index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "multi_processor_count": props.multi_processor_count,
                    "major_capability": props.major,
                    "minor_capability": props.minor,
                }
            )
        info["frameworks"]["cuda_version"] = torch.version.cuda
        info["frameworks"]["cudnn_version"] = torch.backends.cudnn.version()
    else:
        info["hardware"]["gpu_count"] = 0
        info["hardware"]["gpu_devices"] = []

    return info
