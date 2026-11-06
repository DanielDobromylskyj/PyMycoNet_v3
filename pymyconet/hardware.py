import importlib.util
import platform
import subprocess


def get_system_info():
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count()
    }


def check_opencl():
    result = {
        "installed": importlib.util.find_spec("pyopencl") is not None,
        "available": False,
        "platforms": []
    }

    if not result["installed"]:
        return result

    try:
        import pyopencl as cl

        for p in cl.get_platforms():
            platform_info = {
                "name": p.name,
                "version": p.version,
                "vendor": p.vendor,
                "devices": []
            }

            for d in p.get_devices():
                platform_info["devices"].append({
                    "name": d.name,
                    "type": cl.device_type.to_string(d.type),
                    "vendor": d.vendor,
                    "version": d.version,
                    "opencl_c_version": d.opencl_c_version,
                    "global_memory": d.global_mem_size,
                    "max_work_group_size": d.max_work_group_size
                })

            result["platforms"].append(platform_info)

        result["available"] = len(result["platforms"]) > 0

    except Exception as e:
        result["error"] = str(e)

    return result


def check_cuda():
    result = {
        "installed": importlib.util.find_spec("torch") is not None,
        "available": False,
        "devices": []
    }

    if result["installed"]:
        try:
            import torch

            result["available"] = torch.cuda.is_available()
            result["cuda_version"] = torch.version.cuda
            result["rocm_version"] = torch.version.hip

            if result["available"]:
                for i in range(torch.cuda.device_count()):
                    result["devices"].append({
                        "name": torch.cuda.get_device_name(i),
                        "memory": torch.cuda.get_device_properties(i).total_memory,
                        "compute_capability": (
                            torch.cuda.get_device_capability(i)
                        )
                    })

        except Exception as e:
            result["error"] = str(e)

    # Check for NVIDIA driver independently
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if output.returncode == 0:
            result["nvidia_gpus"] = output.stdout.strip().splitlines()

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return result


def check_tensorflow():
    result = {
        "installed": importlib.util.find_spec("tensorflow") is not None,
        "available": False,
        "gpus": []
    }

    if not result["installed"]:
        return result

    try:
        import tensorflow as tf

        result["version"] = tf.__version__
        result["available"] = True

        result["gpus"] = [
            {
                "name": device.name,
                "type": device.device_type
            }
            for device in tf.config.list_physical_devices("GPU")
        ]

    except Exception as e:
        result["error"] = str(e)

    return result


def detect_capabilities():
    return {
        "system": get_system_info(),
        "opencl": check_opencl(),
        "cuda": check_cuda(),
        "tensorflow": check_tensorflow()
    }


def get_hardware_options() -> dict:
    return {
        "cpu": {
            "default": "np",
            "options": {
                "np": {"requires": ["numpy"], "supported": True, "path": "cpu_np"},
                "test": {"requires": ["pandas", "matplotlib"], "supported": False, "path": ""},
            }
        },

        "gpu": {
            "default": "cl",
            "options": {
                "cl": {"requires": ["pyopencl", "numpy"], "supported": False, "path": "", "check": check_opencl},
                "tensorflow": {"requires": ["tensorflow", "numpy"], "supported": False, "path": "", "check": check_tensorflow},
                "pytorch": {"requires": ["pytorch", "numpy"], "supported": False, "path": ""},
                "cuda": {"requires": ["cuda-python", "numpy"], "supported": False, "path": "", "check": check_cuda},
            }
        }
    }

def _is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None

def _is_hardware_option_available(config: dict) -> dict:
    if config["supported"] is False:
        return {
            "available": False,
            "error": f"Not Supported by PyMycoNet"
        }

    for requirement in config["requires"]:
        if not _is_installed(requirement):
            return {
                "available": False,
                "error": f"Missing Package: {requirement}"
            }

    if "check" in config:
        if not config["check"]():
            return {
                "available": False,
                "error": f"Failed to find required drivers/hardware/software"
            }

    return {
        "available": True
    }

def _infer_hardware_option(compute_mode: str) -> dict | None:
    hardware_options = get_hardware_options()

    if "-" not in compute_mode:  # find first matching specific for generic
        if compute_mode in hardware_options:
            default = hardware_options[compute_mode]["default"]
            config = hardware_options[compute_mode]["options"][default]

            if _is_hardware_option_available(config)["available"]:
                return {
                    "mode": compute_mode + "-" + default,
                    "config": config
                }

            else:
                for option in hardware_options[compute_mode]["options"]:
                    config = hardware_options[compute_mode]["options"][option]

                    if _is_hardware_option_available(config)["available"]:
                        return {
                            "mode": compute_mode + "-" + option,
                            "config": config
                        }

                return None
        else:
            return None

    else:
        if compute_mode.count("-") != 1:
            return None

        generic, specific = compute_mode.split("-")

        if generic not in hardware_options:
            return None

        options = hardware_options[generic]["options"]

        if specific not in options:
            return None

        return {
            "mode": compute_mode,
            "config": options[specific]
        }

def get_hardware_option(option: str) -> str:
    data = _infer_hardware_option(option)

    if data is None:
        raise ModuleNotFoundError(f"Could not find support for '{option}' compute mode")

    result = _is_hardware_option_available(data["config"])

    if not result["available"]:
        raise ImportError(f"Cannot use '{option}': {result['error']}")

    return data["mode"]


def get_core_path(option: str) -> str:
    data = _infer_hardware_option(option)

    if data is None:
        raise ModuleNotFoundError(f"Could not find support for '{option}' compute mode")

    return f"core.{data["config"]["path"]}"


def load_core_segment(location: str, class_name: str):
    if not importlib.util.find_spec(location):
        raise ModuleNotFoundError(f"Current hardware mode does not support '{class_name}' ({location})")

    module = importlib.import_module(location)

    cls = getattr(module, class_name)
    return cls