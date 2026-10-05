import platform, socket, subprocess, sys, json
from pathlib import Path


def _disks():
    import psutil
    disks = []
    try:
        partitions = psutil.disk_partitions(all=False)
    except Exception:
        partitions = []
    for part in partitions:
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except Exception:
            continue
        disks.append({
            "device": part.device,
            "mountpoint": part.mountpoint,
            "filesystem": part.fstype,
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "percent": usage.percent,
        })
    return disks


def _usb_devices():
    """Best-effort USB device enumeration. Platform tooling varies, so this
    never raises: on failure it returns an empty list plus a warning that is
    surfaced the same way the GPU inventory warning is."""
    system = platform.system()
    devices = []
    try:
        if system == "Linux":
            out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    line = line.strip()
                    if line:
                        devices.append({"description": line})
            elif out.stderr:
                return devices, out.stderr.strip()
        elif system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-PnpDevice -PresentOnly -Class USB | Select-Object -ExpandProperty FriendlyName"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    line = line.strip()
                    if line:
                        devices.append({"description": line})
            elif out.stderr:
                return devices, out.stderr.strip()
        elif system == "Darwin":
            out = subprocess.run(["system_profiler", "SPUSBDataType"], capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    stripped = line.strip()
                    # Device names are the non-indented-field lines ending in ':'
                    # with no nested "Key: value" pair on them.
                    if stripped.endswith(":") and ":" not in stripped[:-1] and len(line) - len(line.lstrip(" ")) <= 8:
                        name = stripped[:-1]
                        if name and name not in ("USB", "Hub"):
                            devices.append({"description": name})
            elif out.stderr:
                return devices, out.stderr.strip()
        else:
            return devices, f"USB enumeration not supported on platform '{system}'"
    except FileNotFoundError as exc:
        return devices, f"USB enumeration tool not available: {exc}"
    except Exception as exc:
        return devices, str(exc)
    return devices, None


def _windows_video_adapters():
    if platform.system() != "Windows":
        return [], None
    try:
        command = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return [], result.stderr.strip() or "Windows video adapter query returned no data"
        value = json.loads(result.stdout)
        if isinstance(value, dict):
            value = [value]
        return [
            {
                "index": index,
                "name": item.get("Name") or "Unknown",
                "vram_total_bytes": item.get("AdapterRAM"),
                "driver_version": item.get("DriverVersion"),
                "integrated": "Intel" in (item.get("Name") or ""),
                "telemetry_backend": "windows-gpu-counters",
            }
            for index, item in enumerate(value)
        ], None
    except Exception as exc:
        return [], str(exc)


def _memory_bus_speed():
    if platform.system() != "Windows":
        return None
    try:
        command = "Get-CimInstance Win32_PhysicalMemory | Select-Object -ExpandProperty ConfiguredClockSpeed | ConvertTo-Json -Compress"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        values = json.loads(result.stdout)
        if not isinstance(values, list):
            values = [values]
        values = [float(value) for value in values if value is not None]
        return sum(values) / len(values) if values else None
    except Exception:
        return None


def _cpu_model() -> str:
    """Human-readable CPU name on every platform. platform.processor() is
    empty or just 'x86_64'/'aarch64' on Linux and ARM, so read the real one."""
    system = platform.system()
    try:
        if system == "Linux":
            info = Path("/proc/cpuinfo").read_text(errors="replace")
            for key in ("model name", "Hardware", "Processor", "cpu model"):
                for line in info.splitlines():
                    if line.lower().startswith(key.lower()) and ":" in line:
                        value = line.split(":", 1)[1].strip()
                        if value and not value.isdigit():
                            return value
            board = Path("/proc/device-tree/model")
            if board.exists():                        # Raspberry Pi, NVIDIA Jetson, other ARM boards
                return board.read_text(errors="replace").strip("\x00 \n") + f" ({platform.machine()})"
            out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                if line.lower().startswith("model name:"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
            if out.stdout.strip():
                return out.stdout.strip()
        elif system == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _linux_gpus() -> list[dict]:
    """Non-NVIDIA GPUs on Linux (AMD, Intel, ARM, Jetson) from sysfs."""
    try:
        from .telemetry import _linux_sysfs_gpus
        return [{"name": g["name"], "vram_total_bytes": None, "driver_version": None} for g in _linux_sysfs_gpus(False)
                if "NVIDIA (" not in g["name"]]
    except Exception:
        return []


def collect():
    import psutil
    data = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_model": _cpu_model(),
        "logical_cpu_count": psutil.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "memory_bus_speed_mhz": _memory_bus_speed(),
        "gpus": [],
        "disks": _disks(),
        "usb_devices": [],
    }
    try:
        import pynvml as nv
        nv.nvmlInit()
        gpu_warnings = []
        for i in range(nv.nvmlDeviceGetCount()):
            # Enumerate each GPU independently: a query failure on one card
            # (driver quirk, permissions, unsupported field on that model)
            # must not abort enumeration of the remaining GPUs.
            try:
                h = nv.nvmlDeviceGetHandleByIndex(i)
                name = nv.nvmlDeviceGetName(h)
                try:
                    mem = nv.nvmlDeviceGetMemoryInfo(h)
                    vram_total = mem.total
                except Exception:
                    vram_total = None
                try:
                    driver_version = str(nv.nvmlSystemGetDriverVersion())
                except Exception:
                    driver_version = None
                data["gpus"].append({
                    "index": i,
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "vram_total_bytes": vram_total,
                    "driver_version": driver_version,
                })
            except Exception as exc:
                gpu_warnings.append(f"GPU {i}: {exc}")
        if gpu_warnings:
            data["gpu_inventory_warning"] = "; ".join(gpu_warnings)
    except Exception as exc:
        data["gpu_inventory_warning"] = str(exc)
    if platform.system() == "Windows":
        adapters, warning = _windows_video_adapters()
        known = {str(g.get("name", "")).lower() for g in data["gpus"]}
        for adapter in adapters:
            if adapter["name"].lower() not in known:
                data["gpus"].append(adapter)
        if warning:
            data["gpu_inventory_warning"] = (data.get("gpu_inventory_warning", "") + "; " + warning).strip("; ")

    if platform.system() == "Linux":
        known = {str(g.get("name", "")).lower() for g in data["gpus"]}
        for g in _linux_gpus():
            if g["name"].lower() not in known:
                g["index"] = len(data["gpus"])
                data["gpus"].append(g)
    usb_devices, usb_warning = _usb_devices()
    data["usb_devices"] = usb_devices
    if usb_warning:
        data["usb_inventory_warning"] = usb_warning
    return data
