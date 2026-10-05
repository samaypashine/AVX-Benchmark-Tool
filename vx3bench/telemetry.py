import os,sys,threading,time,platform,subprocess,json
import psutil
from pathlib import Path
from .live_publish import publish_json

# PowerShell helpers run hidden and at below-normal priority so the
# occasional query they make never competes with stream capture.
_PS_FLAGS = (0x08000000 | 0x00004000) if os.name == "nt" else 0  # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
LIVE_SAMPLE_LIMIT = 900


def _powershell(command, timeout):
    return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                          capture_output=True, text=True, timeout=timeout, creationflags=_PS_FLAGS)


class _WindowsGpuCounterPoller(threading.Thread):
    """Windows GPU-engine counters for adapters NVML can't see (e.g. an
    Intel iGPU). Get-Counter is expensive -- it starts PowerShell, walks every
    GPU engine instance of every process, and blocks ~1 s to compute a rate --
    so it runs on its own low-priority thread at a slow cadence and the
    sampler only reads the cached result. Previously it ran (together with a
    second PowerShell for adapter names) inside every telemetry sample, which
    kept PowerShell/WMI busy almost continuously for the whole session."""

    def __init__(self, adapter_names, interval, stop_event):
        super().__init__(daemon=True)
        self.adapter_names = adapter_names
        self.interval = max(2.0, float(interval))
        self.stop_event = stop_event
        self.value = None

    def _query(self):
        command = (
            "Get-Counter '\\GPU Engine(*)\\Utilization Percentage' | "
            "Select-Object -ExpandProperty CounterSamples | "
            "Select-Object CookedValue | ConvertTo-Json -Compress"
        )
        result = _powershell(command, 10)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        samples = json.loads(result.stdout)
        if isinstance(samples, dict):
            samples = [samples]
        return min(100.0, sum(float(x.get("CookedValue", 0) or 0) for x in samples))

    def run(self):
        while True:
            try:
                self.value = self._query()
            except Exception:
                self.value = None
            if self.stop_event.wait(self.interval):
                return

    def adapters(self):
        return [{"name": name, "gpu_percent": self.value, "telemetry_backend": "windows-gpu-counters",
                 "integrated": True} for name in self.adapter_names]


_DRM_VENDORS = {"0x1002": "AMD", "0x8086": "Intel", "0x10de": "NVIDIA", "0x5143": "Qualcomm", "0x13b5": "ARM Mali",
                "0x1010": "Imagination", "0x14e4": "Broadcom VideoCore"}


def _read(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def _linux_sysfs_gpus(have_nvml: bool):
    """GPUs NVML can't see on Linux (cheap sysfs reads, no subprocesses):
    AMD (amdgpu reports busy %), Intel and ARM GPUs (listed), and NVIDIA
    Jetson (integrated GPU load), so every platform has a GPU entry."""
    if not sys.platform.startswith("linux"):
        return []
    out = []
    for load_file in ("/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load",
                      "/sys/devices/platform/17000000.ga10b/load", "/sys/devices/platform/17000000.gv11b/load"):
        val = _read(load_file)
        if val is not None and val.isdigit():
            model = (_read("/proc/device-tree/model") or "NVIDIA Jetson").replace("\x00", "").strip("\x00 ")
            out.append({"name": f"{model} GPU", "gpu_percent": int(val) / 10.0, "telemetry_backend": "jetson-sysfs",
                        "integrated": True})
            break
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")) if Path("/sys/class/drm").exists() else []:
        if "-" in card.name:
            continue                                   # connectors like card0-HDMI-A-1
        vendor = _read(card / "device" / "vendor")
        if vendor == "0x10de" and (have_nvml or out):
            continue                                   # NVIDIA: NVML (or Jetson) already reports it
        name = f"{_DRM_VENDORS.get(vendor, 'GPU')} ({card.name})"
        busy = _read(card / "device" / "gpu_busy_percent")
        used, total = _read(card / "device" / "mem_info_vram_used"), _read(card / "device" / "mem_info_vram_total")
        entry = {"name": name, "telemetry_backend": "linux-sysfs",
                 "gpu_percent": float(busy) if busy and busy.isdigit() else None}
        if used and total and used.isdigit() and total.isdigit() and int(total) > 0:
            entry.update(vram_used_bytes=int(used), vram_percent=100 * int(used) / int(total))
        out.append(entry)
    return out


class Sampler(threading.Thread):
    def __init__(self,path,hz,stop,options=None):
        super().__init__(daemon=True)
        self.options = options or {}
        self.path = Path(path)
        self.hz = float(hz)
        self.stop_event = stop
        self.samples = []
        self.warning = ""
        self.live_path = self.path / "live-telemetry.json"
        self.gpu_count = 0
        self._tracked = {}
        self._names = {}
        self.memory_bus_speed_mhz = None
        self.gpu_adapters = []

    def _windows_integrated_adapters(self):
        """Adapter names never change during a run, so they are looked up
        once. Non-NVIDIA adapters are reported through Windows GPU-engine
        counters (NVIDIA GPUs already come from NVML)."""
        if platform.system() != "Windows":
            return []
        try:
            result = _powershell("Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress", 10)
            names = json.loads(result.stdout) if result.stdout.strip() else []
            if not isinstance(names, list):
                names = [names]
            # Every adapter NVML can't see: Intel, AMD, Qualcomm Adreno
            # (Windows on ARM), ... -- NVIDIA GPUs already come from NVML.
            return [str(name) for name in names
                    if name and "nvidia" not in str(name).lower() and "basic display" not in str(name).lower()
                    and "basic render" not in str(name).lower()]
        except Exception:
            return []

    def _memory_bus_speed(self):
        if platform.system() != "Windows":
            return None
        try:
            command = "Get-CimInstance Win32_PhysicalMemory | Select-Object -ExpandProperty ConfiguredClockSpeed | ConvertTo-Json -Compress"
            result = _powershell(command, 10)
            values = json.loads(result.stdout) if result.stdout.strip() else []
            if not isinstance(values, list):
                values = [values]
            values = [float(x) for x in values if x is not None]
            return sum(values) / len(values) if values else None
        except Exception:
            return None

    def _process_snapshot(self, root):
        """Process-tree metrics for this tool, aggregated by executable name.

        psutil.Process objects are kept across samples: cpu_percent(None) is
        a delta since the previous call on the SAME object, so recreating
        them every sample (as before) made every child always report 0 %.
        Rows are also collapsed to one per executable name, which keeps each
        sample -- and the live JSON the UI polls every second -- small even
        with 30+ stream processes."""
        try:
            children = root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
        current = {root.pid: root}
        for child in children:
            known = self._tracked.get(child.pid)
            if known is not None and known == child:
                current[child.pid] = known
                continue
            try:
                child.cpu_percent(None)  # prime; the real value comes next sample
                self._names[child.pid] = child.name()
                current[child.pid] = child
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        for pid in list(self._names):
            if pid not in current:
                self._names.pop(pid, None)
        self._tracked = current
        grouped = {}
        cpu_total = 0.0
        rss_total = 0
        count = 0
        for pid, process in current.items():
            try:
                cpu = process.cpu_percent(None)
                rss = process.memory_info().rss
                name = self._names.get(pid) or process.name()
                self._names[pid] = name
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            count += 1
            cpu_total += cpu
            rss_total += rss
            row = grouped.setdefault(name, {"name": name, "count": 0, "cpu_percent": 0.0, "rss_bytes": 0})
            row["count"] += 1
            row["cpu_percent"] += cpu
            row["rss_bytes"] += rss
        return cpu_total, rss_total, count, list(grouped.values())

    def run(self):
        import psutil
        proc = psutil.Process(os.getpid())
        proc.cpu_percent(None)
        net0 = psutil.net_io_counters()
        disk0 = psutil.disk_io_counters()
        last = time.time()
        started = last
        self.memory_bus_speed_mhz = self._memory_bus_speed()
        self._tracked = {proc.pid: proc}
        self._names = {proc.pid: proc.name()}
        nv = None
        handles=[]
        try:
            import pynvml as nv
            nv.nvmlInit()
            handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
            self.gpu_count = len(handles)
        except Exception as exc:
            self.warning=f"GPU telemetry unavailable: {exc}"
        poller = None
        if platform.system() == "Windows" and self.options.get("windows_gpu_counters", True):
            names = self._windows_integrated_adapters()
            if names:
                poller = _WindowsGpuCounterPoller(names, self.options.get("windows_gpu_counter_interval_seconds", 10), self.stop_event)
                poller.start()

        while not self.stop_event.wait( 1 / max(.1, self.hz)):
            now = time.time()
            dt = max(.001, now-last)
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            mi = proc.memory_info()
            tool_cpu, tool_rss, tool_count, process_details = self._process_snapshot(proc)
            net = psutil.net_io_counters()
            disk = psutil.disk_io_counters()
            du = psutil.disk_usage(str(self.path))
            try:
                freq = psutil.cpu_freq()        # can raise / be None on some ARM boards and VMs
            except Exception:
                freq = None
            temp=None

            try:
                temps = [z.current for group in psutil.sensors_temperatures().values() for z in group if z.current is not None]
                temp = max(temps) if temps else None
            except Exception: pass

            s = {"ts": now,"elapsed_seconds": now-started,"cpu_percent": psutil.cpu_percent(),"cpu_frequency_mhz": freq.current if freq else None,"cpu_temperature_c": temp,"ram_percent": vm.percent,"ram_used_bytes": vm.used,"ram_available_bytes": vm.available,"swap_used_bytes": sw.used,"process_rss_bytes": mi.rss,"process_vms_bytes": mi.vms,"tool_cpu_percent": tool_cpu,"tool_rss_bytes": tool_rss,"tool_process_count": tool_count,"processes": process_details,"network_tx_mbps": max(0,(net.bytes_sent-net0.bytes_sent)*8/dt/1e6),"network_rx_mbps":max(0,(net.bytes_recv-net0.bytes_recv)*8/dt/1e6),"disk_read_mbps":max(0,(disk.read_bytes-disk0.read_bytes)/dt/1e6) if (disk and disk0) else 0,"disk_write_mbps":max(0,(disk.write_bytes-disk0.write_bytes)/dt/1e6) if (disk and disk0) else 0,"disk_total_bytes":du.total,"disk_used_bytes":du.used,"disk_free_bytes":du.free,"disk_percent":du.percent,"gpus":[]}
            if nv:
                for i, h in enumerate(handles):
                    def opt(fn):
                        try: return fn()
                        except Exception:return None
                    # Query each field independently so a failure on any one
                    # metric (or on this GPU specifically) never drops the
                    # whole GPU from the sample — every detected GPU always
                    # gets an entry, even if some fields come back as None.
                    u = opt(lambda: nv.nvmlDeviceGetUtilizationRates(h))
                    m = opt(lambda: nv.nvmlDeviceGetMemoryInfo(h))
                    vram_used = m.used if m is not None else None
                    vram_total = m.total if m is not None else None
                    vram_percent = (100*vram_used/max(1,vram_total)) if (vram_used is not None and vram_total) else None
                    s["gpus"].append({
                        "index": i,
                        "name": (lambda value: value.decode() if isinstance(value, bytes) else str(value))(
                            opt(lambda: nv.nvmlDeviceGetName(h)) or f"GPU {i}"
                        ),
                        "integrated": False,
                        "gpu_percent": u.gpu if u is not None else None,
                        "memory_controller_percent": u.memory if u is not None else None,
                        "vram_used_bytes": vram_used,
                        "vram_total_bytes": vram_total,
                        "vram_percent": vram_percent,
                        "temperature_c": opt(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                        "power_watts": opt(lambda: nv.nvmlDeviceGetPowerUsage(h)/1000),
                        "power_limit_watts": opt(lambda: nv.nvmlDeviceGetEnforcedPowerLimit(h)/1000),
                        "graphics_clock_mhz": opt(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS)),
                        "memory_clock_mhz": opt(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                        "fan_percent": opt(lambda: nv.nvmlDeviceGetFanSpeed(h)),
                        "pcie_tx_mbps": opt(lambda: nv.nvmlDeviceGetPcieThroughput(h, nv.NVML_PCIE_UTIL_TX_BYTES)*1024/1e6),
                        "pcie_rx_mbps": opt(lambda: nv.nvmlDeviceGetPcieThroughput(h, nv.NVML_PCIE_UTIL_RX_BYTES)*1024/1e6),
                        "encoder_percent": opt(lambda: nv.nvmlDeviceGetEncoderUtilization(h)[0]),
                        "decoder_percent": opt(lambda: nv.nvmlDeviceGetDecoderUtilization(h)[0]),
                    })
            for gpu in _linux_sysfs_gpus(nv is not None):
                gpu["index"] = len(s["gpus"])
                s["gpus"].append(gpu)
            windows_gpus = poller.adapters() if poller else []
            known_names = {str(g.get("name", "")).lower() for g in s["gpus"]}
            for gpu in windows_gpus:
                if str(gpu.get("name", "")).lower() not in known_names:
                    gpu["index"] = len(s["gpus"])
                    s["gpus"].append(gpu)
                    known_names.add(str(gpu.get("name", "")).lower())
            s["memory_bus_speed_mhz"] = self.memory_bus_speed_mhz
            gpus = s["gpus"]
            numeric = lambda key: [float(g[key]) for g in gpus if g.get(key) is not None]
            for key, output in (("gpu_percent", "gpu_percent"), ("vram_percent", "vram_percent"), ("temperature_c", "temperature_c")):
                values = numeric(key)
                if values:
                    s.setdefault("gpu_combined", {})[f"{output}_avg"] = sum(values) / len(values)
                    s["gpu_combined"][f"{output}_max"] = max(values)
            powers = numeric("power_watts")
            if powers:
                s.setdefault("gpu_combined", {})["power_watts_total"] = sum(powers)
            s.setdefault("gpu_combined", {})["gpu_count"] = len(gpus)
            self.samples.append(s)
            warn = publish_json(self.live_path, {"warning":self.warning,"sample_count":len(self.samples),"samples":self.samples[-LIVE_SAMPLE_LIMIT:]});self.warning=self.warning or (warn or "")
            net0, disk0, last = net, disk, now

    def summary(self):
        if not self.samples:
            return {}
        def stat(path):
            v=[]
            for s in self.samples:
                try:
                    x=s
                    for k in path:x=x[k]
                    if x is not None:v.append(float(x))
                except Exception:pass
            return {"min":min(v),"mean":sum(v)/len(v),"max":max(v),"start":v[0],"end":v[-1]} if v else {}
        keys=["cpu_percent","tool_cpu_percent","cpu_frequency_mhz","cpu_temperature_c","ram_percent","ram_used_bytes","memory_bus_speed_mhz","tool_rss_bytes","process_rss_bytes","process_vms_bytes","swap_used_bytes","network_tx_mbps","network_rx_mbps","disk_read_mbps","disk_write_mbps","disk_free_bytes","disk_percent"]
        out={k:stat([k]) for k in keys};out["process_rss_growth_bytes"]=self.samples[-1]["process_rss_bytes"]-self.samples[0]["process_rss_bytes"];out["disk_consumed_bytes"]=max(0,self.samples[0]["disk_free_bytes"]-self.samples[-1]["disk_free_bytes"]);out["gpus"]=[]
        gpu_count = max(self.gpu_count, max((len(s.get("gpus", [])) for s in self.samples), default=0))
        for i in range(gpu_count):
            item = {k:stat(["gpus",i,k]) for k in ["gpu_percent","memory_controller_percent","vram_used_bytes","vram_percent","temperature_c","power_watts","graphics_clock_mhz","memory_clock_mhz","fan_percent","pcie_tx_mbps","pcie_rx_mbps","encoder_percent","decoder_percent"]}
            item["name"] = next((g.get("name") for s in self.samples for g in s.get("gpus",[]) if g.get("index", i) == i and g.get("name")), f"GPU {i}")
            out["gpus"].append(item)
        return out
