# VX3 Benchmark

A hardware-qualification benchmark for VX3 video-analytics machines. It loads a machine the way a real deployment does, all at the same time:

- **Many live NDI camera streams**, received and composited by **OBS Studio**
- **AI models** (ONNX) that must sustain a target frame rate *per stream*
- **Live video encoding** (H.264, GPU or CPU)
- **Image persistence** (saving one image per AI inference to disk)
- **Image capture** (periodic snapshot polling)

While all of that runs, it measures every stream's frame rate, every model's throughput and latency, and the machine itself (CPU, RAM, GPU, VRAM, temperatures, disk, network). It then writes an interactive **HTML report** and a **PDF report**, each with a bottleneck analysis.

Everything is configured and monitored from a **web interface**. No code changes are needed to change stream counts, models, tests or duration.

---

## Contents

1. [How it works](#1-how-it-works)
2. [Requirements](#2-requirements)
3. [Installation (benchmark PC)](#3-installation-benchmark-pc)
4. [Starting the tool](#4-starting-the-tool)
5. [Using the web interface](#5-using-the-web-interface)
6. [Common workflows](#6-common-workflows)
7. [NDI camera emulator (second PC)](#7-ndi-camera-emulator-second-pc)
8. [Reading the results](#8-reading-the-results)
9. [Configuration reference](#9-configuration-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [Project layout](#11-project-layout)
12. [Known limitations](#12-known-limitations)

---

## 1. How it works

```
 Camera network ──NDI──►  OBS Studio (launched by the benchmark)
                          scene "VX3 NDI": one DistroAV NDI Source per stream,
                          + synthetic streams (SpeedHQ clips), tiled as a multiview
                          └─ Lua script reads OBS's Source Profiler ──► per-stream FPS
                                         ▲ obs-websocket (scene/sources setup, stats)
 Web UI (browser) ◄──► VX3 web server ──► session engine
                                            ├─ AI model workers (ONNX Runtime, auto-scaling)
                                            ├─ Image persistence (one saver per model)
                                            ├─ Encode test (N × FFmpeg H.264 encoders)
                                            ├─ Image capture (snapshot polling)
                                            └─ Telemetry sampler ──► live UI + reports
```

**Streams are received by OBS, not by the benchmark.** When you ask for *N* NDI streams, the benchmark:
1. launches OBS,
2. lets OBS's DistroAV plugin discover the NDI sources on the network,
3. creates *N* NDI Sources in a scene, each assigned a **different** source,
4. reads each source's frame rate from OBS's **Source Profiler**, the same numbers as *View → Source Profiler* in OBS.

If fewer real sources exist than requested, the shortfall can be filled with **synthetic streams**. These play a looping clip in the SpeedHQ codec (the codec full-bandwidth NDI uses) inside the same OBS scene. For realistic network load, see the [camera emulator](#7-ndi-camera-emulator-second-pc).

**AI throughput target.** Each enabled model must sustain `target FPS per stream × number of streams` (default 15 FPS per stream). The benchmark finds how many worker instances are needed. It adds them one at a time and stops when:
- the target is met,
- adding another raises throughput by less than the minimum gain (the model is marked **Saturated**; an instance that *lowers* throughput is closed again),
- or the maximum instance count is reached.

---

## 2. Requirements

### 2.1 Benchmark PC (the machine under test)

| Item | Requirement | Notes |
|---|---|---|
| OS / CPU | **Windows 10/11** (x64 or ARM64), **Linux** (x86-64 or aarch64, e.g. Ubuntu 22.04/24.04, NVIDIA Jetson, Raspberry Pi 5) or **macOS** (Apple Silicon) | OBS is a desktop application: on Linux it needs a graphical display (see [3.6](#36-linux-notes)). Platform specifics: [3.7](#37-platform-notes-windows-linux-arm-macos). |
| Python | **3.10 or newer**, 64-bit (3.12 / 3.13 recommended) | A virtual environment or conda env is recommended. |
| OBS Studio | **30.1 or newer** | 30.1 introduced the Source Profiler used for stream FPS. obs-websocket (built in since OBS 28) is used for automation. |
| DistroAV | Current release for your OBS version | The OBS plugin that provides the "NDI Source" input (formerly obs-ndi). |
| NDI Runtime | NDI 6 | Installed together with **NDI Tools**, or on its own. Required by DistroAV. |
| GPU stack *(optional, for GPU inference)* | NVIDIA: driver + **CUDA 12.x** + **cuDNN 9.x** (+ optional **TensorRT**). Other GPUs: DirectML (Windows), ROCm (Linux AMD), CoreML (macOS) | Not required: if the selected GPU is missing or unusable, models **fall back automatically** (ending on the CPU) and the report says so. |
| FFmpeg *(for the encode test and synthetic streams)* | Any recent build | `imageio-ffmpeg` (installed by requirements) bundles one. **For GPU encoding (NVENC / Quick Sync / AMF) install a *full* FFmpeg build**, because the bundled build may not include hardware encoders. See [3.5](#35-ffmpeg-with-hardware-encoders-recommended). |
| Network | Enough bandwidth for your streams | One full-bandwidth 1080p30 NDI stream is about 100–125 Mbit/s, so **12 streams ≈ 1.2–1.5 Gbit/s, more than a 1 GbE link can carry.** |

### 2.2 Python packages

Install from `requirements.txt` (benchmark PC). Summary:

| Package | Purpose | Required? |
|---|---|---|
| `numpy`, `psutil` | Core buffers, telemetry | Yes |
| `onnxruntime-gpu` **or** `onnxruntime` | AI inference (GPU or CPU) | Yes. `requirements.txt` picks the right one per platform automatically. |
| `reportlab`, `pypdf` | PDF report | Yes (without them only the HTML report is written) |
| `imageio-ffmpeg` | Bundled FFmpeg | Yes, unless FFmpeg is on PATH |
| `opencv-python`, `Pillow` | Image encoding for persistence | Yes (Pillow is the fallback) |
| `nvidia-ml-py` | NVIDIA GPU telemetry | Optional |

### 2.3 Second PC (only for the camera emulator)

Python 3.10+, the **NDI Runtime**, and `requirements-emulator.txt` (`numpy`, `cyndilib`, `psutil`). It must be on the **same network/switch** as your cameras.

---

## 3. Installation (benchmark PC)

### 3.1 Get the code

Place the project anywhere, for example `C:\VX3\Benchmark-Tool`. The folder should contain `vx3bench\`, `README.md` and `requirements.txt` (see [Project layout](#11-project-layout)).

### 3.2 Create a Python environment and install packages

Using **conda**:
```bat
conda create -n bench python=3.12 -y
conda activate bench
cd C:\VX3\Benchmark-Tool
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Using **venv**:
```bat
cd C:\VX3\Benchmark-Tool
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> `requirements.txt` installs **`onnxruntime-gpu` on x86-64 Windows/Linux** (it also runs on the CPU) and **`onnxruntime` on ARM64 and macOS**. Never install both packages in one environment. For NVIDIA Jetson or a non-NVIDIA Windows GPU, see [3.7](#37-platform-notes-windows-linux-arm-macos).

Check the AI runtime sees your GPU:
```bat
python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```
For GPU inference the list must include `CUDAExecutionProvider` (and `TensorrtExecutionProvider` if TensorRT is installed). If it only shows `CPUExecutionProvider`, fix the CUDA/cuDNN installation first; see [Troubleshooting](#10-troubleshooting).

### 3.3 Install OBS Studio, DistroAV and the NDI Runtime

1. Install **OBS Studio 30.1+** from obsproject.com. The default path `C:\Program Files\obs-studio\` is detected automatically.
2. Install the **NDI Runtime** (NDI 6), for example via **NDI Tools** from ndi.video.
3. Install **DistroAV** for your OBS version (github.com/DistroAV/DistroAV).
4. Start OBS once by hand to confirm it opens and that **Sources → + → NDI Source** exists. Then **close OBS**.

You do **not** need to configure obs-websocket. For each session the benchmark:
- enables the websocket server if it's off (your original settings are restored afterwards),
- uses a one-time password,
- uses its own scene collection, **"VX3 Benchmark"**, so your other collections are never modified.

### 3.4 Verify the setup

```bat
python -c "import numpy, psutil, onnxruntime, reportlab, pypdf, imageio_ffmpeg; print('Python packages OK')"
```

### 3.5 FFmpeg with hardware encoders (recommended)

The encode test prefers a **GPU encoder**. At the start of each encode test the benchmark checks every FFmpeg it can find for a *working* NVIDIA NVENC, Intel Quick Sync or AMD AMF H.264 encoder, in this order:
1. the `VX3_FFMPEG` environment variable,
2. `ffmpeg` on PATH,
3. the bundled `imageio-ffmpeg` build.

If none works, it encodes on the **CPU** (libx264). To make sure GPU encoders are available:
1. Download a **full** Windows FFmpeg build (e.g. the "full" build from gyan.dev) and extract it, e.g. to `C:\ffmpeg`.
2. Either add `C:\ffmpeg\bin` to PATH, or set:
   ```bat
   setx VX3_FFMPEG "C:\ffmpeg\bin\ffmpeg.exe"
   ```
   (Open a new terminal afterwards.)

The report's *GPU encoder detection* section shows exactly what was found and why each encoder was or wasn't usable.

### 3.6 Linux notes

- **Install OBS** from your distribution, the official PPA (`sudo add-apt-repository ppa:obsproject/obs-studio && sudo apt install obs-studio`) or Flathub (`flatpak install flathub com.obsproject.Studio`). Both native and Flatpak installs are detected automatically, including the Flatpak config folder `~/.var/app/com.obsproject.Studio/config/obs-studio`.
- **Install DistroAV and the NDI Runtime** for Linux (see the DistroAV installation guide; the Flatpak has its own DistroAV extension).
- **A display is required.** Start the benchmark from a terminal **inside the desktop session**, not over SSH or as a system service. For a headless machine, run it under a virtual display:
  ```bash
  sudo apt install xvfb
  xvfb-run -a python -m vx3bench --host 0.0.0.0
  ```
  (A virtual display has no GPU acceleration for OBS's rendering, so results differ from a real desktop session.)
- If OBS fails to start, the error shown in the UI includes OBS's own messages, the path of OBS's log, and `obs-output.log` in the session folder.

### 3.7 Platform notes (Windows, Linux, ARM, macOS)

The tool behaves the same on every platform; what differs is which accelerators exist. Everything optional **degrades gracefully**: a missing GPU, encoder or sensor is reported in the UI and report, never a crash.

| Platform | AI acceleration (ONNX Runtime) | Encode test (hardware H.264) | GPU telemetry | Notes |
|---|---|---|---|---|
| Windows x64 + NVIDIA | CUDA / TensorRT (`onnxruntime-gpu`) | NVENC | NVML (full) | Primary platform. |
| Windows x64 + AMD/Intel GPU | DirectML (`onnxruntime-directml`) | AMF / Quick Sync / Media Foundation | Windows GPU counters | Replace `onnxruntime-gpu` with `onnxruntime-directml`. |
| Windows on ARM (Snapdragon) | CPU (`onnxruntime`); DirectML if `onnxruntime-directml` is available for ARM64 | Media Foundation (hardware) | Windows GPU counters | Install FFmpeg yourself (no bundled build for ARM64); OpenCV is skipped and Pillow is used. Use a native ARM64 Python. |
| Linux x86-64 + NVIDIA | CUDA / TensorRT (`onnxruntime-gpu`) | NVENC | NVML (full) | Desktop session required for OBS. |
| Linux x86-64 + AMD / Intel | ROCm (AMD, ROCm build of ONNX Runtime) or CPU | VA-API (`/dev/dri/renderD128`) | AMD: busy % and VRAM via sysfs; Intel: listed | |
| NVIDIA Jetson (aarch64) | CUDA / TensorRT with **NVIDIA's Jetson `onnxruntime-gpu` wheel** for your JetPack | depends on the FFmpeg build (Jetson's hardware encoder is not in standard FFmpeg; CPU otherwise) | Jetson GPU load via sysfs | After `pip install -r requirements.txt`, run `pip uninstall onnxruntime` and install NVIDIA's wheel. |
| Raspberry Pi / other ARM Linux | CPU | V4L2 M2M where supported | listed if the GPU appears in `/sys/class/drm` | Expect CPU-bound results. |
| macOS (Apple Silicon) | CoreML or CPU (`onnxruntime`) | VideoToolbox | not available | |

**AI device selection and fallback.** Choose *Auto*, *GPU – CUDA*, *GPU – TensorRT*, *GPU – DirectML*, *CoreML* or *CPU* per model. The chosen device is tried first, then other GPU providers, then the CPU:
- TensorRT → CUDA → ROCm → DirectML → CoreML → CPU
- CUDA → ROCm → DirectML → CoreML → CPU
- Auto → CUDA → ROCm → DirectML → CoreML → CPU

A provider is only used if a session **really runs on it**. ONNX Runtime may list CUDA/TensorRT without the GPU libraries being present; that case is detected and skipped. Any step down is shown as **Device fallback** on the model card, in both reports, and as an *AI Device* finding in the bottleneck analysis. Example: *GPU – TensorRT requested → running on CPU (no GPU provider is installed in this ONNX Runtime)*.

**Encode test.** Hardware encoders are probed with a real 10-frame encode, in this order: NVENC, Quick Sync, AMF, VideoToolbox, VA-API, V4L2 M2M, Media Foundation. The first that works is used, otherwise the CPU (libx264). The report lists every encoder checked and why it was or wasn't usable.

---

## 4. Starting the tool

Close OBS if it is running, then from the project folder (with your environment activated):

```bat
python -m vx3bench
```

The web interface opens automatically at **http://localhost:8420**.

| Option | Default | Meaning |
|---|---|---|
| `--port` | `8420` | Web interface port |
| `--host` | `127.0.0.1` | Listen address. Use `0.0.0.0` to open the UI from another PC on the network. |
| `--config-dir` | `configs` | Where scenarios are stored (created if missing) |
| `--reports-dir` | `reports` | Where session results are written (created if missing) |
| `--no-browser` | off | Don't open a browser automatically |

Example: `python -m vx3bench --host 0.0.0.0 --port 9000 --no-browser`

> Run the command **from the project root** (the folder that contains `vx3bench\`). Relative paths such as `configs`, `reports` and uploaded models are resolved from there.

---

## 5. Using the web interface

The left side edits the **scenario** (a saved configuration). The right side shows **live** results during a session. Scenarios are stored as JSON in `configs\`, and you can edit them in the **Parametric View** (form) or the **JSON Editor**. The **System Information** panel shows the detected hardware (CPU, RAM, GPUs, disks).

### 5.1 Scenarios
- **Create / select / save** scenarios at the top. Each has a name, a description and a **session duration** (value + unit: seconds, minutes, hours or days).
- Older scenarios are upgraded automatically when loaded: missing settings get defaults and retired settings are removed.

### 5.2 Input Streams

Add one or more inputs with **+ Add Input Stream**. Each input has a **Transport**:

**NDI (received in OBS)**, the normal way to test cameras:

| Setting | Meaning |
|---|---|
| Number of Streams | How many streams OBS should receive. Each gets its own NDI Source in scene **"VX3 NDI"**, assigned a **different** NDI source found on the network. |
| Fill Missing NDI Streams with Synthetic | If fewer NDI sources are found than requested, add synthetic streams to make up the difference (e.g. 12 real + 8 synthetic = 20). |
| Synthetic Stream Format | 720p30, 1080p30, 1080p60 or 4K30 for fill streams. |
| Only Use NDI Sources Containing | Optional name filter (e.g. a camera machine name). OBS's own NDI outputs are always skipped. |
| NDI Discovery Wait (s) | How long OBS may search for enough sources before the session starts with those found. |
| Bandwidth | Highest (full quality) or Lowest (preview-quality NDI). |
| Allow NDI Hardware Acceleration (DistroAV) | Lets the NDI SDK decode on the GPU where possible (mainly NDI\|HX sources). |
| OBS Executable | Leave empty for auto-detection, or give the full path to `obs64.exe`. |
| obs-websocket Port | Default 4455. A new password is generated for every session. |
| Start OBS Minimized / Tile Sources in a Grid / Keep OBS Open After the Session | Behaviour of the OBS window. Tiling places every source on the program scene like a multiview. |

**Synthetic.** Generated test streams (width, height, frame rate, count). **When an NDI input is present, synthetic inputs are played inside the same OBS scene too**, so every stream loads OBS the same way. Without an NDI input they run inside the benchmark.

### 5.3 AI Models

Upload `.onnx` files with **Browse** (they are stored in the project's `models\` folder), then configure each model:

| Setting | Meaning |
|---|---|
| Enabled | Include this model in the session. |
| Device | Auto (best available), GPU – CUDA, GPU – TensorRT, GPU – DirectML, CoreML, or CPU. If the chosen device is not installed or not usable on this machine, the model **falls back** down the chain to the CPU and the card/report show *Device fallback* with the reason (see [3.7](#37-platform-notes-windows-linux-arm-macos)). |
| Warmup Iterations | First inferences excluded from all statistics (GPU/cuDNN/TensorRT warm-up). The first one is reported as cold-start latency. |
| Starting Processes | Instances to start with (auto-scaling adds more). |
| AI FPS Target per Stream (this model) | Optional per-model override of the global target. |

Global AI settings:

| Setting | Default | Meaning |
|---|---|---|
| AI FPS Target per Stream | 15 | Each model must sustain this × number of streams. 0 = no target (instances run flat out). |
| AI Worker Mode | Shared | **Shared**: all instances of a model are threads in one process, sharing one GPU context. **Process**: one OS process per instance. |
| Auto-scale Processes | on | Add instances until the target is met. |
| Max Processes per Model | 16 | Upper limit while searching. |
| Settle Time (s) | 8 | Steady running time before an instance count is judged. |
| Target Tolerance (%) | 2 | A total within this % of the target counts as met. |
| Min Gain per Added Process (%) | 5 | **Always on.** An added instance must raise the measured total by at least this much; otherwise scaling stops (Saturated). An instance that *lowers* the total is closed. |

### 5.4 Encode Test

N independent live **H.264** encode streams at **1080p30** or **4K30**. Each stream has its own encoder process fed with moving frames at exactly 30 FPS. It uses a **GPU encoder** when one works on this machine, otherwise the CPU. If a stream can't open a GPU encoder session (e.g. the NVIDIA consumer-GPU session limit), that stream falls back to the CPU and the report says why.

### 5.5 Image Persistence

One saver per enabled AI model repeatedly writes the **same image to the same file**, one save per AI inference, following that model's live measured FPS. Requests are queued, so a slow save is caught up; requests older than 2 s are dropped and counted.

| Setting | Meaning |
|---|---|
| Directory | Where images are written. Empty = the session folder. **Point it at the disk you want to qualify.** |
| Width / Height / Format / Quality | The saved image. |
| Force to Disk (fsync) | On: every save is flushed to the storage device (a true disk test). Off: writes may only reach the OS cache. |
| Worker Mode | Shared (threads in one process) or one process per model. |

### 5.6 Image Capture, Telemetry, Targets
- **Image Capture:** polls a snapshot image at the configured interval (ms).
- **Telemetry:** sample rate (Hz) of CPU/RAM/GPU/disk/network sampling.
- **Pass / Fail Targets:** thresholds used by the bottleneck analysis (minimum stream FPS, maximum CPU/GPU/VRAM/RAM %, maximum temperature, dropped-frame %, memory growth, minimum free disk). See [9.6](#96-targets).

### 5.7 Running a session

Press **Start** (the **Stop** button ends a session early). The live panels show:
- **Live Stream FPS:** every stream, its FPS against the target, resolution, OBS input name, rendered FPS and render time. Synthetic and emulated streams are labelled.
- **OBS card:** OBS render FPS, CPU, RAM, skipped frames, the DistroAV settings used and any warnings.
- **Live AI Models & Processes:** total FPS against target per model, state badge (Scaling / Target met / Saturated / Max processes reached / Failed), and a bar per instance.
- **Live Encode Test & Image Persistence:** per-stream encode FPS, kept-up %, bitrate, GPU/CPU encoder; saves/s against model FPS.
- **Live Telemetry:** charts of CPU, RAM, GPU, disk, network and tool processes.
- **Live Log:** the session's console output (OBS setup, assignments, scaling decisions, warnings).

When the duration ends (or you press **Stop**), the reports are generated and linked from the session panel (**Open report** / **Open PDF report**).

---

## 6. Common workflows

### 6.1 Qualify a machine with your real cameras
1. Add an **NDI** input, set **Number of Streams** to the number of cameras, and turn **Fill Missing** off.
2. Add your AI model(s), keep the target at 15 FPS per stream, and enable Encode Test / Image Persistence as needed.
3. Set a duration (e.g. 30 minutes; hours for soak tests) and press **Start**.
4. Review the report's **Bottleneck Analysis**.

### 6.2 Test more streams than you have cameras
- **Realistic (recommended):** run the [camera emulator](#7-ndi-camera-emulator-second-pc) on a second PC to publish the extra streams as real NDI over the network, set the stream count to the total, and turn **Fill Missing** off.
- **Quick:** leave **Fill Missing** on. The shortfall is filled with synthetic streams in OBS. These load OBS's decoding like a real stream but add **no network load**, so they stay at 30 FPS even when real streams are network-limited.

### 6.3 Compare worker layouts (A/B)
Run the same scenario twice, changing only **AI Worker Mode**, the encode test's **Worker Mode** and persistence's **Worker Mode** between *Shared* and *Process*. Compare AI totals, encode kept-up %, and the tool's RAM/CPU in the telemetry section.

### 6.4 Soak test
Use a long duration (hours or days). Charts adapt their time axis automatically, and long histories are compacted without losing dips. Watch *Memory Growth* and *Thermal* in the bottleneck analysis.

---

## 7. NDI camera emulator (second PC)

`ndi_camera_emulator.py` publishes realistic emulated NDI cameras from **another** PC. NDI traffic between programs on the *same* machine never goes through the network card, so only a second PC can reproduce real network load.

- Moving, detailed test content, so NDI's encoder produces a real camera's data rate (default ≈ 110 Mbit/s per 1080p30 stream).
- Each camera runs in its own process, on its own precise clock.
- A live status line shows FPS sent per camera, receivers connected, and total network TX.

### Setup (second PC)
```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-emulator.txt
```
Also install the **NDI Runtime** (NDI Tools).

### Run
```bat
python ndi_camera_emulator.py --streams 8
```

| Option | Default | Meaning |
|---|---|---|
| `--streams` | 8 | Number of emulated cameras |
| `--format` | 1080p30 | 720p30, 720p60, 1080p25, 1080p30, 1080p50, 1080p60, 4k30 |
| `--detail` | 5.5 | Picture detail/noise 0–10. Higher means a higher NDI data rate (≈100 Mbit/s at 5, ≈119 at 6 for 1080p30). Tune with the TX readout. |
| `--name` | `VX3 EMU` | Source name prefix. Sources appear as `SENDER-PC (VX3 EMU 01)`. |
| `--frames` | 1 s worth (10 for 4K) | Distinct frames per camera loop. Lower it to save RAM (about 120 MiB per 1080p camera at 30 frames). |
| `--status-seconds` | 2 | Status interval |

Example status line:
```
[14:15:42] 01: 30.0fps/1rx  02: 30.0fps/1rx ... | network TX 880 Mbit/s, 110 Mbit/s per received stream
```
Stop with **Ctrl+C**. On the benchmark PC these sources are discovered like real cameras and labelled **emulated camera** in the UI and reports.

---

## 8. Reading the results

Each session is written to `reports\<scenario>\<YYYYMMDD-HHMMSS-scenario>\`:

| File / folder | Content |
|---|---|
| `report.html` | Interactive report. Charts zoom and pan on the time axis (wheel/drag, Shift-drag to select, double-click to reset), have legends you can click to hide series, a "fit Y" toggle and hover values. |
| `report.pdf` | The same content as a PDF: vector charts (zoom without blur), one **layer per chart line** (toggle in the Layers panel of Acrobat, Foxit, PDF-XChange or Okular), bookmarks, readable configuration pages. **Attachments:** `report.html`, `results.json`, `config.json` and full-resolution CSVs. |
| `results.json` | All raw measurements. |
| `scenario.json`, `effective-config.json` | The scenario as saved, and as run. |
| `session.log` | Full console log of the session. |
| `streams\`, `ai\`, `aux\`, `live-*.json`, `obs-source-profiler.json` | Per-worker live data. |
| `persistence\` | Images written by the persistence test (when no directory was set). |

The synthetic-stream clip cache is kept in `reports\<scenario>\_vx3_cache\` and can be deleted at any time.

### Key metrics

| Metric | How it is measured |
|---|---|
| **Stream FPS (NDI in OBS)** | OBS Source Profiler **async input** for the source: frames DistroAV delivered to OBS, averaged by OBS over about 5 s. *Rendered FPS* is how many of those OBS drew. Frame counts are estimated from the FPS. |
| **Stream FPS (synthetic in OBS)** | The same profiler value for the OBS Media Source playing the SpeedHQ clip. |
| **AI FPS** | Exact completed inferences per second, from each instance's counters. Only the inference call is timed; warm-up is excluded; on GPU, inputs and outputs stay on the device. |
| **AI latency** | Per-inference time; mean, p50/p90/p95/p99 (1 µs resolution), max, cold start. *Capacity FPS* = 1000 / mean latency. |
| **Scaling steps** | The settled total FPS at each instance count, plus every scaling decision with its reason. |
| **Encode** | Frames out of each encoder (from FFmpeg's own counter, 2 s warm-up excluded), kept-up % against 30 FPS, dropped frame slots, measured bitrate, frame hand-off time. |
| **Persistence** | Saves/s against the model's inference rate, kept-up %, dropped requests, save latency (open, write, flush, fsync, close), MB/s. |
| **Telemetry** | CPU, RAM, swap, disk, network RX/TX, per-GPU utilisation/VRAM/temperature/power/clocks/PCIe/encoder/decoder, and the tool's own process tree (including OBS). |

### Bottleneck analysis

The report ranks findings such as *NDI Stream Rate*, *AI Throughput*, *Encode*, *Storage / Image Persistence*, *CPU*, *GPU Compute/Memory*, *System Memory*, *Thermal* and *Memory Growth*. Each comes with the evidence and a recommendation, based on the thresholds in **Targets**.

**Tip:** if every NDI stream sits below 30 FPS while **Network RX** flattens around 900–950 Mbit/s, the **1 GbE link is saturated**. Consider 2.5/10 GbE, NDI\|HX sources, lower bandwidth, or fewer streams per machine.

---

## 9. Configuration reference

Scenarios are JSON files in `configs\`. The web interface covers every setting below. Defaults are shown.

### 9.1 Session
```json
"session": { "duration": { "value": 5, "unit": "minutes" } }
```
`unit`: `seconds`, `minutes`, `hours` or `days`.

### 9.2 Inputs
```json
"inputs": [
  { "id": "ndi", "transport": "ndi", "enabled": true, "stream_count": 12,
    "fill_with_synthetic": true, "synthetic_format": "1080p30", "source_filter": "",
    "discovery_timeout_seconds": 15, "bandwidth": "highest", "hardware_acceleration": true,
    "obs_path": "", "websocket_port": 4455, "minimize": true, "arrange_grid": true,
    "keep_obs_open": false },
  { "id": "synthetic", "transport": "synthetic", "enabled": false, "stream_count": 1,
    "width": 1920, "height": 1080, "framerate": 30 }
]
```
Advanced NDI-input keys (JSON only):

| Key | Default | Meaning |
|---|---|---|
| `obs_ready_timeout_seconds` | 90 | How long to wait for OBS to finish loading after its websocket opens. |
| `source_name_key` | `ndi_source_name` | DistroAV's source-name setting, in case a future version renames it. |
| `emulator_name` | `VX3 EMU` | Name prefix used to label emulated cameras. |
| `receive_audio` | false | Keep NDI audio enabled on the sources. |
| `ai_enabled` | true | Count this input's streams toward the AI target (applies to any input). |

### 9.3 AI
```json
"ai": {
  "target_fps_per_stream": 15, "worker_mode": "shared",
  "autoscale": { "enabled": true, "max_instances": 16, "settle_seconds": 8,
                 "tolerance_percent": 2, "min_gain_percent": 5 },
  "models": {
    "yolo": { "enabled": true, "path": "../models/yolo.onnx", "device": "cuda",
              "warmup_iterations": 10, "instances": 1 }
  }
}
```
Per-model optional keys: `target_fps_per_stream`, `worker_mode`, `device_id` (GPU index), `intra_op_num_threads`, `inter_op_num_threads`. `device`: `auto`, `cuda`, `tensorrt`, `directml`, `coreml` or `cpu` (with automatic fallback). `min_gain_percent` must be greater than 0.

### 9.4 Encode test and image persistence
```json
"encode_test": { "enabled": false, "resolution": "1080p30", "stream_count": 1, "worker_mode": "shared" },
"image_persistence": { "enabled": false, "directory": "", "width": 1920, "height": 1080,
                       "format": "jpeg", "quality": 90, "fsync": true, "worker_mode": "shared" }
```
`resolution`: `1080p30` or `4k30`. `format`: `jpeg`, `png` or `webp`. Persistence advanced key: `max_backlog_seconds` (default 2).

### 9.5 Image capture and telemetry
```json
"image_capture": { "enabled": false, "snapshot_polling_rate": 1000 },
"telemetry": { "sample_hz": 2 }
```
Telemetry advanced keys: `fps_history_interval_seconds` (1), `windows_gpu_counters` (true; Intel iGPU counters via Windows), `windows_gpu_counter_interval_seconds` (10).

### 9.6 Targets
```json
"targets": { "stream_fps_min": 29, "cpu_percent_max": 90, "gpu_percent_max": 95,
             "vram_percent_max": 90, "ram_percent_max": 90, "temperature_c_max": 90,
             "dropped_frame_percent_max": 1, "process_rss_growth_mb_max": 256,
             "disk_free_gb_min": 20 }
```

### 9.7 Environment variables

| Variable | Purpose |
|---|---|
| `VX3_FFMPEG` | Full path to the FFmpeg executable to prefer (e.g. a full build with NVENC). |

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| OBS card: **"OBS Studio not found"** | Set **OBS Executable** on the NDI input to the full path of `obs64.exe` (Windows) or `obs` (Linux). On Linux, install OBS (see [3.6](#36-linux-notes)). |
| OBS card: **"no graphical display"** (Linux) | The benchmark was started without a desktop session (SSH/service). Start it from the desktop, set `DISPLAY=:0`, or use `xvfb-run` (see [3.6](#36-linux-notes)). |
| **"OBS aborted (SIGABRT) during startup"** (Linux) | Usually OBS can't open the display or create its OpenGL context. The error lists OBS's own last messages and log path; check graphics drivers, try running `obs` by hand in the same terminal, or use `xvfb-run`. |
| OBS card: **"OBS has no 'NDI Source' input"** | DistroAV isn't installed for this OBS (or the NDI Runtime is missing). Install both and check *Sources → + → NDI Source* in OBS. |
| **"OBS was still not ready after … s"** | OBS took too long to load (plugins/scene collection). Close other heavy programs, or raise `obs_ready_timeout_seconds`. Short "not ready" periods are retried automatically. |
| **"could not connect to obs-websocket"** | Another OBS is already running, or the port is in use. Close all OBS instances or change **obs-websocket Port**. |
| **"… is not DistroAV's source-name setting"** | A DistroAV version renamed the setting. Set `source_name_key` on the NDI input. |
| OBS card: **"Source Profiler: …"** warning, or stream FPS stays 0 | OBS older than 30.1, or the profiler script couldn't load. Update OBS, and make sure `vx3bench/obs/vx3_source_profiler.lua` is the latest version (older versions failed on Linux with *libobs.so: cannot open shared object file*). In OBS, *Tools → Scripts* should list the script. |
| **Fewer NDI streams than requested** | Not enough sources were visible within the discovery wait. Check the cameras in NDI Studio Monitor, raise **NDI Discovery Wait**, check **Only Use NDI Sources Containing**, or enable **Fill Missing**. |
| Real streams at 14–20 FPS, synthetic at 30 | Network-limited (see the tip in [8](#8-reading-the-results)). Check **Network RX** in telemetry. |
| AI card shows **"Device fallback: GPU – … requested → running on CPU"** | The chosen accelerator isn't usable here. *"no GPU provider is installed"*: the CPU-only `onnxruntime` package is installed (or both packages are); install `onnxruntime-gpu` (x64 NVIDIA) or `onnxruntime-directml`. *"not usable on this machine"*: the provider is installed but the GPU or its libraries (driver, CUDA 12, cuDNN 9, TensorRT) are missing or mismatched. Verify with the command in [3.2](#32-create-a-python-environment-and-install-packages). The session still runs, on the device shown. |
| AI model **Saturated** early | The hardware can't add throughput with more instances (the added instance was below the minimum gain). This is a result, not an error. See the *Settled total at each instance count* table. |
| Encode: **CPU (libx264)** although you have a GPU | The FFmpeg in use has no working hardware encoder for this GPU (or, on Linux, no access to `/dev/dri`: add your user to the `video`/`render` groups). See [3.5](#35-ffmpeg-with-hardware-encoders-recommended), [3.7](#37-platform-notes-windows-linux-arm-macos) and the report's *GPU encoder detection*. |
| Encode stream **"GPU fallback: … failed to start"** | The GPU's concurrent encode-session limit was reached (common on consumer NVIDIA cards). Those streams run on the CPU. |
| Persistence **< 98 % kept up** | The storage path can't sustain one save per inference with fsync. Compare *Capacity FPS* and *max latency*; try another disk, exclude the directory from antivirus scanning, or turn fsync off to compare. |
| **No PDF report** | `reportlab`/`pypdf` are missing. The session log says so; install them. The HTML report is still written. |
| Behaviour doesn't match the documentation after an update | Old files are still in place. Replace the **whole** `vx3bench\` folder. The session log's first AI line shows the scaler version, e.g. `[ai] scaler version 7 …`. |

**Logs:** the session output is shown in the **Live Log** panel and saved as `session.log` in the session folder. Include it when reporting a problem.

---

## 11. Project layout

```
Benchmark-Tool\
├─ README.md
├─ requirements.txt              # benchmark PC
├─ requirements-emulator.txt     # second PC
├─ ndi_camera_emulator.py        # run on the second PC
├─ vx3bench\                     # the application package
│  ├─ __main__.py, app.py        # entry point (python -m vx3bench)
│  ├─ webserver.py, session.py   # web UI backend, session management
│  ├─ engine.py                  # session engine (starts and stops all workloads)
│  ├─ config.py, duration.py     # scenario defaults and validation
│  ├─ obs_controller.py          # launches/drives OBS, reads Source Profiler data
│  ├─ obs\vx3_source_profiler.lua# OBS script exporting per-source profiler data
│  ├─ model_workers.py           # AI instances, auto-scaler, measurements
│  ├─ encode_workload.py         # encode test (FFmpeg, GPU/CPU)
│  ├─ aux_workloads.py           # image persistence
│  ├─ image_capture.py           # snapshot polling
│  ├─ stream_worker.py, pipeline.py, sources.py, live_metrics.py  # synthetic streams run by the benchmark
│  ├─ telemetry.py, inventory.py # system sampling and hardware inventory
│  ├─ bottleneck.py              # bottleneck analysis
│  ├─ report.py, report_charts.py, report_fps.py, report_pdf.py   # HTML/PDF reports
│  ├─ live_publish.py            # atomic JSON publishing between processes
│  ├─ snapshot.jpeg              # image served for the image-capture test
│  └─ web\static\                # index.html, app.js, stream-fps.js, *.css
├─ models\                       # uploaded ONNX models (created on first upload)
├─ configs\                      # scenarios (created on first start)
└─ reports\                      # session results (created on first session)
```

---

## 12. Known limitations

- **Stream FPS comes from OBS's Source Profiler**, which averages over about 5 s. Frame counts are estimated, and OBS does not expose per-frame loss or duplicate information.
- **Synthetic streams** reproduce OBS's decoding and rendering load, not network load. Use the camera emulator on a second PC when network behaviour matters.
- The OBS automation expects **one OBS instance**. Close OBS before starting a session.
- Hardware encode-session limits depend on the GPU and driver.
- NDI hardware acceleration is a request to the NDI SDK; whether it applies depends on the source type (mainly NDI\|HX) and hardware.
