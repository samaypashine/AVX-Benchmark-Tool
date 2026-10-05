"""PDF edition of the qualification report.

What makes the PDF explorable (PDF viewers can't run the HTML report's
JavaScript, so it uses PDF-native features instead):

* Every chart is vector graphics, one chart per landscape page, drawn with
  thin lines -- zoom in as far as the viewer allows and lines/text stay sharp.
* Every chart series (each stream, each model, each telemetry line) is its
  own PDF layer (Optional Content Group), grouped by chart in the viewer's
  Layers panel, so series can be hidden/shown to isolate one stream.
  Supported by Adobe Acrobat/Reader, Foxit, PDF-XChange, Okular and others;
  some browser built-in viewers ignore layers (everything is simply shown).
* Bookmarks for every section and chart.
* Attachments: the interactive HTML report (open it in a browser for
  time-axis zoom/pan and hover tooltips), results.json, and full-resolution
  CSVs of stream FPS history, AI FPS history and telemetry.

Dependencies: reportlab (drawing) and pypdf (layers + attachments), both
pure-Python wheels on Windows, Linux and ARM64.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from xml.sax.saxutils import escape

from .report_charts import PDF_MAX_POINTS, decimate, fmt_duration, fmt_value, nice_ticks, time_ticks

PAGE = landscape(letter)
MARGIN = 0.45 * inch
INK = colors.HexColor("#1b2a3a")
MUTED = colors.HexColor("#5b6f84")
GRID = colors.HexColor("#d9e2ec")
AXIS = colors.HexColor("#8aa0b6")
HEAD_BG = colors.HexColor("#0d1d30")
ROW_ALT = colors.HexColor("#f2f6fa")
TARGET = colors.HexColor("#d99a00")

# Colours tuned for a dark UI are too pale on white paper; darken them a bit.
def _paper(hex_color: str):
    c = colors.HexColor(hex_color)
    return colors.Color(c.red * 0.82, c.green * 0.82, c.blue * 0.82)


def _tick(v: float, unit: str) -> str:
    text = f"{round(v):,}" if abs(v) >= 1000 else f"{round(v, 4):g}"
    return text if not unit else (text + unit if unit in ("%", "°C") else f"{text} {unit}")


def _p(text: Any, style) -> Paragraph:
    return Paragraph(escape(str(text)), style)


# --------------------------------------------------------------------------- #
# Chart flowable
# --------------------------------------------------------------------------- #
class LayerRegistry:
    def __init__(self):
        self.layers: list[dict[str, Any]] = []   # {"tag","group","name","page"}

    def new(self, group: str, name: str, page: int) -> str:
        tag = f"OC{len(self.layers) + 1}"
        self.layers.append({"tag": tag, "group": group, "name": name, "page": page})
        return tag


class ChartFlowable(Flowable):
    LEGEND_ROW = 11

    def __init__(self, spec: dict, width: float, height: float, registry: LayerRegistry):
        super().__init__()
        self.spec = spec
        self.width = width
        self.registry = registry
        n = len(spec["series"])
        cols = max(1, int(width // 150))
        self.legend_rows = math.ceil(n / cols) if n > 1 else 0
        self.legend_cols = cols
        legend_h = self.legend_rows * self.LEGEND_ROW + (6 if self.legend_rows else 0)
        self.plot_height = max(160, height - legend_h)
        self.height = self.plot_height + legend_h

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        spec = self.spec
        L, R, T, B = 56, 12, 8, 34
        x_org, y_org = L, self.height - self.plot_height + B
        pw, ph = self.width - L - R, self.plot_height - T - B

        series = [dict(s, points=decimate(s["points"], PDF_MAX_POINTS)) for s in spec["series"]]
        xs = [p[0] for s in series for p in s["points"]]
        ys = [p[1] for s in series for p in s["points"]]
        x0, x1 = min(0.0, min(xs)), max(xs)   # time axis always starts at session start
        if x1 <= x0:
            x1 = x0 + 1
        if spec.get("target") is not None:
            ys.append(spec["target"])
        lo, hi = min(ys), max(ys)
        if spec.get("y_from_zero", True):
            lo = min(0.0, lo)
        if hi <= lo:
            hi = lo + max(1.0, abs(lo) * 0.1)
        hi += (hi - lo) * 0.06
        if spec.get("y_range"):
            lo, hi = spec["y_range"]
        sx = lambda x: x_org + (x - x0) / (x1 - x0) * pw
        sy = lambda y: y_org + (y - lo) / (hi - lo) * ph

        c.saveState()
        c.setFont("Helvetica", 7)
        c.setLineWidth(0.4)
        # Y grid + labels
        for v in nice_ticks(lo, hi, 6):
            if v < lo - 1e-9 or v > hi + 1e-9:
                continue
            y = sy(v)
            c.setStrokeColor(GRID); c.line(x_org, y, x_org + pw, y)
            c.setFillColor(MUTED); c.drawRightString(x_org - 4, y - 2.5, _tick(v, spec["unit"]))
        # X grid + adaptive time labels
        ticks, axis_title = time_ticks(x0, x1, 9)
        for pos, label in ticks:
            if pos < x0 - 1e-9 or pos > x1 + 1e-9:
                continue
            x = sx(pos)
            c.setStrokeColor(GRID); c.line(x, y_org, x, y_org + ph)
            c.setFillColor(MUTED); c.drawCentredString(x, y_org - 11, label)
        c.setStrokeColor(AXIS); c.setLineWidth(0.6)
        c.line(x_org, y_org, x_org + pw, y_org); c.line(x_org, y_org, x_org, y_org + ph)
        c.setFillColor(INK); c.setFont("Helvetica", 7.5)
        c.drawCentredString(x_org + pw / 2, y_org - 24,
                            f"{axis_title}  -  session span {fmt_duration(x0)} to {fmt_duration(x1)}")
        # Target line
        if spec.get("target") is not None:
            y = sy(spec["target"])
            c.setStrokeColor(TARGET); c.setDash(4, 3); c.setLineWidth(0.7)
            c.line(x_org, y, x_org + pw, y); c.setDash()
            c.setFillColor(TARGET); c.setFont("Helvetica", 7)
            c.drawString(x_org + 4, y - 9, spec.get("target_label") or f"Target {spec['target']}")
        # Series, each in its own optional-content layer, clipped to the plot
        clip = c.beginPath(); clip.rect(x_org, y_org, pw, ph)
        c.clipPath(clip, stroke=0, fill=0)
        page = c.getPageNumber()
        for s in series:
            tag = self.registry.new(spec["title"], s["name"], page)
            col = _paper(s["color"])
            c.saveState()
            c._code.append(f"/OC /{tag} BDC")
            c.setStrokeColor(col); c.setFillColor(col)
            c.setLineWidth(0.9); c.setLineJoin(1); c.setLineCap(1)
            pts = s["points"]
            if len(pts) == 1:
                c.circle(sx(pts[0][0]), sy(pts[0][1]), 1.6, stroke=0, fill=1)
            else:
                path = c.beginPath()
                path.moveTo(sx(pts[0][0]), sy(pts[0][1]))
                for x, y in pts[1:]:
                    path.lineTo(sx(x), sy(y))
                c.drawPath(path, stroke=1, fill=0)
            c._code.append("EMC")
            c.restoreState()
        c.restoreState()

        # Legend (always visible; layers toggle the lines themselves)
        if self.legend_rows:
            c.saveState()
            c.setFont("Helvetica", 7)
            col_w = (self.width - L) / self.legend_cols
            top = self.height - self.plot_height - 4
            for i, s in enumerate(spec["series"]):
                r, k = divmod(i, self.legend_cols)
                x = L + k * col_w
                y = top - (r + 1) * self.LEGEND_ROW + 3
                c.setFillColor(_paper(s["color"])); c.rect(x, y, 7, 7, stroke=0, fill=1)
                c.setFillColor(INK)
                name = s["name"] if len(s["name"]) <= 34 else s["name"][:32] + "..."
                c.drawString(x + 10, y + 0.5, name)
            c.restoreState()


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
class _Doc(BaseDocTemplate):
    def __init__(self, filename, title, **kw):
        super().__init__(filename, pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN, title=title, author="VX3 Benchmark", **kw)
        frame = Frame(MARGIN, MARGIN, PAGE[0] - 2 * MARGIN, PAGE[1] - 2 * MARGIN - 10, id="main")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._footer)])
        self.report_title = title

    def _footer(self, canv, doc):
        canv.saveState()
        canv.setFont("Helvetica", 7.5); canv.setFillColor(MUTED)
        canv.drawString(MARGIN, MARGIN - 18, f"VX3 NDI Stream Qualification  -  {self.report_title}")
        canv.drawRightString(PAGE[0] - MARGIN, MARGIN - 18, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, flowable):
        level = getattr(flowable, "_outline_level", None)
        if level is not None:
            key = f"bm{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level, closed=level > 0)


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=24, leading=28, textColor=INK, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, leading=20, textColor=INK, spaceBefore=4, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, leading=15, textColor=INK, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12, textColor=INK),
        "muted": ParagraphStyle("m", parent=ss["BodyText"], fontSize=8, leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=ss["BodyText"], fontSize=7.5, leading=9.5, textColor=INK),
        "cellw": ParagraphStyle("cw", parent=ss["BodyText"], fontSize=7.5, leading=9.5, textColor=colors.white),
        "mono": ParagraphStyle("mono", fontName="Courier", fontSize=6.8, leading=8.2, textColor=INK),
    }


def _heading(text, style, level):
    p = Paragraph(escape(text), style)
    p._outline_level = level
    return p


def _table(rows, st, col_widths=None, highlight=None):
    data = [[Paragraph(escape(str(v)), st["cellw"] if r == 0 else st["cell"]) for v in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style = [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.25, GRID), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for r in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, r), (-1, r), ROW_ALT))
    for (r, col, bg) in (highlight or []):
        style.append(("BACKGROUND", (col, r), (col, r), bg))
    t.setStyle(TableStyle(style))
    return t


def _n(v, digits=2):
    return "N/A" if v is None else f"{v:,.{digits}f}"


def _gib(v):
    return "N/A" if v is None else f"{v / 2**30:,.2f} GiB"


# --------------------------------------------------------------------------- #
# Configuration, rendered as readable cards and tables
# --------------------------------------------------------------------------- #
CARD_HEAD = colors.HexColor("#15324d")
CARD_EDGE = colors.HexColor("#c9d6e3")
YES = colors.HexColor("#1f7a4d")
NO = colors.HexColor("#8a97a6")

CONFIG_LABELS = {
    "name": "Scenario name", "duration": "Duration", "id": "ID", "transport": "Transport", "enabled": "Enabled",
    "stream_count": "Streams", "framerate": "Frame rate", "target_fps_per_stream": "AI FPS target per stream",
    "max_instances": "Max processes per model", "settle_seconds": "Settle time", "tolerance_percent": "Target tolerance",
    "min_gain_percent": "Min gain per added process", "device": "Device", "path": "Model file",
    "instances": "Starting processes", "warmup_iterations": "Warm-up iterations", "device_id": "GPU index",
    "intra_op_num_threads": "Intra-op threads", "inter_op_num_threads": "Inter-op threads",
    "resolution": "Stream format", "directory": "Directory", "fsync": "Force to disk (fsync)",
    "quality": "Quality", "format": "Format", "sample_hz": "Sample rate", "stream_fps_min": "Minimum stream FPS",
    "snapshot_polling_rate": "Snapshot polling interval", "ai_enabled": "AI enabled",
    "discovery_timeout_seconds": "NDI discovery timeout", "windows_gpu_counters": "Windows GPU counters",
    "fps_history_interval_seconds": "FPS history interval", "max_backlog_seconds": "Max save backlog",
    "worker_mode": "Worker mode", "hardware_acceleration": "NDI hardware acceleration (DistroAV)",
    "source_filter": "NDI source filter", "obs_path": "OBS executable", "websocket_port": "obs-websocket port",
    "minimize": "Start OBS minimized", "arrange_grid": "Tile sources in a grid", "keep_obs_open": "Keep OBS open afterwards",
    "bandwidth": "NDI bandwidth",
}
SECTION_TITLES = {"session": "Session", "inputs": "Input Streams", "ai": "AI Models", "encode_test": "Encode Test",
                  "image_persistence": "Image Persistence", "image_capture": "Image Capture", "telemetry": "Telemetry",
                  "targets": "Pass / Fail Targets"}
DEVICE_NAMES = {"auto": "Auto (CUDA if available, else CPU)", "cuda": "GPU - CUDA", "gpu": "GPU - CUDA",
                "tensorrt": "GPU - TensorRT", "cpu": "CPU"}
ENCODE_NAMES = {"1080p30": "1080p30 (1920 x 1080, 30 FPS)", "4k30": "4K30 (3840 x 2160, 30 FPS)"}


def _label(key: str) -> str:
    if key in CONFIG_LABELS:
        return CONFIG_LABELS[key]
    text = key.replace("_", " ").strip().capitalize()
    for a, b in ((" fps", " FPS"), ("Fps", "FPS"), (" ai", " AI"), (" ndi", " NDI"), (" gpu", " GPU"), (" cpu", " CPU")):
        text = text.replace(a, b)
    return text


def _unit(key: str, value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    if key.endswith("_percent"):
        return " %"
    if key.endswith("_seconds"):
        return " s"
    if key.endswith("_hz"):
        return " Hz"
    if key == "snapshot_polling_rate":
        return " ms"
    if "fps" in key or key == "framerate":
        return " FPS"
    return ""


def _value_cell(key: str, value, st):
    if isinstance(value, bool):
        word = ("Yes" if key not in ("enabled", "ai_enabled") else "Enabled") if value else ("No" if key not in ("enabled", "ai_enabled") else "Disabled")
        col = YES if value else NO
        return Paragraph(f'<font color="{col.hexval()}"><b>{word}</b></font>', st["cell"])
    if value is None or value == "":
        text = "Session output folder" if key == "directory" else "(default)"
        return Paragraph(f'<font color="#8a97a6">{text}</font>', st["cell"])
    if key == "format":
        value = str(value).upper()
    if key == "device":
        value = DEVICE_NAMES.get(str(value).lower(), value)
    if key == "resolution":
        value = ENCODE_NAMES.get(str(value).lower(), value)
    if key == "worker_mode":
        value = {"shared": "Shared process (threads)", "process": "One process per worker"}.get(str(value).lower(), value)
    if isinstance(value, dict) and set(value) >= {"value", "unit"}:
        value = f"{value['value']} {value['unit']}"
    elif isinstance(value, dict):
        value = ", ".join(f"{_label(k)}: {v}" for k, v in value.items())
    elif isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    return Paragraph(escape(f"{value}{_unit(key, value) if isinstance(value, (int, float)) else ''}"), st["cell"])


def _flatten(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, dict) and not (set(v) >= {"value", "unit"}) and v and all(not isinstance(x, (dict, list)) for x in v.values()) and len(v) > 2:
            yield from _flatten(v, f"{prefix}{_label(k)} / ")
        else:
            yield f"{prefix}{_label(k)}", k, v


def _kv_card(title: str, items, st, width):
    """A titled two-column card: coloured header band, label | value rows."""
    data = [[Paragraph(f'<font color="white"><b>{escape(title)}</b></font>', st["cell"]), ""]]
    for label, key, value in items:
        data.append([Paragraph(f'<font color="#5b6f84">{escape(label)}</font>', st["cell"]), _value_cell(key, value, st)])
    if len(data) == 1:
        data.append([Paragraph('<font color="#8a97a6">No settings</font>', st["cell"]), ""])
    t = Table(data, colWidths=[width * 0.46, width * 0.54], hAlign="LEFT")
    style = [("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (-1, 0), CARD_HEAD),
             ("BOX", (0, 0), (-1, -1), 0.6, CARD_EDGE), ("LINEBELOW", (0, 1), (-1, -2), 0.25, GRID),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 7)]
    for r in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, r), (-1, r), ROW_ALT))
    t.setStyle(TableStyle(style))
    return t


def _grid_table(title: str, columns: list[str], rows: list[list], st, width, keys: list[str]):
    """Full-width titled table for lists of similar items (streams, models)."""
    head = [Paragraph(f'<font color="white"><b>{escape(title)}</b></font>', st["cell"])] + [""] * (len(columns) - 1)
    cols = [Paragraph(f'<b>{escape(c)}</b>', st["cell"]) for c in columns]
    body = [[_value_cell(keys[i], v, st) for i, v in enumerate(r)] for r in rows]
    t = Table([head, cols] + body, repeatRows=2, hAlign="LEFT", colWidths=[width / len(columns)] * len(columns))
    style = [("SPAN", (0, 0), (-1, 0)), ("BACKGROUND", (0, 0), (-1, 0), CARD_HEAD), ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e6eef6")),
             ("BOX", (0, 0), (-1, -1), 0.6, CARD_EDGE), ("LINEBELOW", (0, 1), (-1, -1), 0.25, GRID),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    for r in range(3, len(body) + 2, 2):
        style.append(("BACKGROUND", (0, r), (-1, r), ROW_ALT))
    t.setStyle(TableStyle(style))
    return t


def _pair(cards, width):
    """Lay small cards out two per row."""
    out = []
    gap = 12
    half = (width - gap) / 2
    for i in range(0, len(cards), 2):
        left = cards[i](half)
        right = cards[i + 1](half) if i + 1 < len(cards) else ""
        row = Table([[left, "", right]], colWidths=[half, gap, half], hAlign="LEFT")
        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        out += [row, Spacer(1, 10)]
    return out


def _config_story(cfg: dict, st, width) -> list:
    story = []
    small = []   # callables width -> card
    session = {"name": cfg.get("name"), **(cfg.get("session") or {})}
    small.append(lambda w: _kv_card("Session", list(_flatten(session)), st, w))

    ai = cfg.get("ai") or {}
    scaling = {"target_fps_per_stream": ai.get("target_fps_per_stream"), "worker_mode": ai.get("worker_mode", "shared"),
               **(ai.get("autoscale") or {})}
    small.append(lambda w: _kv_card("AI Throughput & Process Scaling",
                                    [("Auto-scale processes" if k == "enabled" else _label(k), k, v) for k, v in scaling.items()], st, w))
    for key in ("encode_test", "image_persistence", "image_capture", "telemetry", "targets"):
        if isinstance(cfg.get(key), dict):
            small.append(lambda w, k=key: _kv_card(SECTION_TITLES[k], list(_flatten(cfg[k])), st, w))
    known = {"name", "session", "inputs", "ai"} | set(SECTION_TITLES)
    for key, value in cfg.items():
        if key not in known and not key.startswith("_"):
            items = list(_flatten(value)) if isinstance(value, dict) else [(_label(key), key, value)]
            small.append(lambda w, k=key, it=items: _kv_card(_label(k), it, st, w))

    # Inputs: one row per input
    inputs = [x for x in cfg.get("inputs") or [] if isinstance(x, dict)]
    if inputs:
        main = ["id", "transport", "enabled", "stream_count", "resolution_", "framerate"]
        extra = sorted({k for x in inputs for k in x} - set(main) - {"width", "height"})
        cols = ["ID", "Transport", "Enabled", "Streams", "Resolution", "Frame rate"] + [_label(k) for k in extra]
        rows = [[x.get("id"), str(x.get("transport", "")).upper(), x.get("enabled", True), x.get("stream_count"),
                 f"{x.get('width')} x {x.get('height')}" if x.get("width") else None, x.get("framerate")]
                + [x.get(k) for k in extra] for x in inputs]
        story += [_grid_table("Input Streams", cols, rows, st, width, main + extra), Spacer(1, 10)]

    models = ai.get("models") or {}
    if models:
        main = ["enabled", "device", "path", "instances", "warmup_iterations"]
        extra = sorted({k for m in models.values() if isinstance(m, dict) for k in m} - set(main))
        cols = ["Model", "Enabled", "Device", "Model file", "Starting processes", "Warm-up"] + [_label(k) for k in extra]
        rows = [[name, m.get("enabled", False), m.get("device", "auto"), m.get("path"), m.get("instances", 1),
                 m.get("warmup_iterations")] + [m.get(k) for k in extra]
                for name, m in models.items() if isinstance(m, dict)]
        story += [_grid_table("AI Models", cols, rows, st, width, ["name"] + main + extra), Spacer(1, 10)]

    return _pair(small, width) + story


# --------------------------------------------------------------------------- #
# CSV attachments (full resolution, not decimated)
# --------------------------------------------------------------------------- #
def _csv(rows, header) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _stream_csv(streams):
    rows = [[s.get("stream_id"), x.get("elapsed_seconds"), x.get("fps"), x.get("fps_min"), x.get("fps_max"),
             x.get("interval_seconds")] for s in streams for x in s.get("fps_history") or []]
    return _csv(rows, ["stream_id", "elapsed_seconds", "fps", "fps_min", "fps_max", "interval_seconds"])


def _ai_csv(models):
    rows = []
    for m in models:
        rows += [[m.get("name"), "total", x.get("elapsed_seconds"), x.get("fps"), m.get("target_fps")] for x in m.get("fps_history") or []]
        for inst in (m.get("instances") or []) + (m.get("retired_instances") or []):
            rows += [[m.get("name"), inst.get("instance"), x.get("elapsed_seconds"), x.get("fps"), inst.get("assigned_fps")]
                     for x in inst.get("fps_history") or []]
    return _csv(rows, ["model", "instance", "elapsed_seconds", "instant_fps", "target_or_assigned_fps"])


def _telemetry_csv(samples):
    keys = []
    for s in samples:
        for k, v in s.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in keys:
                keys.append(k)
    gpu_keys = []
    for s in samples:
        for g in s.get("gpus", []):
            for k, v in g.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in gpu_keys and k != "index":
                    gpu_keys.append(k)
    n_gpu = max((len(s.get("gpus", [])) for s in samples), default=0)
    header = keys + [f"gpu{i}_{k}" for i in range(n_gpu) for k in gpu_keys]
    rows = []
    for s in samples:
        row = [s.get(k) for k in keys]
        gpus = s.get("gpus", [])
        for i in range(n_gpu):
            g = gpus[i] if i < len(gpus) else {}
            row += [g.get(k) for k in gpu_keys]
        rows.append(row)
    return _csv(rows, header)


# --------------------------------------------------------------------------- #
# Optional content (layers) + attachments
# --------------------------------------------------------------------------- #
def _attach(writer, attachments: dict[str, bytes]):
    """Embedded files, Flate-compressed (JSON/CSV/HTML shrink ~10-20x)."""
    from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject, NameObject,
                               NumberObject, TextStringObject)
    mime = {".html": "text/html", ".json": "application/json", ".csv": "text/csv"}
    names = ArrayObject()
    for name in sorted(attachments):
        data = attachments[name]
        raw = DecodedStreamObject()
        raw.set_data(data)
        stream = raw.flate_encode()
        stream[NameObject("/Type")] = NameObject("/EmbeddedFile")
        stream[NameObject("/Subtype")] = NameObject("/" + mime.get(Path(name).suffix, "application/octet-stream").replace("/", "#2F"))
        stream[NameObject("/Params")] = DictionaryObject({NameObject("/Size"): NumberObject(len(data))})
        spec = DictionaryObject({
            NameObject("/Type"): NameObject("/Filespec"),
            NameObject("/F"): TextStringObject(name), NameObject("/UF"): TextStringObject(name),
            NameObject("/Desc"): TextStringObject(ATTACHMENT_NOTES.get(name, name)),
            NameObject("/EF"): DictionaryObject({NameObject("/F"): writer._add_object(stream),
                                                 NameObject("/UF"): writer._add_object(stream)}),
        })
        names += [TextStringObject(name), writer._add_object(spec)]
    if names:
        writer._root_object[NameObject("/Names")] = DictionaryObject({
            NameObject("/EmbeddedFiles"): DictionaryObject({NameObject("/Names"): names})})


ATTACHMENT_NOTES = {
    "config.json": "Exact benchmark configuration used for this session",
    "report.html": "Interactive report: open in a browser for zoom/pan/hover on every chart",
    "results.json": "All raw measurements recorded by the engine",
    "stream_fps_history.csv": "Full-resolution per-stream FPS history",
    "ai_fps_history.csv": "Per-model FPS history",
    "telemetry.csv": "Every telemetry sample (system, process and per-GPU fields)",
}
MAX_ATTACHMENT_BYTES = 200 * 2**20   # larger raw files are left next to the PDF instead


def _finalize(tmp: Path, final: Path, registry: LayerRegistry, attachments: dict[str, bytes], title: str):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ArrayObject, DictionaryObject, NameObject, TextStringObject

    writer = PdfWriter(clone_from=PdfReader(str(tmp)))
    refs, per_page, order, groups = {}, {}, ArrayObject(), {}
    for layer in registry.layers:
        ocg = DictionaryObject({NameObject("/Type"): NameObject("/OCG"),
                                NameObject("/Name"): TextStringObject(layer["name"])})
        ref = writer._add_object(ocg)
        refs[layer["tag"]] = ref
        per_page.setdefault(layer["page"], []).append(layer["tag"])
        if layer["group"] not in groups:
            groups[layer["group"]] = ArrayObject([TextStringObject(layer["group"])])
            order.append(groups[layer["group"]])
        groups[layer["group"]].append(ref)

    for page_no, tags in per_page.items():
        page = writer.pages[page_no - 1]
        resources = page.get("/Resources")
        resources = resources.get_object() if resources is not None else DictionaryObject()
        props = resources.get("/Properties")
        props = props.get_object() if props is not None else DictionaryObject()
        for tag in tags:
            props[NameObject("/" + tag)] = refs[tag]
        resources[NameObject("/Properties")] = props
        page[NameObject("/Resources")] = resources

    if refs:
        all_refs = ArrayObject(refs.values())
        writer._root_object[NameObject("/OCProperties")] = DictionaryObject({
            NameObject("/OCGs"): all_refs,
            NameObject("/D"): DictionaryObject({
                NameObject("/Name"): TextStringObject("Chart series"),
                NameObject("/BaseState"): NameObject("/ON"),
                NameObject("/ON"): ArrayObject(refs.values()),
                NameObject("/Order"): order,
            }),
        })
    writer._root_object[NameObject("/PageMode")] = NameObject("/UseOutlines")
    _attach(writer, attachments)
    writer.add_metadata({"/Title": f"VX3 Qualification - {title}", "/Author": "VX3 Benchmark"})
    with open(final, "wb") as fh:
        writer.write(fh)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def write_pdf(path, p: dict, *, fps_target: float, chart_sections, attachments: dict[str, Path] | None = None):
    path = Path(path)
    st = _styles()
    summary = p.get("run_summary", {})
    a = p.get("bottleneck_analysis", {})
    title = str(summary.get("scenario", "Benchmark"))
    width = PAGE[0] - 2 * MARGIN
    chart_h = PAGE[1] - 2 * MARGIN - 64
    registry = LayerRegistry()
    story: list = []
    streams = p.get("streams", []) or []
    models = p.get("ai_models", []) or []
    files = {"config.json": json.dumps(p.get("config", {}), indent=2).encode("utf-8"),
             "stream_fps_history.csv": _stream_csv(streams), "ai_fps_history.csv": _ai_csv(models),
             "telemetry.csv": _telemetry_csv(p.get("telemetry", []) or [])}
    omitted = []
    for name, src in (attachments or {}).items():
        try:
            src = Path(src)
            if src.stat().st_size > MAX_ATTACHMENT_BYTES:
                omitted.append(name)
                continue
            files[name] = src.read_bytes()
        except OSError:
            omitted.append(name)

    # ---- Cover / summary -------------------------------------------------
    story += [Paragraph("VX3 NDI STREAM QUALIFICATION", st["muted"]), _heading(title, st["title"], 0), Spacer(1, 6)]
    story.append(_table([
        ["Status", "Primary bottleneck", "Duration", "Streams", "Total frames", "Dropped", "Generated"],
        ["Attention Required" if a.get("observations") else "No Dominant Bottleneck", a.get("primary_bottleneck", "N/A"),
         fmt_duration(summary.get("actual_duration_seconds")), summary.get("streams", 0),
         f"{summary.get('total_frames', 0):,}", f"{summary.get('total_dropped', 0):,}", p.get("generated_at", "")],
    ], st))
    story += [Spacer(1, 10), Paragraph("<b>Exploring this PDF.</b> Charts are vector graphics: zoom in as far as your "
              "viewer allows and they stay sharp. Every chart line is its own <b>layer</b> - open the Layers panel "
              "(Adobe Acrobat/Reader, Foxit, PDF-XChange, Okular) to hide or show individual streams, models or "
              "telemetry series; browser built-in viewers may not offer layers. The <b>Attachments</b> panel contains "
              "the interactive HTML report (open in a browser for time-axis zoom, pan and hover values), "
              "results.json, and full-resolution CSV data for every chart. Bookmarks list every section and chart." +
              (f" <i>Not embedded because of size (kept next to this PDF): {escape(', '.join(omitted))}.</i>" if omitted else ""),
              st["body"])]

    story += [Spacer(1, 12), _heading("Bottleneck Analysis and Summary", st["h1"], 0),
              _p(a.get("summary", ""), st["body"]), _p(f"Confidence: {a.get('confidence', 'N/A')}", st["muted"])]
    obs = a.get("observations") or []
    if obs:
        story.append(_table([["Subsystem", "Severity", "Evidence", "Recommendation"]] +
                            [[o.get("subsystem"), o.get("severity"), o.get("evidence"), o.get("recommendation")] for o in obs],
                            st, [1.5 * inch, 0.8 * inch, 3.6 * inch, width - 5.9 * inch]))
    else:
        story.append(_p("No dominant threshold violation detected.", st["muted"]))

    # ---- Inventory ------------------------------------------------------
    inv = p.get("inventory", {}) or {}
    story += [CondPageBreak(2.5 * inch), _heading("Hardware and Software Inventory", st["h1"], 0)]
    story.append(_table([["Host", "OS", "CPU", "Cores (phys/logical)", "RAM", "Memory bus", "Python"],
                         [inv.get("hostname"), inv.get("os"), inv.get("cpu_model"),
                          f"{inv.get('physical_cpu_count', '?')} / {inv.get('logical_cpu_count', '?')}",
                          _gib(inv.get("ram_total_bytes")),
                          f"{inv['memory_bus_speed_mhz']:.0f} MT/s" if inv.get("memory_bus_speed_mhz") else "N/A",
                          inv.get("python")]], st))
    gpus = inv.get("gpus") or []
    if gpus:
        story += [Spacer(1, 6), _table([["GPU", "Name", "VRAM", "Driver"]] +
                                       [[g.get("index"), g.get("name"), _gib(g.get("vram_total_bytes")), g.get("driver_version") or "?"] for g in gpus], st)]
    disks = inv.get("disks") or []
    if disks:
        story += [Spacer(1, 6), _table([["Mount", "Device", "Filesystem", "Used", "Total", "Used %"]] +
                                       [[d.get("mountpoint"), d.get("device"), d.get("filesystem"), _gib(d.get("used_bytes")),
                                         _gib(d.get("total_bytes")), f"{d.get('percent', 0):.1f}%"] for d in disks], st)]

    # ---- Streams --------------------------------------------------------
    story += [CondPageBreak(3 * inch), _heading("Per-Stream Results", st["h1"], 0),
              _p(f"Configured stream FPS target: {fps_target:.2f} FPS. Min/Mean/Max are over 1 s intervals for the whole session.", st["muted"])]
    rows = [["Stream", "Received by", "Resolution", "Frames", "Avg FPS", "Min FPS", "Mean FPS", "Max FPS", "Rendered FPS", "Active", "State"]]
    marks = []
    for r, s in enumerate(streams, start=1):
        fps = s.get("fps_mean", s.get("fps"))
        rows.append([s.get("stream_id"), f"OBS ({s['obs_input_name']})" if s.get("obs_input_name") else "benchmark (synthetic)",
                     f"{s.get('width')}x{s.get('height')}" if s.get("width") else "N/A",
                     ("~" if s.get("frames_estimated") else "") + f"{s.get('frames', 0):,}", _n(s.get("fps")), _n(s.get("fps_min")),
                     _n(s.get("fps_mean")), _n(s.get("fps_max")), _n(s.get("obs_rendered_fps")),
                     fmt_duration(s.get("active_seconds")), s.get("state")])
        if fps is not None:
            bg = "#d7f5e7" if fps >= fps_target else "#fff0cc" if fps >= fps_target * 0.9 else "#ffdde1"
            marks.append((r, 6, colors.HexColor(bg)))
    ob = p.get("obs_ndi") or {}
    if ob:
        stt = ob.get("obs_version_stats") or {}
        story.append(_p(f"Received by OBS Studio (DistroAV), scene '{ob.get('scene')}', {ob.get('sources', 0)} source(s): OBS render "
                        f"{stt.get('activeFps', 'N/A')} FPS, skipped render frames {stt.get('renderSkippedFrames', 'N/A')}/"
                        f"{stt.get('renderTotalFrames', 'N/A')}, OBS CPU {stt.get('cpuUsage', 'N/A')}%." +
                        (f" Error: {ob['error']}" if ob.get("error") else ""), st["muted"]))
    modes = sorted({s.get("ndi_receive_mode") for s in streams if s.get("ndi_receive_mode")})
    if modes:
        story.append(_p("NDI receive: " + "; ".join(modes) + ".", st["muted"]))
    if streams:
        story.append(_table(rows, st, [2.0 * inch, 1.3 * inch, 0.8 * inch, 0.75 * inch, 0.6 * inch, 0.6 * inch, 0.65 * inch,
                                       0.6 * inch, 0.8 * inch, 0.8 * inch, width - 8.9 * inch], marks))
    else:
        story.append(_p("No streams were recorded.", st["muted"]))

    # ---- AI -------------------------------------------------------------
    story += [CondPageBreak(2 * inch), _heading("Independent AI Model Workload", st["h1"], 0)]
    if models:
        story.append(_p("Target = AI FPS per stream x AI-enabled streams. Instances are added one at a time (each paced to an "
                        "equal share of the target) until the total sustains the target, so 'Required' is the minimum count; "
                        "the search ends at the target, when an added instance raises the total by less than the minimum gain "
                        "(saturated), at the configured max processes per model, or if a new instance fails to start.", st["muted"]))
        labels = {"target_met": "Target met", "saturated": "Saturated", "instance_failed": "Instance failed to start", "max_instances": "Max processes reached",
                  "failed": "Failed", "scaling": "Still scaling", "fixed": "Fixed (no target)"}
        marks = []
        rows = [["Model", "Provider", "Target FPS", "Per stream x streams", "Achieved FPS", "% of target", "Instances",
                 "Required", "Result", "Mean ms", "Worst p95 ms"]]
        for r, m in enumerate(models, start=1):
            target = m.get("target_fps") or 0
            pct = 100 * (m.get("fps") or 0) / target if target else None
            rows.append([m.get("name"), m.get("active_device") or m.get("provider") or m.get("requested_device"), _n(target) if target else "None",
                         f"{_n(m.get('target_fps_per_stream'), 1)} x {m.get('stream_count', 0)}", _n(m.get("fps")),
                         "N/A" if pct is None else f"{pct:.1f}%", m.get("instance_count", 0), m.get("required_instances") or "-",
                         labels.get(m.get("state"), m.get("state")), _n(m.get("latency_mean_ms"), 3), _n(m.get("latency_p95_ms"), 3)])
            if target:
                marks.append((r, 8, colors.HexColor("#d7f5e7" if m.get("state") == "target_met" else "#ffdde1")))
        story.append(_table(rows, st, highlight=marks))
        for m in models:
            block = [_heading(f"{m.get('name')} instances", st["h2"], 1)]
            if m.get("scaling_reason"):
                block.append(_p(m["scaling_reason"], st["muted"]))
            if m.get("device_fallback"):
                text = f"Device fallback: {m['device_fallback']}."
                if m.get("device_fallback_detail"):
                    text += f" Steps tried: {m['device_fallback_detail']}."
                block.append(_p(text, st["muted"]))
            irows = [["Instance", "State", "Assigned FPS", "Avg FPS (lifetime)", "Capacity FPS", "Inferences", "Mean ms",
                      "p50 ms", "p95 ms", "p99 ms", "Max ms", "Cold start ms", "Note"]]
            for x in (m.get("instances") or []) + (m.get("retired_instances") or []):
                irows.append([f"#{x.get('instance')}", "stopped" if x.get("retired") else x.get("state"),
                              _n(x.get("assigned_fps") or x.get("target_rate_fps")), _n(x.get("fps")), _n(x.get("compute_fps"), 1),
                              f"{int(x.get('inferences') or 0):,}", _n(x.get("latency_mean_ms"), 3), _n(x.get("latency_p50_ms"), 3),
                              _n(x.get("latency_p95_ms"), 3), _n(x.get("latency_p99_ms"), 3), _n(x.get("latency_max_ms"), 3),
                              _n(x.get("cold_start_latency_ms"), 1),
                              x.get("saturation_note") or x.get("retired_reason") or x.get("error") or ""])
            block.append(_table(irows, st))
            steps = m.get("scaling_steps") or []
            if steps:
                tgt = m.get("target_fps") or 0
                block += [Spacer(1, 4), _table([["Instances", "Settled total FPS", "% of target"]] +
                                               [[x.get("instances"), _n(x.get("total_fps")),
                                                 f"{100 * (x.get('total_fps') or 0) / tgt:.1f}%" if tgt else "N/A"] for x in steps], st)]
            ev = m.get("scaling_events") or []
            if ev:
                block += [Spacer(1, 4), _table([["At", "Action", "Instances", "Total FPS", "Reason"]] +
                                               [[fmt_duration(x.get("elapsed_seconds")), x.get("action"), x.get("instances"),
                                                 _n(x.get("total_fps")), x.get("reason")] for x in ev], st,
                                               [0.8 * inch, 1.1 * inch, 0.8 * inch, 0.9 * inch, width - 3.6 * inch])]
            story += [Spacer(1, 6)] + block
    else:
        story.append(_p("No AI model workers were enabled for this session.", st["muted"]))

    # ---- Encode test & image persistence ------------------------------
    enc = p.get("encode_test")
    pers = p.get("image_persistence") or []
    story += [CondPageBreak(2 * inch), _heading("Encode Test & Image Persistence", st["h1"], 0)]
    if enc:
        streams = enc.get("streams") or []
        n, t = enc.get("stream_count") or len(streams), enc.get("target_fps") or 30
        kept = sum(1 for x in streams if (x.get("keep_up_ratio") or 0) >= 0.98)
        story.append(_p(f"Encode test: {n} x {enc.get('resolution')} H.264 stream(s), one process each, fed at {t} FPS. "
                        f"Encoder: {('GPU - ' + enc['gpu_encoder']) if enc.get('gpu_encoder') else 'CPU (libx264) - no working GPU encoder found'}. "
                        f"Total {_n(enc.get('total_fps'))} of {n * t} FPS; {kept} of {n} stream(s) kept up (>= 98% of frames due). "
                        "Encoded FPS comes from the encoder's own frame counter; 2 s warm-up excluded.", st["muted"]))
        rows = [["Stream", "Encoder", "Encoded FPS", "Kept up", "Frames", "Dropped", "Bitrate Mbps",
                 "Hand-off mean ms", "Hand-off p95 ms", "Hand-off max ms", "Note"]]
        marks = []
        for i, x in enumerate(streams, start=1):
            r = x.get("keep_up_ratio")
            rows.append([f"#{i}", x.get("encoder", ""), _n(x.get("fps")), "N/A" if r is None else f"{r * 100:.1f}%",
                         f"{int(x.get('frames_encoded') or 0):,}", f"{int(x.get('frames_dropped') or 0):,}",
                         _n(x.get("bitrate_mbps"), 1), _n(x.get("write_mean_ms")), _n(x.get("write_p95_ms")),
                         _n(x.get("write_max_ms")), x.get("fallback_reason") or x.get("error") or ""])
            if r is not None:
                marks.append((i, 3, colors.HexColor("#d7f5e7" if r >= 0.98 else "#ffdde1")))
        story.append(_table(rows, st, highlight=marks))
        if enc.get("error"):
            story.append(_p(enc["error"], st["muted"]))
        if enc.get("probe"):
            story.append(_p("GPU encoder detection: " + " | ".join(enc["probe"]), st["muted"]))
    if pers:
        story += [Spacer(1, 6), _p("Image persistence: one process per AI model saving the same image to the same file, "
                                   "driven by the model's measured inference rate (one queued save request per inference; "
                                   "requests older than 2 s are dropped). Only open/write/flush/fsync/close is timed.", st["muted"])]
        rows = [["Model", "Image", "Model FPS", "Saves/s", "Kept up", "Dropped", "Mean ms", "p95 ms", "p99 ms",
                 "Max ms", "Capacity FPS", "MB/s", "Saves"]]
        marks = []
        for x in pers:
            r = x.get("keep_up_ratio")
            rows.append([x.get("model"), f"{x.get('width')}x{x.get('height')} {str(x.get('format', '')).upper()}, "
                         f"{(x.get('image_bytes') or 0) / 1024:,.0f} KiB, fsync {'on' if x.get('fsync') else 'off'}",
                         _n(x.get("requested_fps_avg") or x.get("requested_fps")), _n(x.get("fps")), "N/A" if r is None else f"{r * 100:.1f}%",
                         f"{int(x.get('dropped_requests') or 0):,}", _n(x.get("latency_mean_ms"), 3),
                         _n(x.get("latency_p95_ms"), 3), _n(x.get("latency_p99_ms"), 3), _n(x.get("latency_max_ms"), 3),
                         _n(x.get("capacity_fps"), 1), _n(x.get("throughput_mb_per_s")), f"{int(x.get('operations') or 0):,}"])
            if r is not None:
                marks.append((len(rows) - 1, 4, colors.HexColor("#d7f5e7" if r >= 0.98 else "#ffdde1")))
        story.append(_table(rows, st, highlight=marks))
        for x in pers:
            story.append(_p(f"{x.get('model')}: saving to {x.get('image_path', '')}" + (f" - {x['error']}" if x.get("error") else ""), st["muted"]))
    if not enc and not pers:
        story.append(_p("The encode test and image persistence were disabled for this session.", st["muted"]))

    # ---- Telemetry summary ---------------------------------------------
    t = p.get("telemetry_summary", {}) or {}
    story += [CondPageBreak(3 * inch), _heading("Telemetry Summary", st["h1"], 0)]
    trows = [["Metric", "Mean", "Minimum", "Maximum", "Start", "End"]]

    def stat_row(label, stats, conv):
        trows.append([label] + [conv(stats.get(k)) for k in ("mean", "min", "max", "start", "end")])
    for key, value in t.items():
        if key == "gpus":
            continue
        label = key.replace("_", " ").title()
        conv = _gib if "bytes" in key else (lambda v: _n(v))
        if isinstance(value, dict) and value:
            stat_row(label, value, conv)
        elif not isinstance(value, dict):
            trows.append([label, conv(value), "", "", "", ""])
    story.append(_table(trows, st))
    from .report import GPU_SUMMARY_METRICS
    for i, g in enumerate(t.get("gpus", []) or []):
        grows = [["Metric", "Mean", "Minimum", "Maximum", "Start", "End"]]
        for key, label, unit in GPU_SUMMARY_METRICS:
            stats = g.get(key)
            if isinstance(stats, dict) and stats:
                conv = _gib if unit == "bytes" else (lambda v, u=unit: fmt_value(v, u))
                grows.append([label] + [conv(stats.get(k)) for k in ("mean", "min", "max", "start", "end")])
        story += [Spacer(1, 6), KeepTogether([_heading(f"{g.get('name') or 'GPU'} (GPU {i})", st["h2"], 1),
                                             _table(grows, st) if len(grows) > 1 else _p("No readings recorded.", st["muted"])])]

    # ---- Charts (one per landscape page) ------------------------------
    for section, specs in chart_sections:
        specs = [s for s in specs if s]
        if not specs:
            continue
        story += [PageBreak(), _heading(f"{section} Charts", st["h1"], 0)]
        for k, spec in enumerate(specs):
            if k:
                story.append(PageBreak())
            story.append(_heading(spec["title"], st["h2"], 1))
            note = spec.get("pdf_note")
            if note:
                story.append(_p(note, st["muted"]))
            story.append(ChartFlowable(spec, width, chart_h - (24 if k == 0 else 0) - (22 if note else 0), registry))

    # ---- Configuration --------------------------------------------------
    story += [PageBreak(), _heading("Benchmark Configuration", st["h1"], 0),
              _p("The settings this session ran with. The exact machine-readable configuration is attached to this PDF "
                 "as config.json.", st["muted"]), Spacer(1, 6)]
    story += _config_story(p.get("config", {}) or {}, st, width)

    tmp = path.with_suffix(".tmp.pdf")
    _Doc(str(tmp), title).build(story)

    try:
        _finalize(tmp, path, registry, files, title)
    finally:
        tmp.unlink(missing_ok=True)
    return path
